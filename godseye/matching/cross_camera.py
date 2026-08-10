from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from godseye.domain import MatchDecision, TrackSegment, TrackSummary
from godseye.matching.camera_flow import CameraFlowGraph
from godseye.storage.event_log import write_jsonl
from godseye.storage.track_io import load_track_summaries


@dataclass(frozen=True)
class CrossCameraReport:
    decisions: list[MatchDecision]
    text: str


def compare_track_files(
    config_path: str | Path,
    source_tracks_path: str | Path,
    target_tracks_path: str | Path,
    top_k: int = 3,
    source_person_id: int | None = None,
) -> CrossCameraReport:
    graph = CameraFlowGraph.from_json(config_path)
    source_tracks = load_track_summaries(source_tracks_path)
    if source_person_id is not None:
        source_tracks = [track for track in source_tracks if track.track_id == source_person_id]
    target_tracks = load_track_summaries(target_tracks_path)
    return compare_tracks(
        graph=graph,
        source_tracks=source_tracks,
        target_tracks=target_tracks,
        source_name=str(source_tracks_path),
        target_name=str(target_tracks_path),
        top_k=top_k,
    )


def compare_tracks(
    graph: CameraFlowGraph,
    source_tracks: list[TrackSummary],
    target_tracks: list[TrackSummary],
    source_name: str = "source",
    target_name: str = "target",
    top_k: int = 3,
) -> CrossCameraReport:
    decisions: list[MatchDecision] = []
    lines = [
        "Cross-Camera Match Report",
        f"Source Tracks: {source_name}",
        f"Target Tracks: {target_name}",
        f"Source Count: {len(source_tracks)}",
        f"Target Count: {len(target_tracks)}",
        "",
    ]

    if not source_tracks:
        lines.append("No source tracks found.")
        return CrossCameraReport(decisions=[], text="\n".join(lines) + "\n")
    if not target_tracks:
        lines.append("No target tracks found.")
        return CrossCameraReport(decisions=[], text="\n".join(lines) + "\n")

    target_segments = [_summary_to_segment(track) for track in target_tracks]
    target_by_id = {str(track.track_id): track for track in target_tracks}

    for source_track in source_tracks:
        source_segment = _summary_to_segment(source_track)
        ranked = graph.rank_candidates(source_segment, target_segments)[:top_k]
        decisions.extend(ranked)

        lines.extend(
            [
                f"Source Person ID: {source_track.track_id}",
                f"Camera: {source_track.camera_id}",
                f"Exit Time: {source_track.exit_time_s:.1f} sec",
                f"Exit Zone: {source_track.exit_zone.value}",
                "",
            ]
        )

        if not ranked:
            lines.extend(["No candidates found.", ""])
            continue

        for index, decision in enumerate(ranked, start=1):
            target_track = target_by_id.get(decision.candidate_track_id)
            entry_time = target_track.entry_time_s if target_track else 0.0
            entry_zone = target_track.entry_zone.value if target_track else "unknown"
            lines.extend(
                [
                    f"Candidate {index}: Person ID {decision.candidate_track_id}",
                    f"Target Camera: {decision.to_camera}",
                    f"Entry Time: {entry_time:.1f} sec",
                    f"Entry Zone: {entry_zone}",
                    f"Score: {decision.score * 100:.1f}%",
                    f"Decision: {_format_decision(decision.decision)}",
                    f"Reason: {', '.join(decision.reasons)}",
                    "",
                ]
            )

    return CrossCameraReport(decisions=decisions, text="\n".join(lines))


def write_cross_camera_report(
    report: CrossCameraReport,
    text_path: str | Path,
    jsonl_path: str | Path | None = None,
) -> None:
    target = Path(text_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.text, encoding="utf-8")

    if jsonl_path is not None:
        write_jsonl(jsonl_path, report.decisions)


def _summary_to_segment(summary: TrackSummary) -> TrackSegment:
    return TrackSegment(
        track_id=str(summary.track_id),
        camera_id=summary.camera_id,
        entry_time_s=summary.entry_time_s,
        exit_time_s=summary.exit_time_s,
        entry_bbox=summary.entry_bbox,
        exit_bbox=summary.exit_bbox,
        entry_zone=summary.entry_zone,
        exit_zone=summary.exit_zone,
        appearance_score=summary.average_confidence,
        metadata={
            "appearance_embedding": summary.appearance_embedding,
            "face_embedding": summary.face_embedding,
        },
    )


def _format_decision(decision: str) -> str:
    if decision == "auto_match":
        return "AUTO MATCH"
    if decision == "ask_user":
        return "ASK USER TO CONFIRM"
    return "REJECT"
