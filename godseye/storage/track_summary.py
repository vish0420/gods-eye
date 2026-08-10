from __future__ import annotations

from pathlib import Path

from godseye.domain import TrackSummary


def format_track_summary(camera_id: str, tracks: list[TrackSummary]) -> str:
    lines = [
        f"Camera: {camera_id}",
        f"Unique Persons: {len(tracks)}",
        "",
    ]

    if not tracks:
        lines.append("No persons detected.")
        return "\n".join(lines) + "\n"

    for track in tracks:
        lines.extend(
            [
                f"Person ID: {track.track_id}",
                f"Entered: {_format_seconds(track.entry_time_s)}",
                f"Exited: {_format_seconds(track.exit_time_s)}",
                f"Duration: {_format_seconds(track.duration_s)}",
                f"Direction: {track.direction.value}",
                f"Entry Zone: {track.entry_zone.value}",
                f"Exit Zone: {track.exit_zone.value}",
                f"Detections: {track.detection_count}",
                f"Average Confidence: {track.average_confidence:.2f}",
                "",
            ]
        )

    return "\n".join(lines)


def write_track_summary(
    path: str | Path,
    camera_id: str,
    tracks: list[TrackSummary],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_track_summary(camera_id, tracks), encoding="utf-8")


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.1f} sec"

