from __future__ import annotations

import json
from pathlib import Path

from godseye.domain import CameraLink, MatchDecision, TrackSegment, Zone


class CameraFlowGraph:
    def __init__(self, links: list[CameraLink]) -> None:
        self.links = links

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraFlowGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        links = [
            CameraLink(
                from_camera=item["from_camera"],
                to_camera=item["to_camera"],
                exit_zone=Zone(item["exit_zone"]),
                entry_zone=Zone(item["entry_zone"]),
                min_travel_s=float(item["min_travel_s"]),
                max_travel_s=float(item["max_travel_s"]),
                note=str(item.get("note", "")),
            )
            for item in data.get("links", [])
        ]
        return cls(links)

    def expected_links(self, camera_id: str, exit_zone: Zone) -> list[CameraLink]:
        return [
            link
            for link in self.links
            if link.from_camera == camera_id
            and link.exit_zone in (Zone.UNKNOWN, exit_zone)
        ]

    def score_transition(
        self,
        source: TrackSegment,
        candidate: TrackSegment,
        appearance_score: float | None = None,
    ) -> MatchDecision:
        link = self._find_link(source, candidate)
        if link is None:
            return MatchDecision(
                source_track_id=source.track_id,
                candidate_track_id=candidate.track_id,
                from_camera=source.camera_id,
                to_camera=candidate.camera_id,
                score=0.0,
                decision="reject",
                reasons=("no configured camera link",),
            )

        gap_s = candidate.entry_time_s - source.exit_time_s
        time_score = _score_time_gap(gap_s, link.min_travel_s, link.max_travel_s)
        zone_score = _score_zone(source, candidate, link)
        appearance = _appearance_similarity(source, candidate, appearance_score)

        score = (0.45 * time_score) + (0.25 * zone_score) + (0.30 * appearance)
        decision = _decision(score)
        expected_arrival = source.exit_time_s + link.min_travel_s
        reasons = (
            f"expected_arrival_s={expected_arrival:.2f}",
            f"time_score={time_score:.2f}",
            f"zone_score={zone_score:.2f}",
            f"appearance_score={appearance:.2f}",
            f"gap_s={gap_s:.2f}",
        )

        return MatchDecision(
            source_track_id=source.track_id,
            candidate_track_id=candidate.track_id,
            from_camera=source.camera_id,
            to_camera=candidate.camera_id,
            score=round(score, 4),
            decision=decision,
            reasons=reasons,
        )

    def rank_candidates(
        self,
        source: TrackSegment,
        candidates: list[TrackSegment],
    ) -> list[MatchDecision]:
        decisions = [self.score_transition(source, candidate) for candidate in candidates]
        return sorted(decisions, key=lambda item: item.score, reverse=True)

    def _find_link(
        self,
        source: TrackSegment,
        candidate: TrackSegment,
    ) -> CameraLink | None:
        for link in self.links:
            if link.from_camera != source.camera_id:
                continue
            if link.to_camera != candidate.camera_id:
                continue
            if link.exit_zone != Zone.UNKNOWN and source.exit_zone not in (Zone.UNKNOWN, link.exit_zone):
                continue
            return link
        return None


def _score_time_gap(gap_s: float, min_s: float, max_s: float) -> float:
    if min_s <= gap_s <= max_s:
        return 1.0
    if gap_s < 0:
        return 0.0

    early_tolerance = max(1.0, min_s)
    late_tolerance = max(2.0, max_s * 0.5)

    if gap_s < min_s:
        return max(0.0, 1.0 - ((min_s - gap_s) / early_tolerance))
    return max(0.0, 1.0 - ((gap_s - max_s) / late_tolerance))


def _score_zone(source: TrackSegment, candidate: TrackSegment, link: CameraLink) -> float:
    score = 0.0

    if source.exit_zone in (Zone.UNKNOWN, link.exit_zone):
        score += 0.5
    if candidate.entry_zone in (Zone.UNKNOWN, link.entry_zone):
        score += 0.5

    return score


def _normalize_score(score: float | None) -> float:
    if score is None:
        return 0.5
    return max(0.0, min(1.0, float(score)))


def _appearance_similarity(
    source: TrackSegment,
    candidate: TrackSegment,
    fallback: float | None,
) -> float:
    source_face = source.metadata.get("face_embedding", ())
    candidate_face = candidate.metadata.get("face_embedding", ())
    if source_face and candidate_face and len(source_face) == len(candidate_face):
        distance = sum((float(a) - float(b)) ** 2 for a, b in zip(source_face, candidate_face)) ** 0.5
        return _normalize_score(1.0 - (distance / 0.60))
    source_vector = source.metadata.get("appearance_embedding", ())
    candidate_vector = candidate.metadata.get("appearance_embedding", ())
    if source_vector and candidate_vector and len(source_vector) == len(candidate_vector):
        score = sum(float(a) * float(b) for a, b in zip(source_vector, candidate_vector))
        return _normalize_score(score)
    return _normalize_score(fallback if fallback is not None else candidate.appearance_score)


def _decision(score: float) -> str:
    if score >= 0.85:
        return "auto_match"
    if score >= 0.60:
        return "ask_user"
    return "reject"
