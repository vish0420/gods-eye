from __future__ import annotations

from pathlib import Path

from godseye.domain import BoundingBox, Direction, TrackSummary, Zone
from godseye.storage.event_log import read_jsonl


def load_track_summaries(path: str | Path) -> list[TrackSummary]:
    records = read_jsonl(path)
    return [_track_summary_from_dict(record) for record in records]


def _track_summary_from_dict(record: dict[str, object]) -> TrackSummary:
    return TrackSummary(
        track_id=int(record["track_id"]),
        camera_id=str(record["camera_id"]),
        entry_time_s=float(record["entry_time_s"]),
        exit_time_s=float(record["exit_time_s"]),
        first_frame_index=int(record["first_frame_index"]),
        last_frame_index=int(record["last_frame_index"]),
        detection_count=int(record["detection_count"]),
        entry_bbox=_bbox_from_dict(record["entry_bbox"]),
        exit_bbox=_bbox_from_dict(record["exit_bbox"]),
        direction=Direction(str(record.get("direction", Direction.UNKNOWN.value))),
        entry_zone=Zone(str(record.get("entry_zone", Zone.UNKNOWN.value))),
        exit_zone=Zone(str(record.get("exit_zone", Zone.UNKNOWN.value))),
        average_confidence=float(record.get("average_confidence", 0.0)),
        appearance_embedding=tuple(
            float(value) for value in record.get("appearance_embedding", [])
        ),
        face_embedding=tuple(float(value) for value in record.get("face_embedding", [])),
    )


def _bbox_from_dict(value: object) -> BoundingBox:
    if not isinstance(value, dict):
        raise ValueError(f"Expected bounding box dictionary, got {type(value).__name__}")

    return BoundingBox(
        x1=float(value["x1"]),
        y1=float(value["y1"]),
        x2=float(value["x2"]),
        y2=float(value["y2"]),
    )
