from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


PERSON_CLASS_ID = 0


class Direction(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    STATIONARY = "stationary"
    UNKNOWN = "unknown"


class Zone(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return self.x1 + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y1 + self.height / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class Detection:
    camera_id: str
    frame_index: int
    timestamp_s: float
    bbox: BoundingBox
    confidence: float
    class_id: int = PERSON_CLASS_ID
    label: str = "person"
    track_id: int | None = None
    direction: Direction = Direction.UNKNOWN
    snapshot_path: str | None = None


@dataclass(frozen=True)
class TrackSegment:
    track_id: str
    camera_id: str
    entry_time_s: float
    exit_time_s: float
    entry_bbox: BoundingBox | None = None
    exit_bbox: BoundingBox | None = None
    entry_zone: Zone = Zone.UNKNOWN
    exit_zone: Zone = Zone.UNKNOWN
    appearance_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.exit_time_s - self.entry_time_s)


@dataclass(frozen=True)
class TrackSummary:
    track_id: int
    camera_id: str
    entry_time_s: float
    exit_time_s: float
    first_frame_index: int
    last_frame_index: int
    detection_count: int
    entry_bbox: BoundingBox
    exit_bbox: BoundingBox
    direction: Direction = Direction.UNKNOWN
    entry_zone: Zone = Zone.UNKNOWN
    exit_zone: Zone = Zone.UNKNOWN
    average_confidence: float = 0.0
    appearance_embedding: tuple[float, ...] = ()
    face_embedding: tuple[float, ...] = ()

    @property
    def duration_s(self) -> float:
        return max(0.0, self.exit_time_s - self.entry_time_s)


@dataclass(frozen=True)
class CameraLink:
    from_camera: str
    to_camera: str
    exit_zone: Zone
    entry_zone: Zone
    min_travel_s: float
    max_travel_s: float
    note: str = ""


@dataclass(frozen=True)
class MatchDecision:
    source_track_id: str
    candidate_track_id: str
    from_camera: str
    to_camera: str
    score: float
    decision: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TimelineEvent:
    person_id: str
    camera_id: str
    entry_time_s: float
    exit_time_s: float
    direction: Direction = Direction.UNKNOWN
    confidence: float = 1.0
    source_track_id: str | None = None
