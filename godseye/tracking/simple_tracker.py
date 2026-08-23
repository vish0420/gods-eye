from __future__ import annotations

from dataclasses import dataclass, field, replace

from godseye.domain import BoundingBox, Detection, Direction, TrackSummary
from godseye.movement import infer_direction, infer_zone


@dataclass
class _ActiveTrack:
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    missed_frames: int = 0
    appearance_samples: list[tuple[float, ...]] = field(default_factory=list)
    face_samples: list[tuple[float, ...]] = field(default_factory=list)

    @property
    def last_detection(self) -> Detection:
        return self.detections[-1]


class SimplePersonTracker:
    """Small IoU tracker for demo videos.

    It is intentionally lightweight: good enough to count simple recorded demos,
    while ByteTrack/DeepSORT can replace it later for crowded scenes.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_missed_frames: int = 30,
        min_detections: int = 3,
        min_average_confidence: float = 0.60,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.min_detections = min_detections
        self.min_average_confidence = min_average_confidence
        self._next_track_id = 1
        self._active: list[_ActiveTrack] = []
        self._finished: list[_ActiveTrack] = []
        self.track_id_mapping: dict[int, int] = {}

    def update(
        self,
        detections: list[Detection],
        frame_width: int | None = None,
        frame_height: int | None = None,
        frame: object | None = None,
    ) -> list[Detection]:
        for track in self._active:
            track.missed_frames += 1

        assigned_track_ids: set[int] = set()
        assigned_detection_indexes: set[int] = set()
        updated_detections: list[Detection] = []

        matches = self._match_detections(detections)
        for track_index, detection_index in matches:
            track = self._active[track_index]
            detection = detections[detection_index]
            direction = _current_direction(
                previous=track.last_detection.bbox,
                current=detection.bbox,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            tracked_detection = _with_track_id(detection, track.track_id, direction)
            track.detections.append(tracked_detection)
            appearance = _appearance_embedding(frame, detection.bbox)
            if appearance:
                track.appearance_samples.append(appearance)
            if len(track.face_samples) < 3:
                face = _face_embedding(frame, detection.bbox)
                if face:
                    track.face_samples.append(face)
            track.missed_frames = 0
            assigned_track_ids.add(track.track_id)
            assigned_detection_indexes.add(detection_index)
            updated_detections.append(tracked_detection)

        for index, detection in enumerate(detections):
            if index in assigned_detection_indexes:
                continue
            track = _ActiveTrack(track_id=self._next_track_id)
            self._next_track_id += 1
            tracked_detection = _with_track_id(detection, track.track_id, Direction.UNKNOWN)
            track.detections.append(tracked_detection)
            appearance = _appearance_embedding(frame, detection.bbox)
            if appearance:
                track.appearance_samples.append(appearance)
            if len(track.face_samples) < 3:
                face = _face_embedding(frame, detection.bbox)
                if face:
                    track.face_samples.append(face)
            self._active.append(track)
            assigned_track_ids.add(track.track_id)
            updated_detections.append(tracked_detection)

        still_active: list[_ActiveTrack] = []
        for track in self._active:
            if (
                track.track_id not in assigned_track_ids
                and track.missed_frames > self.max_missed_frames
            ):
                self._finished.append(track)
            else:
                still_active.append(track)
        self._active = still_active

        return sorted(updated_detections, key=lambda item: item.track_id or 0)

    def finish(self, frame_width: int, frame_height: int) -> list[TrackSummary]:
        all_tracks = [*self._finished, *self._active]
        summaries: list[TrackSummary] = []
        for track in all_tracks:
            if len(track.detections) < self.min_detections:
                continue

            summary = _summarize_track(track, frame_width, frame_height)
            if summary.average_confidence < self.min_average_confidence:
                continue

            summaries.append(summary)

        merged = _merge_duplicate_summaries(summaries, frame_width, frame_height)
        self.track_id_mapping = _build_id_mapping(summaries, merged)
        return merged

    def _match_detections(self, detections: list[Detection]) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._active):
            for detection_index, detection in enumerate(detections):
                score = match_score(track.last_detection.bbox, detection.bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, track_index, detection_index))

        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            matches.append((track_index, detection_index))

        return matches


def iou(first: BoundingBox, second: BoundingBox) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection_area = intersection_width * intersection_height
    union_area = first.area + second.area - intersection_area

    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def match_score(first: BoundingBox, second: BoundingBox) -> float:
    overlap_score = iou(first, second)
    center_score = _center_score(first, second)
    return max(overlap_score, center_score)


def _center_score(first: BoundingBox, second: BoundingBox) -> float:
    dx = first.center_x - second.center_x
    dy = first.center_y - second.center_y
    distance = (dx * dx + dy * dy) ** 0.5
    max_distance = max(first.width, first.height, second.width, second.height) * 1.5

    if max_distance <= 0:
        return 0.0
    return max(0.0, 1.0 - (distance / max_distance))


def _current_direction(
    previous: BoundingBox,
    current: BoundingBox,
    frame_width: int | None,
    frame_height: int | None,
) -> Direction:
    if frame_width is None or frame_height is None:
        return Direction.UNKNOWN
    return infer_direction(
        previous,
        current,
        frame_width,
        frame_height,
        min_delta_ratio=0.015,
    )


def _with_track_id(
    detection: Detection,
    track_id: int,
    direction: Direction,
) -> Detection:
    return Detection(
        camera_id=detection.camera_id,
        frame_index=detection.frame_index,
        timestamp_s=detection.timestamp_s,
        bbox=detection.bbox,
        confidence=detection.confidence,
        class_id=detection.class_id,
        label=detection.label,
        track_id=track_id,
        direction=direction,
        snapshot_path=detection.snapshot_path,
    )


def _summarize_track(
    track: _ActiveTrack,
    frame_width: int,
    frame_height: int,
) -> TrackSummary:
    first = track.detections[0]
    last = track.detections[-1]
    average_confidence = sum(item.confidence for item in track.detections) / len(
        track.detections
    )

    return TrackSummary(
        track_id=track.track_id,
        camera_id=first.camera_id,
        entry_time_s=first.timestamp_s,
        exit_time_s=last.timestamp_s,
        first_frame_index=first.frame_index,
        last_frame_index=last.frame_index,
        detection_count=len(track.detections),
        entry_bbox=first.bbox,
        exit_bbox=last.bbox,
        direction=infer_direction(first.bbox, last.bbox, frame_width, frame_height),
        entry_zone=infer_zone(first.bbox, frame_width, frame_height),
        exit_zone=infer_zone(last.bbox, frame_width, frame_height),
        average_confidence=round(average_confidence, 4),
        appearance_embedding=_average_embedding(track.appearance_samples),
        face_embedding=_average_embedding(track.face_samples),
    )


def _appearance_embedding(frame: object | None, bbox: BoundingBox) -> tuple[float, ...]:
    """Create a compact clothing-colour signature from the middle of a person crop."""
    if frame is None:
        return ()
    try:
        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        x1, x2 = max(0, int(bbox.x1)), min(width, int(bbox.x2))
        y1, y2 = max(0, int(bbox.y1)), min(height, int(bbox.y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ()
        # Avoid face and shoes: the torso is most useful for clothing colour.
        top, bottom = int(crop.shape[0] * 0.20), int(crop.shape[0] * 0.75)
        torso = crop[top:bottom, :]
        if torso.size == 0:
            torso = crop
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
        vector = histogram.flatten().astype("float32")
        norm = float(np.linalg.norm(vector))
        return tuple((vector / norm).tolist()) if norm else ()
    except (ImportError, AttributeError, TypeError):
        return ()


def _average_embedding(samples: list[tuple[float, ...]]) -> tuple[float, ...]:
    if not samples:
        return ()
    usable = [sample for sample in samples if sample]
    if not usable:
        return ()
    length = len(usable[0])
    compatible = [sample for sample in usable if len(sample) == length]
    averaged = [sum(sample[index] for sample in compatible) / len(compatible) for index in range(length)]
    norm = sum(value * value for value in averaged) ** 0.5
    return tuple(value / norm for value in averaged) if norm else ()


def _merge_duplicate_summaries(
    summaries: list[TrackSummary],
    frame_width: int,
    frame_height: int,
) -> list[TrackSummary]:
    """Join likely ID fragments caused by a brief detector/tracker loss."""
    pending = sorted(summaries, key=lambda item: item.entry_time_s)
    merged: list[TrackSummary] = []
    while pending:
        current = pending.pop(0)
        next_index = next(
            (
                index
                for index, candidate in enumerate(pending)
                if _should_merge_tracks(current, candidate)
            ),
            None,
        )
        if next_index is None:
            merged.append(current)
            continue
        current = _combine_tracks(current, pending.pop(next_index), frame_width, frame_height)
        pending.insert(0, current)
    return sorted(merged, key=lambda item: item.track_id)


def _should_merge_tracks(first: TrackSummary, second: TrackSummary) -> bool:
    if first.camera_id != second.camera_id:
        return False
    # A detector can create the new ID a few frames before it retires the old one.
    if second.entry_time_s < first.exit_time_s - 0.5:
        return False
    # A lost track should reappear quickly. Longer gaps may be different people.
    if second.entry_time_s - first.exit_time_s > 6.0:
        return False
    face_score = _cosine_similarity(first.face_embedding, second.face_embedding)
    appearance_score = _cosine_similarity(first.appearance_embedding, second.appearance_embedding)
    return face_score >= 0.72 or appearance_score >= 0.92


def _build_id_mapping(
    original: list[TrackSummary], merged: list[TrackSummary]
) -> dict[int, int]:
    """Map every short-lived tracker ID to the surviving merged ID."""
    mapping: dict[int, int] = {}
    for track in original:
        match = next(
            (
                candidate
                for candidate in merged
                if candidate.camera_id == track.camera_id
                and candidate.entry_time_s <= track.entry_time_s
                and candidate.exit_time_s >= track.exit_time_s
            ),
            track,
        )
        mapping[track.track_id] = match.track_id
    return mapping


def remap_detection_ids(
    detections: list[Detection], id_mapping: dict[int, int]
) -> list[Detection]:
    """Apply merged IDs to every saved frame detection."""
    return [
        replace(item, track_id=id_mapping.get(item.track_id, item.track_id))
        if item.track_id is not None
        else item
        for item in detections
    ]


def _cosine_similarity(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    if not first or not second or len(first) != len(second):
        return 0.0
    return sum(a * b for a, b in zip(first, second))


def _combine_tracks(
    first: TrackSummary,
    second: TrackSummary,
    frame_width: int,
    frame_height: int,
) -> TrackSummary:
    count = first.detection_count + second.detection_count
    confidence = (
        (first.average_confidence * first.detection_count)
        + (second.average_confidence * second.detection_count)
    ) / count
    return TrackSummary(
        track_id=first.track_id,
        camera_id=first.camera_id,
        entry_time_s=first.entry_time_s,
        exit_time_s=second.exit_time_s,
        first_frame_index=first.first_frame_index,
        last_frame_index=second.last_frame_index,
        detection_count=count,
        entry_bbox=first.entry_bbox,
        exit_bbox=second.exit_bbox,
        direction=infer_direction(first.entry_bbox, second.exit_bbox, frame_width, frame_height),
        entry_zone=first.entry_zone,
        exit_zone=second.exit_zone,
        average_confidence=round(confidence, 4),
        appearance_embedding=_combine_embeddings(first.appearance_embedding, second.appearance_embedding),
        face_embedding=_combine_embeddings(first.face_embedding, second.face_embedding),
    )


def _combine_embeddings(first: tuple[float, ...], second: tuple[float, ...]) -> tuple[float, ...]:
    if not first:
        return second
    if not second or len(first) != len(second):
        return first
    return _average_embedding([first, second])


def _face_embedding(frame: object | None, bbox: BoundingBox) -> tuple[float, ...]:
    """Return a face vector when an optional local face-recognition install can see one."""
    if frame is None:
        return ()
    try:
        import face_recognition
    except ImportError:
        return ()
    try:
        height, width = frame.shape[:2]
        x1, x2 = max(0, int(bbox.x1)), min(width, int(bbox.x2))
        y1, y2 = max(0, int(bbox.y1)), min(height, int(bbox.y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ()
        rgb = crop[:, :, ::-1]
        encodings = face_recognition.face_encodings(rgb)
        return tuple(encodings[0].tolist()) if encodings else ()
    except (AttributeError, IndexError, ValueError):
        return ()
