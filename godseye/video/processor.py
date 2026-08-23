from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass

from godseye.detection.yolo_detector import YoloPersonDetector, draw_detections
from godseye.domain import Detection, TrackSummary
from godseye.storage.event_log import write_jsonl
from godseye.tracking import SimplePersonTracker, remap_detection_ids


@dataclass(frozen=True)
class VideoProcessingResult:
    detections: list[Detection]
    tracks: list[TrackSummary]


class VideoDetectionProcessor:
    def __init__(self, detector: YoloPersonDetector) -> None:
        self.detector = detector

    def process(
        self,
        video_path: str | Path,
        camera_id: str,
        every_n_frames: int = 1,
        max_frames: int | None = None,
        output_jsonl: str | Path | None = None,
        annotated_output: str | Path | None = None,
        min_track_detections: int = 3,
        min_track_confidence: float = 0.60,
        show: bool = False,
        window_name: str = "God's Eye Tracking",
        show_delay_ms: int = 1,
        hold_window: bool = False,
    ) -> VideoProcessingResult:
        cv2 = _import_cv2()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = _make_writer(cv2, annotated_output, fps, width, height)
        if show:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if width > 0 and height > 0:
                display_width = min(width, 960)
                display_height = int(height * (display_width / width))
                cv2.resizeWindow(window_name, display_width, display_height)

        detections: list[Detection] = []
        last_display_detections: list[Detection] = []
        closed_by_user = False
        tracker = SimplePersonTracker(
            iou_threshold=0.03,
            max_missed_frames=max(30, every_n_frames * 6),
            min_detections=min_track_detections,
            min_average_confidence=min_track_confidence,
        )
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if max_frames is not None and frame_index >= max_frames:
                    break

                frame_detections: list[Detection] = []
                if frame_index % max(1, every_n_frames) == 0:
                    timestamp_s = frame_index / fps
                    frame_detections = self.detector.detect_frame(
                        frame=frame,
                        camera_id=camera_id,
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                    )
                    frame_detections = tracker.update(
                        frame_detections,
                        frame_width=width,
                        frame_height=height,
                        frame=frame,
                    )
                    detections.extend(frame_detections)
                    last_display_detections = frame_detections

                display_detections = frame_detections or last_display_detections
                annotated_frame = draw_detections(frame.copy(), display_detections)

                if writer is not None:
                    writer.write(annotated_frame)

                if show:
                    cv2.imshow(window_name, annotated_frame)
                    key = cv2.waitKey(max(1, show_delay_ms)) & 0xFF
                    if key == ord("q"):
                        closed_by_user = True
                        break

                frame_index += 1
        finally:
            if show and hold_window and not closed_by_user:
                cv2.waitKey(0)
            capture.release()
            if writer is not None:
                writer.release()
            if show:
                _close_window_and_verify(cv2, window_name)

        tracks = tracker.finish(frame_width=width, frame_height=height)
        detections = remap_detection_ids(detections, tracker.track_id_mapping)

        if output_jsonl is not None:
            write_jsonl(output_jsonl, detections)

        return VideoProcessingResult(detections=detections, tracks=tracks)


def _make_writer(
    cv2: object,
    output_path: str | Path | None,
    fps: float,
    width: int,
    height: int,
) -> object | None:
    if output_path is None:
        return None

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(target), fourcc, fps, (width, height))


def play_video(video_path: str | Path, window_name: str = "God's Eye Output") -> None:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    delay_ms = max(1, int(1000 / fps))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        _close_window_and_verify(cv2, window_name)


def _import_cv2() -> object:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video processing. Install it with: "
            "pip install -r requirements.txt"
        ) from exc

    return cv2


def _close_window_and_verify(cv2: object, window_name: str, timeout_s: float = 2.0) -> None:
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return
        if visible < 1:
            return
        cv2.waitKey(1)
        time.sleep(0.05)

    raise RuntimeError(f"Video window did not close cleanly: {window_name}")
