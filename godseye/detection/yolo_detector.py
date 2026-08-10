from __future__ import annotations

from pathlib import Path
from typing import Any

from godseye.domain import BoundingBox, Detection, Direction, PERSON_CLASS_ID


class YoloPersonDetector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.35,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model = self._load_model(model_path)

    def detect_image(
        self,
        image_path: str | Path,
        camera_id: str = "image",
        annotated_path: str | Path | None = None,
    ) -> list[Detection]:
        cv2 = _import_cv2()
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        detections = self.detect_frame(
            frame=frame,
            camera_id=camera_id,
            frame_index=0,
            timestamp_s=0.0,
        )
        if annotated_path is not None:
            annotated = draw_detections(frame.copy(), detections)
            Path(annotated_path).parent.mkdir(parents=True, exist_ok=True)
            saved = cv2.imwrite(str(annotated_path), annotated)
            if not saved:
                raise RuntimeError(f"Could not save annotated image: {annotated_path}")

        return detections

    def detect_frame(
        self,
        frame: Any,
        camera_id: str,
        frame_index: int,
        timestamp_s: float,
    ) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0].item())
                if class_id != PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    Detection(
                        camera_id=camera_id,
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=float(box.conf[0].item()),
                        class_id=class_id,
                        label="person",
                    )
                )

        return detections

    @staticmethod
    def _load_model(model_path: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for detection. Install it with: "
                "pip install -r requirements.txt"
            ) from exc

        return YOLO(model_path)


def draw_detections(frame: Any, detections: list[Detection]) -> Any:
    cv2 = _import_cv2()

    for detection in detections:
        bbox = detection.bbox
        x1, y1, x2, y2 = map(int, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        label_parts = []
        if detection.track_id is not None:
            label_parts.append(f"ID {detection.track_id}")
        label_parts.append(detection.label)
        if detection.direction != Direction.UNKNOWN:
            label_parts.append(detection.direction.value)
        label_parts.append(f"{detection.confidence:.2f}")
        label = " ".join(label_parts)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
        _draw_direction_arrow(cv2, frame, detection)

    return frame


def _draw_direction_arrow(cv2: Any, frame: Any, detection: Detection) -> None:
    if detection.direction not in {
        Direction.LEFT,
        Direction.RIGHT,
        Direction.UP,
        Direction.DOWN,
    }:
        return

    bbox = detection.bbox
    start = (int(bbox.center_x), int(bbox.center_y))
    arrow_length = 45
    dx, dy = {
        Direction.LEFT: (-arrow_length, 0),
        Direction.RIGHT: (arrow_length, 0),
        Direction.UP: (0, -arrow_length),
        Direction.DOWN: (0, arrow_length),
    }[detection.direction]
    end = (start[0] + dx, start[1] + dy)
    cv2.arrowedLine(frame, start, end, (0, 220, 255), 3, tipLength=0.35)


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for image and video processing. Install it with: "
            "pip install -r requirements.txt"
        ) from exc

    return cv2
