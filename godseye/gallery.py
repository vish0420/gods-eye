from __future__ import annotations

from pathlib import Path

from godseye.storage.event_log import read_jsonl


def create_person_gallery(
    video_path: str | Path,
    detections_path: str | Path,
    output_path: str | Path,
    allowed_track_ids: set[int] | None = None,
) -> int:
    """Save one labelled crop for each tracked person in a recorded video."""
    import cv2

    best_by_id: dict[int, dict[str, object]] = {}
    for record in read_jsonl(detections_path):
        track_id = record.get("track_id")
        if track_id is None:
            continue
        if allowed_track_ids is not None and int(track_id) not in allowed_track_ids:
            continue
        current = best_by_id.get(int(track_id))
        if current is None or float(record["confidence"]) > float(current["confidence"]):
            best_by_id[int(track_id)] = record

    if not best_by_id:
        raise ValueError("No tracked people found. Run detect-video first.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    tiles: list[tuple[int, object]] = []
    try:
        for track_id, record in sorted(best_by_id.items()):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(record["frame_index"]))
            ok, frame = capture.read()
            if not ok:
                continue
            bbox = record["bbox"]
            x1, y1 = max(0, int(bbox["x1"])), max(0, int(bbox["y1"]))
            x2, y2 = min(frame.shape[1], int(bbox["x2"])), min(frame.shape[0], int(bbox["y2"]))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            tile = _make_tile(cv2, crop, track_id, float(record["confidence"]))
            tiles.append((track_id, tile))
    finally:
        capture.release()

    if not tiles:
        raise ValueError("Could not extract person images from the video.")

    sheet = _make_sheet(cv2, tiles)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), sheet):
        raise RuntimeError(f"Could not save gallery: {target}")
    return len(tiles)


def show_person_gallery(image_path: str | Path) -> None:
    """Open the generated gallery so the user can choose a person ID."""
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not open person gallery: {image_path}")
    cv2.namedWindow("God's Eye - Select Person ID", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("God's Eye - Select Person ID", min(1000, image.shape[1]), min(750, image.shape[0]))
    cv2.imshow("God's Eye - Select Person ID", image)
    print("The person gallery is open. Note the Person ID, then press any key to continue.")
    try:
        cv2.waitKey(0)
    finally:
        # Windows can close the window itself after a keypress. In that case,
        # OpenCV raises while destroying an already closed window.
        try:
            cv2.destroyWindow("God's Eye - Select Person ID")
        except cv2.error:
            pass


def save_track_exit_image(
    video_path: str | Path,
    detections_path: str | Path,
    track_id: int,
    output_path: str | Path,
) -> None:
    """Save the final detected crop for one selected track."""
    import cv2

    records = [
        record for record in read_jsonl(detections_path)
        if record.get("track_id") == track_id
    ]
    if not records:
        raise ValueError(f"No detections found for Person ID {track_id}.")
    record = max(records, key=lambda item: int(item["frame_index"]))
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(record["frame_index"]))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Could not read the final exit frame.")
        bbox = record["bbox"]
        x1, y1 = max(0, int(bbox["x1"])), max(0, int(bbox["y1"]))
        x2, y2 = min(frame.shape[1], int(bbox["x2"])), min(frame.shape[0], int(bbox["y2"]))
        crop = frame[y1:y2, x1:x2]
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if crop.size == 0 or not cv2.imwrite(str(target), crop):
            raise RuntimeError(f"Could not save exit image: {target}")
    finally:
        capture.release()


def create_comparison_image(
    source_video: str | Path,
    source_detections: str | Path,
    source_track_id: int,
    target_video: str | Path,
    target_detections: str | Path,
    target_track_id: int,
    output_path: str | Path,
    show: bool = False,
) -> None:
    """Create a labelled side-by-side image for one cross-camera match."""
    import cv2
    import numpy as np

    source_crop = _track_crop(source_video, source_detections, source_track_id, last=True)
    target_crop = _track_crop(target_video, target_detections, target_track_id, last=False)
    source_tile = _comparison_tile(cv2, source_crop, f"Source: Person {source_track_id}")
    target_tile = _comparison_tile(cv2, target_crop, f"Matched: Person {target_track_id}")
    comparison = np.hstack([source_tile, target_tile])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), comparison):
        raise RuntimeError(f"Could not save comparison image: {target}")
    if show:
        cv2.imshow("God's Eye Person Comparison", comparison)
        cv2.waitKey(0)
        cv2.destroyWindow("God's Eye Person Comparison")


def _track_crop(video_path: str | Path, detections_path: str | Path, track_id: int, last: bool) -> object:
    import cv2

    records = [record for record in read_jsonl(detections_path) if record.get("track_id") == track_id]
    if not records:
        raise ValueError(f"No detections found for Person ID {track_id}.")
    record = max(records, key=lambda item: int(item["frame_index"])) if last else min(records, key=lambda item: int(item["frame_index"]))
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(record["frame_index"]))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Could not read comparison frame.")
        bbox = record["bbox"]
        x1, y1 = max(0, int(bbox["x1"])), max(0, int(bbox["y1"]))
        x2, y2 = min(frame.shape[1], int(bbox["x2"])), min(frame.shape[0], int(bbox["y2"]))
        return frame[y1:y2, x1:x2]
    finally:
        capture.release()


def _comparison_tile(cv2: object, crop: object, label: str) -> object:
    import numpy as np

    width, height = 260, 360
    tile = np.full((height, width, 3), 25, dtype=np.uint8)
    if crop.size:
        scale = min(width / crop.shape[1], (height - 45) / crop.shape[0])
        resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
        x = (width - resized.shape[1]) // 2
        y = 45 + (height - 45 - resized.shape[0]) // 2
        tile[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(tile, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    return tile


def _make_tile(cv2: object, crop: object, track_id: int, confidence: float) -> object:
    target_width, target_height = 180, 270
    scale = min(target_width / crop.shape[1], (target_height - 34) / crop.shape[0])
    resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
    tile = cv2.copyMakeBorder(
        resized,
        0,
        target_height - 34 - resized.shape[0],
        (target_width - resized.shape[1]) // 2,
        target_width - resized.shape[1] - ((target_width - resized.shape[1]) // 2),
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25),
    )
    return cv2.copyMakeBorder(tile, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))


def _make_sheet(cv2: object, tiles: list[tuple[int, object]]) -> object:
    import numpy as np

    columns = min(4, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    blank = np.full_like(tiles[0][1], 25)
    padded = tiles + [(0, blank)] * (rows * columns - len(tiles))
    sheet_rows = [np.hstack([item[1] for item in padded[index:index + columns]]) for index in range(0, len(padded), columns)]
    sheet = np.vstack(sheet_rows)
    for index, (track_id, _) in enumerate(tiles):
        row, column = divmod(index, columns)
        cv2.putText(sheet, f"Person ID: {track_id}", (column * 180 + 10, row * 270 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    return sheet
