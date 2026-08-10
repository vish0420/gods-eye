from __future__ import annotations

from pathlib import Path

from godseye.detection import YoloPersonDetector
from godseye.gallery import (
    create_comparison_image,
    create_person_gallery,
    save_track_exit_image,
)
from godseye.matching import CameraFlowGraph
from godseye.matching.cross_camera import _summary_to_segment
from godseye.storage.event_log import write_jsonl
from godseye.storage.track_io import load_track_summaries
from godseye.storage.track_summary import write_track_summary
from godseye.video import VideoDetectionProcessor


def run_guided_workflow() -> None:
    start_camera = input("Camera name or number (example: cam1 or 1): ").strip().lower()
    if start_camera.isdigit():
        start_camera = f"cam{start_camera}"
    show_answer = input("Show tracking video? Type yes or no (example: yes): ").strip().lower()
    show = show_answer in {"y", "yes", "show"}
    videos = _find_videos()
    if start_camera not in videos:
        raise FileNotFoundError(f"Video not found: videos/{start_camera}.mp4")

    detector = YoloPersonDetector()
    processor = VideoDetectionProcessor(detector)
    processed: set[str] = set()
    _process_camera(start_camera, videos, processor, show, processed)

    gallery_path = f"runs/{start_camera}_people.jpg"
    create_person_gallery(videos[start_camera], f"runs/{start_camera}_detections.jsonl", gallery_path)
    print(f"\nPerson gallery created: {gallery_path}")
    person_id = _read_person_id()
    journey = _follow_person(start_camera, person_id, videos, processor, show, processed)
    summary_path = Path("runs/final_person_timeline.txt")
    summary_path.write_text("\n".join(journey["lines"]) + "\n", encoding="utf-8")
    last_camera, last_track_id = journey["last"]
    save_track_exit_image(
        videos[last_camera],
        f"runs/{last_camera}_detections.jsonl",
        last_track_id,
        "runs/final_exit_person.jpg",
    )
    print(f"\nFinal timeline: {summary_path}")
    print("Final exit image: runs/final_exit_person.jpg")


def _find_videos() -> dict[str, Path]:
    return {
        path.stem.lower(): path
        for path in Path("videos").glob("cam*.mp4")
    }


def _process_camera(
    camera_id: str,
    videos: dict[str, Path],
    processor: VideoDetectionProcessor,
    show: bool,
    processed: set[str],
) -> None:
    if camera_id in processed:
        return
    if camera_id not in videos:
        raise FileNotFoundError(f"Video not found for predicted camera: {camera_id}")
    video_path = videos[camera_id]
    print(f"\nAnalysing predicted camera {camera_id}: {video_path}")
    result = processor.process(
        video_path=video_path,
        camera_id=camera_id,
        every_n_frames=2,
        output_jsonl=f"runs/{camera_id}_detections.jsonl",
        annotated_output=f"runs/{camera_id}_output.mp4",
        show=show,
    )
    write_jsonl(f"runs/{camera_id}_tracks.jsonl", result.tracks)
    write_track_summary(f"runs/{camera_id}_summary.txt", camera_id, result.tracks)
    processed.add(camera_id)
    print(f"{camera_id}: {len(result.tracks)} unique person(s) after filtering weak tracks.")


def _follow_person(
    start_camera: str,
    person_id: int,
    videos: dict[str, Path],
    processor: VideoDetectionProcessor,
    show: bool,
    processed: set[str],
) -> dict[str, object]:
    graph = CameraFlowGraph.from_json("configs/cameras.example.json")
    tracks_by_camera = {start_camera: load_track_summaries(f"runs/{start_camera}_tracks.jsonl")}
    current = next((track for track in tracks_by_camera[start_camera] if track.track_id == person_id), None)
    if current is None:
        raise ValueError(f"Person ID {person_id} was not found in {start_camera}.")

    lines = ["God's Eye Final Person Timeline", ""]
    visited: set[tuple[str, int]] = set()
    while current and (current.camera_id, current.track_id) not in visited:
        visited.add((current.camera_id, current.track_id))
        lines.append(
            f"{current.camera_id.upper()} | Person ID {current.track_id} | "
            f"{current.entry_time_s:.1f}s to {current.exit_time_s:.1f}s | "
            f"direction {current.direction.value}"
        )
        source = _summary_to_segment(current)
        candidates = []
        for link in graph.expected_links(current.camera_id, current.exit_zone):
            _process_camera(link.to_camera, videos, processor, show, processed)
            if link.to_camera not in tracks_by_camera:
                tracks_by_camera[link.to_camera] = load_track_summaries(
                    f"runs/{link.to_camera}_tracks.jsonl"
                )
            for target in tracks_by_camera.get(link.to_camera, []):
                decision = graph.score_transition(source, _summary_to_segment(target))
                candidates.append((decision.score, link, target, decision))
        if not candidates:
            break
        _, link, next_track, decision = max(candidates, key=lambda item: item[0])
        if decision.decision == "reject":
            lines.append(f"Next camera prediction: {link.to_camera.upper()} (not confirmed)")
            break
        expected = current.exit_time_s + link.min_travel_s
        lines.append(
            f"Expected next: {link.to_camera.upper()} from {expected:.1f}s "
            f"(minimum travel {link.min_travel_s:.1f}s, match {decision.score * 100:.1f}%)"
        )
        comparison_path = Path("Tracker Images") / (
            f"{current.camera_id}_to_{next_track.camera_id}_person_{current.track_id}_vs_{next_track.track_id}.jpg"
        )
        create_comparison_image(
            videos[current.camera_id],
            f"runs/{current.camera_id}_detections.jsonl",
            current.track_id,
            videos[next_track.camera_id],
            f"runs/{next_track.camera_id}_detections.jsonl",
            next_track.track_id,
            comparison_path,
            show=show,
        )
        lines.append(f"Comparison image: {comparison_path}")
        current = next_track
    lines.extend(["", f"Last seen: {current.camera_id.upper()}, Person ID {current.track_id}"])
    return {"lines": lines, "last": (current.camera_id, current.track_id)}


def _read_person_id() -> int:
    while True:
        value = input("Enter Person ID from the gallery (example: 1): ").strip()
        if value.isdigit() and int(value) > 0:
            return int(value)
        print("Invalid input. Enter only a positive number, for example: 1")
