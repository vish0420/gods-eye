from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from godseye.detection import YoloPersonDetector
from godseye.domain import MatchDecision, TrackSummary
from godseye.gallery import (
    create_comparison_image,
    create_person_gallery,
    save_track_exit_image,
    show_person_gallery,
)
from godseye.matching import CameraFlowGraph
from godseye.matching.cross_camera import _summary_to_segment
from godseye.storage.event_log import write_jsonl
from godseye.storage.track_io import load_track_summaries
from godseye.storage.track_summary import write_track_summary
from godseye.video import VideoDetectionProcessor


@dataclass(frozen=True)
class TargetSelection:
    primary_track_id: int
    reference_track_ids: tuple[int, ...]


def run_guided_workflow(start_video: str | Path | None = None) -> None:
    if start_video is None:
        start_camera = input("Camera name or number (example: cam1 or 1): ").strip().lower()
        if start_camera.isdigit():
            start_camera = f"cam{start_camera}"
        show_answer = input("Show tracking video? Type yes or no (example: yes): ").strip().lower()
        show = show_answer in {"y", "yes", "show"}
        videos = _find_videos()
        if start_camera not in videos:
            raise FileNotFoundError(f"Video not found: datasets/video/{start_camera}.mp4")
        start_video_path = videos[start_camera]
    else:
        start_video_path = Path(start_video)
        start_camera = start_video_path.stem.lower()
        show = True
        videos = _find_videos()
        videos[start_camera] = start_video_path

    _run_workflow(start_camera, start_video_path, videos, show)


def _run_workflow(
    start_camera: str,
    start_video_path: Path,
    videos: dict[str, Path],
    show: bool,
) -> None:
    detector = YoloPersonDetector()
    processor = VideoDetectionProcessor(detector)
    processed: set[str] = set()

    _process_camera(start_camera, start_video_path, processor, show, processed)

    gallery_path = f"result/image/{start_camera}_people.jpg"
    final_tracks = load_track_summaries(f"result/video/{start_camera}_tracks.jsonl")
    create_person_gallery(
        start_video_path,
        f"result/video/{start_camera}_detections.jsonl",
        gallery_path,
        allowed_track_ids={track.track_id for track in final_tracks},
    )
    print(f"\nPerson gallery created: {gallery_path}")
    show_person_gallery(gallery_path)
    target_selection = _confirm_target_selection(final_tracks)

    journey = _follow_person(
        start_camera,
        target_selection,
        videos,
        processor,
        show,
        processed,
    )
    summary_path = Path("result/video/final_person_timeline.txt")
    summary_path.write_text("\n".join(journey["lines"]) + "\n", encoding="utf-8")
    last_camera, last_track_id = journey["last"]
    save_track_exit_image(
        videos[last_camera],
        f"result/video/{last_camera}_detections.jsonl",
        last_track_id,
        "result/image/final_exit_person.jpg",
    )
    print(f"\nFinal timeline: {summary_path}")
    print("Final exit image: result/image/final_exit_person.jpg")


def _find_videos() -> dict[str, Path]:
    return {
        path.stem.lower(): path
        for path in Path("datasets/video").glob("cam*.mp4")
    }


def _process_camera(
    camera_id: str,
    video_path: Path,
    processor: VideoDetectionProcessor,
    show: bool,
    processed: set[str],
) -> None:
    if camera_id in processed:
        return
    print(f"\nAnalysing predicted camera {camera_id}: {video_path}")
    result = processor.process(
        video_path=video_path,
        camera_id=camera_id,
        every_n_frames=2,
        output_jsonl=f"result/video/{camera_id}_detections.jsonl",
        annotated_output=f"result/video/{camera_id}_output.mp4",
        show=show,
    )
    write_jsonl(f"result/video/{camera_id}_tracks.jsonl", result.tracks)
    write_track_summary(f"result/video/{camera_id}_summary.txt", camera_id, result.tracks)
    processed.add(camera_id)
    print(f"{camera_id}: {len(result.tracks)} unique person(s) after filtering weak tracks.")


def _confirm_target_selection(tracks: list[TrackSummary]) -> TargetSelection:
    track_ids = [track.track_id for track in tracks]
    primary_track_id = _choose_track_id(
        title="God's Eye - Confirm Target Person",
        prompt="Select the main target person from the gallery.",
        track_ids=track_ids,
        allow_multiple=False,
    )[0]
    alternate_track_ids = [track_id for track_id in track_ids if track_id != primary_track_id]
    alternate_ids = (
        _choose_track_id(
            title="God's Eye - Same Person Check",
            prompt=(
                "Select any other gallery entries that are actually the same person "
                "as the target. If one person was split into 2 or more gallery entries, "
                "select every matching entry here."
            ),
            track_ids=alternate_track_ids,
            allow_multiple=True,
        )
        if alternate_track_ids
        else []
    )
    reference_track_ids = tuple(dict.fromkeys([primary_track_id, *alternate_ids]))
    return TargetSelection(
        primary_track_id=primary_track_id,
        reference_track_ids=reference_track_ids,
    )


def _choose_track_id(
    title: str,
    prompt: str,
    track_ids: list[int],
    allow_multiple: bool,
) -> list[int]:
    if not track_ids:
        raise ValueError("No track IDs available for selection.")

    root = tk.Tk()
    root.title(title)
    root.geometry("620x420")
    root.minsize(520, 320)
    root.configure(bg="#101418")

    header = tk.Label(
        root,
        text=prompt,
        fg="#f2f2f2",
        bg="#101418",
        wraplength=560,
        justify="left",
        font=("Segoe UI", 11),
    )
    header.pack(pady=(18, 10), padx=16, anchor="w")

    frame = tk.Frame(root, bg="#101418")
    frame.pack(fill="both", expand=True, padx=16, pady=(0, 14))

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(
        frame,
        selectmode=tk.MULTIPLE if allow_multiple else tk.SINGLE,
        yscrollcommand=scrollbar.set,
        bg="#18212b",
        fg="#f4f7fb",
        selectbackground="#2d7dd2",
        selectforeground="#ffffff",
        relief="flat",
        font=("Segoe UI", 11),
    )
    for track_id in track_ids:
        listbox.insert(tk.END, f"Person ID {track_id}")
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    if not allow_multiple:
        listbox.selection_set(0)
        listbox.activate(0)

    chosen: list[int] = []

    def confirm() -> None:
        selection = listbox.curselection()
        if not selection and not allow_multiple:
            messagebox.showinfo("God's Eye", "Select at least one person ID.")
            return
        if not selection and allow_multiple:
            chosen.clear()
            root.quit()
            return
        chosen[:] = [track_ids[index] for index in selection]
        root.quit()

    controls = tk.Frame(root, bg="#101418")
    controls.pack(fill="x", padx=16, pady=(0, 16))

    help_text = tk.Label(
        controls,
        text="Use Ctrl or Shift for more than one choice.",
        fg="#b9c2cc",
        bg="#101418",
        anchor="w",
        font=("Segoe UI", 10),
    )
    help_text.pack(side="left", fill="x", expand=True)

    button = tk.Button(
        controls,
        text="Confirm",
        command=confirm,
        bg="#2d7dd2",
        fg="#ffffff",
        activebackground="#1f63ae",
        activeforeground="#ffffff",
        relief="flat",
        padx=16,
        pady=8,
        font=("Segoe UI", 10, "bold"),
    )
    button.pack(side="right")

    root.protocol("WM_DELETE_WINDOW", root.quit)
    listbox.bind("<Double-Button-1>", lambda _: confirm())

    root.mainloop()
    try:
        root.destroy()
    except tk.TclError:
        pass

    if not chosen:
        raise RuntimeError("Target selection was cancelled.")
    return chosen


def _follow_person(
    start_camera: str,
    target_selection: TargetSelection,
    videos: dict[str, Path],
    processor: VideoDetectionProcessor,
    show: bool,
    processed: set[str],
) -> dict[str, object]:
    graph = CameraFlowGraph.from_json("configs/cameras.example.json")
    tracks_by_camera: dict[str, list[TrackSummary]] = {
        start_camera: load_track_summaries(f"result/video/{start_camera}_tracks.jsonl")
    }
    source_tracks = [
        track
        for track in tracks_by_camera[start_camera]
        if track.track_id in target_selection.reference_track_ids
    ]
    if not source_tracks:
        raise ValueError(
            f"No selected target IDs were found in {start_camera}: "
            f"{target_selection.reference_track_ids}"
        )

    current = next(
        (
            track
            for track in source_tracks
            if track.track_id == target_selection.primary_track_id
        ),
        source_tracks[0],
    )
    reference_tracks = list(source_tracks)
    lines = [
        "God's Eye Final Person Timeline",
        "",
        f"Target Person: {target_selection.primary_track_id}",
        "Reference IDs: "
        + ", ".join(str(track_id) for track_id in target_selection.reference_track_ids),
        "",
    ]

    visited: set[tuple[str, int]] = set()
    while current and (current.camera_id, current.track_id) not in visited:
        visited.add((current.camera_id, current.track_id))
        lines.append(
            f"{current.camera_id.upper()} | Person ID {current.track_id} | "
            f"{current.entry_time_s:.1f}s to {current.exit_time_s:.1f}s | "
            f"direction {current.direction.value}"
        )
        candidates: list[
            tuple[float, object, TrackSummary, MatchDecision, TrackSummary]
        ] = []
        for link in graph.expected_links(current.camera_id, current.exit_zone):
            next_video = videos.get(link.to_camera)
            if next_video is None:
                continue
            _process_camera(link.to_camera, next_video, processor, show, processed)
            if link.to_camera not in tracks_by_camera:
                tracks_by_camera[link.to_camera] = load_track_summaries(
                    f"result/video/{link.to_camera}_tracks.jsonl"
                )
            for target in tracks_by_camera.get(link.to_camera, []):
                decision, matched_source = _best_transition(
                    graph,
                    reference_tracks,
                    target,
                )
                candidates.append((decision.score, link, target, decision, matched_source))
        if not candidates:
            break
        _, link, next_track, decision, matched_source = max(candidates, key=lambda item: item[0])
        if decision.decision == "reject":
            lines.append(f"Next camera prediction: {link.to_camera.upper()} (not confirmed)")
            break
        expected = current.exit_time_s + link.min_travel_s
        lines.append(
            f"Expected next: {link.to_camera.upper()} from {expected:.1f}s "
            f"(minimum travel {link.min_travel_s:.1f}s, match {decision.score * 100:.1f}%)"
        )
        comparison_path = Path("result/image/comparisons") / (
            f"{current.camera_id}_to_{next_track.camera_id}_person_"
            f"{matched_source.track_id}_vs_{next_track.track_id}.jpg"
        )
        create_comparison_image(
            videos[matched_source.camera_id],
            f"result/video/{matched_source.camera_id}_detections.jsonl",
            matched_source.track_id,
            videos[next_track.camera_id],
            f"result/video/{next_track.camera_id}_detections.jsonl",
            next_track.track_id,
            comparison_path,
            show=show,
        )
        lines.append(f"Comparison image: {comparison_path}")
        current = next_track
        if all(track.track_id != current.track_id for track in reference_tracks):
            reference_tracks.append(current)

    lines.extend(["", f"Last seen: {current.camera_id.upper()}, Person ID {current.track_id}"])
    return {"lines": lines, "last": (current.camera_id, current.track_id)}


def _best_transition(
    graph: CameraFlowGraph,
    source_tracks: list[TrackSummary],
    target_track: TrackSummary,
) -> tuple[MatchDecision, TrackSummary]:
    best_decision: MatchDecision | None = None
    best_source: TrackSummary | None = None
    for source_track in source_tracks:
        decision = graph.score_transition(
            _summary_to_segment(source_track),
            _summary_to_segment(target_track),
        )
        if best_decision is None or decision.score > best_decision.score:
            best_decision = decision
            best_source = source_track
    if best_decision is None or best_source is None:
        raise ValueError("Could not score transition against any source track.")
    return best_decision, best_source
