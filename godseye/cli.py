from __future__ import annotations

import argparse
import sys

from godseye.detection import YoloPersonDetector
from godseye.gallery import create_person_gallery
from godseye.matching import compare_track_files, write_cross_camera_report
from godseye.storage.event_log import write_jsonl
from godseye.storage.track_summary import write_track_summary
from godseye.video import VideoDetectionProcessor, play_video
from godseye.guided_run import run_guided_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="godseye",
        description="God's Eye recorded-footage demo tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("detect-image", help="Detect people in one image")
    image_parser.add_argument("--image", required=True, help="Image path")
    image_parser.add_argument("--camera-id", default="image", help="Camera ID label")
    image_parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    image_parser.add_argument("--confidence", type=float, default=0.35)
    image_parser.add_argument("--device", default=None, help="Example: cpu, cuda, 0")
    image_parser.add_argument("--out", default="runs/image_detections.jsonl")
    image_parser.add_argument("--annotated", default=None, help="Optional annotated image path")
    image_parser.set_defaults(func=run_detect_image)

    video_parser = subparsers.add_parser("detect-video", help="Detect people in a video")
    video_parser.add_argument("--video", required=True, help="Video path")
    video_parser.add_argument("--camera-id", required=True, help="Camera ID label")
    video_parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    video_parser.add_argument("--confidence", type=float, default=0.35)
    video_parser.add_argument("--device", default=None, help="Example: cpu, cuda, 0")
    video_parser.add_argument("--every", type=int, default=2, help="Process every N frames")
    video_parser.add_argument("--max-frames", type=int, default=None)
    video_parser.add_argument("--min-track-detections", type=int, default=3)
    video_parser.add_argument("--min-track-confidence", type=float, default=0.60)
    video_parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for output files. Default uses camera ID, like cam1.",
    )
    video_parser.add_argument("--out", default=None)
    video_parser.add_argument("--tracks-out", default=None)
    video_parser.add_argument("--summary-out", default=None)
    video_parser.add_argument("--annotated", default=None, help="Optional annotated MP4 path")
    video_parser.add_argument(
        "--show",
        action="store_true",
        help="Show annotated video while processing. Press q to close.",
    )
    video_parser.add_argument(
        "--show-delay-ms",
        type=int,
        default=1,
        help="Delay between displayed frames when --show is used.",
    )
    video_parser.add_argument(
        "--hold-window",
        action="store_true",
        help="Keep the final video window open until a key is pressed.",
    )
    video_parser.set_defaults(func=run_detect_video)

    play_parser = subparsers.add_parser("play-video", help="Play a saved video output")
    play_parser.add_argument("--video", required=True, help="Video path to play")
    play_parser.set_defaults(func=run_play_video)

    gallery_parser = subparsers.add_parser(
        "person-gallery", help="Create a labelled image of the people detected in a video"
    )
    gallery_parser.add_argument("--video", required=True, help="Original video path")
    gallery_parser.add_argument("--detections", required=True, help="Detection JSONL from detect-video")
    gallery_parser.add_argument("--out", default="runs/person_gallery.jpg")
    gallery_parser.set_defaults(func=run_person_gallery)

    compare_parser = subparsers.add_parser(
        "compare-cameras",
        help="Compare two camera track files using camera-flow rules",
    )
    compare_parser.add_argument(
        "--config",
        default="configs/cameras.example.json",
        help="Camera flow config JSON",
    )
    compare_parser.add_argument("--source-tracks", required=True)
    compare_parser.add_argument("--target-tracks", required=True)
    compare_parser.add_argument("--top", type=int, default=3)
    compare_parser.add_argument(
        "--person-id",
        type=int,
        default=None,
        help="Only compare this selected person ID from the source camera.",
    )
    compare_parser.add_argument("--out", default="runs/camera_match_report.txt")
    compare_parser.add_argument("--json-out", default="runs/camera_match_decisions.jsonl")
    compare_parser.set_defaults(func=run_compare_cameras)

    return parser


def run_detect_image(args: argparse.Namespace) -> None:
    detector = YoloPersonDetector(
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
    )
    detections = detector.detect_image(
        image_path=args.image,
        camera_id=args.camera_id,
        annotated_path=args.annotated,
    )
    write_jsonl(args.out, detections)
    print(f"Detected {len(detections)} person(s). Results: {args.out}")


def run_detect_video(args: argparse.Namespace) -> None:
    output_paths = _resolve_video_output_paths(args)
    detector = YoloPersonDetector(
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
    )
    processor = VideoDetectionProcessor(detector)
    result = processor.process(
        video_path=args.video,
        camera_id=args.camera_id,
        every_n_frames=args.every,
        max_frames=args.max_frames,
        output_jsonl=output_paths["detections"],
        annotated_output=output_paths["annotated"],
        min_track_detections=args.min_track_detections,
        min_track_confidence=args.min_track_confidence,
        show=args.show,
        show_delay_ms=args.show_delay_ms,
        hold_window=args.hold_window,
    )
    write_jsonl(output_paths["tracks"], result.tracks)
    write_track_summary(output_paths["summary"], args.camera_id, result.tracks)
    print(f"Detected {len(result.tracks)} unique person(s).")
    print(f"Frame detections: {len(result.detections)} saved to {output_paths['detections']}")
    print(f"Track data: {output_paths['tracks']}")
    print(f"Readable summary: {output_paths['summary']}")
    if output_paths["annotated"] is not None:
        print(f"Annotated video: {output_paths['annotated']}")
    for track in result.tracks:
        print(
            "  "
            f"ID {track.track_id}: "
            f"{track.entry_time_s:.1f}s-{track.exit_time_s:.1f}s, "
            f"duration {track.duration_s:.1f}s, "
            f"direction {track.direction.value}, "
            f"detections {track.detection_count}"
        )


def _resolve_video_output_paths(args: argparse.Namespace) -> dict[str, str | None]:
    prefix = _clean_output_prefix(args.output_prefix or args.camera_id)
    return {
        "detections": args.out or f"runs/{prefix}_detections.jsonl",
        "tracks": args.tracks_out or f"runs/{prefix}_tracks.jsonl",
        "summary": args.summary_out or f"runs/{prefix}_summary.txt",
        "annotated": args.annotated,
    }


def _clean_output_prefix(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)
    return cleaned.strip("_") or "video"


def run_play_video(args: argparse.Namespace) -> None:
    play_video(args.video)


def run_person_gallery(args: argparse.Namespace) -> None:
    count = create_person_gallery(args.video, args.detections, args.out)
    print(f"Created gallery with {count} person(s): {args.out}")


def run_compare_cameras(args: argparse.Namespace) -> None:
    report = compare_track_files(
        config_path=args.config,
        source_tracks_path=args.source_tracks,
        target_tracks_path=args.target_tracks,
        top_k=args.top,
        source_person_id=args.person_id,
    )
    write_cross_camera_report(
        report=report,
        text_path=args.out,
        jsonl_path=args.json_out,
    )
    print(f"Compared {args.source_tracks} -> {args.target_tracks}")
    print(f"Readable report: {args.out}")
    print(f"Match data: {args.json_out}")
    print("")
    print(report.text)


def main() -> None:
    if len(sys.argv) == 1:
        run_guided_workflow()
        return
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
