import unittest

from godseye.domain import BoundingBox, CameraLink, Direction, TrackSummary, Zone
from godseye.matching import CameraFlowGraph, compare_tracks


class CrossCameraTests(unittest.TestCase):
    def test_compares_tracks_and_reports_auto_match(self) -> None:
        graph = CameraFlowGraph(
            [
                CameraLink(
                    from_camera="cam1",
                    to_camera="cam2",
                    exit_zone=Zone.RIGHT,
                    entry_zone=Zone.LEFT,
                    min_travel_s=1,
                    max_travel_s=8,
                )
            ]
        )
        source = [
            TrackSummary(
                track_id=1,
                camera_id="cam1",
                entry_time_s=0.0,
                exit_time_s=10.0,
                first_frame_index=0,
                last_frame_index=100,
                detection_count=20,
                entry_bbox=BoundingBox(10, 20, 90, 200),
                exit_bbox=BoundingBox(590, 20, 635, 200),
                direction=Direction.RIGHT,
                entry_zone=Zone.LEFT,
                exit_zone=Zone.RIGHT,
                average_confidence=0.9,
            )
        ]
        target = [
            TrackSummary(
                track_id=2,
                camera_id="cam2",
                entry_time_s=14.0,
                exit_time_s=30.0,
                first_frame_index=0,
                last_frame_index=120,
                detection_count=30,
                entry_bbox=BoundingBox(5, 20, 80, 200),
                exit_bbox=BoundingBox(300, 20, 390, 200),
                direction=Direction.RIGHT,
                entry_zone=Zone.LEFT,
                exit_zone=Zone.CENTER,
                average_confidence=0.9,
            )
        ]

        report = compare_tracks(graph, source, target)

        self.assertEqual(len(report.decisions), 1)
        self.assertEqual(report.decisions[0].decision, "auto_match")
        self.assertIn("AUTO MATCH", report.text)


if __name__ == "__main__":
    unittest.main()

