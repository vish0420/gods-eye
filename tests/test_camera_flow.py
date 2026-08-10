import unittest

from godseye.domain import BoundingBox, CameraLink, TrackSegment, Zone
from godseye.matching import CameraFlowGraph


class CameraFlowTests(unittest.TestCase):
    def test_auto_matches_good_time_zone_and_appearance(self) -> None:
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
        source = TrackSegment(
            track_id="cam1-12",
            camera_id="cam1",
            entry_time_s=0,
            exit_time_s=10,
            exit_bbox=BoundingBox(590, 100, 635, 250),
            exit_zone=Zone.RIGHT,
        )
        candidate = TrackSegment(
            track_id="cam2-4",
            camera_id="cam2",
            entry_time_s=14,
            exit_time_s=10,
            entry_bbox=BoundingBox(5, 100, 55, 250),
            entry_zone=Zone.RIGHT,
            appearance_score=0.95,
        )

        decision = graph.score_transition(source, candidate)

        self.assertEqual(decision.decision, "auto_match")
        self.assertGreaterEqual(decision.score, 0.85)

    def test_asks_user_for_medium_confidence(self) -> None:
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
        source = TrackSegment(
            track_id="cam1-12",
            camera_id="cam1",
            entry_time_s=0,
            exit_time_s=10,
            exit_zone=Zone.RIGHT,
        )
        candidate = TrackSegment(
            track_id="cam2-9",
            camera_id="cam2",
            entry_time_s=14,
            exit_time_s=10,
            entry_zone=Zone.RIGHT,
            appearance_score=0.2,
        )

        decision = graph.score_transition(source, candidate)

        self.assertEqual(decision.decision, "ask_user")


if __name__ == "__main__":
    unittest.main()
