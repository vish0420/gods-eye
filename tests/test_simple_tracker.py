import unittest

from godseye.domain import BoundingBox, Detection, Direction
from godseye.tracking.simple_tracker import SimplePersonTracker, iou


class SimpleTrackerTests(unittest.TestCase):
    def test_iou_scores_overlap(self) -> None:
        first = BoundingBox(0, 0, 100, 100)
        second = BoundingBox(50, 50, 150, 150)

        self.assertAlmostEqual(iou(first, second), 2500 / 17500)

    def test_counts_same_person_as_one_track(self) -> None:
        tracker = SimplePersonTracker(
            iou_threshold=0.1,
            min_detections=1,
            min_average_confidence=0.0,
        )
        first_frame = [
            Detection("cam1", 0, 0.0, BoundingBox(10, 10, 100, 200), 0.9)
        ]
        second_frame = [
            Detection("cam1", 5, 0.2, BoundingBox(30, 10, 120, 200), 0.9)
        ]

        tracker.update(first_frame, frame_width=640, frame_height=480)
        second_update = tracker.update(second_frame, frame_width=640, frame_height=480)
        tracks = tracker.finish(frame_width=640, frame_height=480)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].detection_count, 2)
        self.assertEqual(second_update[0].direction, Direction.RIGHT)
        self.assertEqual(tracks[0].direction, Direction.STATIONARY)

    def test_counts_two_people_as_two_tracks(self) -> None:
        tracker = SimplePersonTracker(
            iou_threshold=0.1,
            min_detections=1,
            min_average_confidence=0.0,
        )
        tracker.update(
            [
                Detection("cam1", 0, 0.0, BoundingBox(10, 10, 100, 200), 0.9),
                Detection("cam1", 0, 0.0, BoundingBox(300, 10, 390, 200), 0.9),
            ]
        )

        tracks = tracker.finish(frame_width=640, frame_height=480)

        self.assertEqual(len(tracks), 2)

    def test_filters_weak_tracks(self) -> None:
        tracker = SimplePersonTracker(
            iou_threshold=0.1,
            min_detections=1,
            min_average_confidence=0.6,
        )
        tracker.update(
            [
                Detection("cam1", 0, 0.0, BoundingBox(10, 10, 100, 200), 0.4),
                Detection("cam1", 0, 0.0, BoundingBox(300, 10, 390, 200), 0.9),
            ]
        )

        tracks = tracker.finish(frame_width=640, frame_height=480)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].average_confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
