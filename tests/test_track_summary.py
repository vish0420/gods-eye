import unittest

from godseye.domain import BoundingBox, Direction, TrackSummary, Zone
from godseye.storage.track_summary import format_track_summary


class TrackSummaryFormattingTests(unittest.TestCase):
    def test_formats_readable_summary(self) -> None:
        tracks = [
            TrackSummary(
                track_id=1,
                camera_id="cam1",
                entry_time_s=0.0,
                exit_time_s=12.4,
                first_frame_index=0,
                last_frame_index=310,
                detection_count=146,
                entry_bbox=BoundingBox(10, 20, 80, 200),
                exit_bbox=BoundingBox(500, 20, 590, 200),
                direction=Direction.RIGHT,
                entry_zone=Zone.LEFT,
                exit_zone=Zone.RIGHT,
                average_confidence=0.873,
            )
        ]

        summary = format_track_summary("cam1", tracks)

        self.assertIn("Unique Persons: 1", summary)
        self.assertIn("Person ID: 1", summary)
        self.assertIn("Duration: 12.4 sec", summary)
        self.assertIn("Direction: right", summary)
        self.assertIn("Average Confidence: 0.87", summary)


if __name__ == "__main__":
    unittest.main()

