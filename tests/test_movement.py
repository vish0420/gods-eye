import unittest

from godseye.domain import BoundingBox, Direction, Zone
from godseye.movement import infer_direction, infer_zone


class MovementAnalyzerTests(unittest.TestCase):
    def test_infers_horizontal_direction(self) -> None:
        start = BoundingBox(10, 100, 50, 200)
        end = BoundingBox(300, 100, 340, 200)

        self.assertEqual(infer_direction(start, end, 640, 480), Direction.RIGHT)

    def test_infers_stationary_when_delta_is_small(self) -> None:
        start = BoundingBox(100, 100, 150, 200)
        end = BoundingBox(108, 105, 158, 205)

        self.assertEqual(infer_direction(start, end, 640, 480), Direction.STATIONARY)

    def test_infers_edge_zone(self) -> None:
        bbox = BoundingBox(0, 120, 50, 260)

        self.assertEqual(infer_zone(bbox, 640, 480), Zone.LEFT)


if __name__ == "__main__":
    unittest.main()

