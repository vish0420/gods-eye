from __future__ import annotations

from godseye.domain import BoundingBox, Direction, Zone


def infer_direction(
    start: BoundingBox | None,
    end: BoundingBox | None,
    frame_width: int,
    frame_height: int,
    min_delta_ratio: float = 0.04,
) -> Direction:
    """Infer simple screen-space movement from first and last boxes."""
    if start is None or end is None or frame_width <= 0 or frame_height <= 0:
        return Direction.UNKNOWN

    dx = end.center_x - start.center_x
    dy = end.center_y - start.center_y
    min_dx = frame_width * min_delta_ratio
    min_dy = frame_height * min_delta_ratio

    if abs(dx) < min_dx and abs(dy) < min_dy:
        return Direction.STATIONARY

    if abs(dx) >= abs(dy):
        return Direction.RIGHT if dx > 0 else Direction.LEFT

    return Direction.DOWN if dy > 0 else Direction.UP


def infer_zone(
    bbox: BoundingBox | None,
    frame_width: int,
    frame_height: int,
    margin_ratio: float = 0.2,
) -> Zone:
    """Return the edge zone where the person center appears."""
    if bbox is None or frame_width <= 0 or frame_height <= 0:
        return Zone.UNKNOWN

    left_limit = frame_width * margin_ratio
    right_limit = frame_width * (1.0 - margin_ratio)
    top_limit = frame_height * margin_ratio
    bottom_limit = frame_height * (1.0 - margin_ratio)

    at_left = bbox.center_x <= left_limit
    at_right = bbox.center_x >= right_limit
    at_top = bbox.center_y <= top_limit
    at_bottom = bbox.center_y >= bottom_limit

    if at_left and at_top:
        return Zone.TOP_LEFT
    if at_right and at_top:
        return Zone.TOP_RIGHT
    if at_left and at_bottom:
        return Zone.BOTTOM_LEFT
    if at_right and at_bottom:
        return Zone.BOTTOM_RIGHT
    if at_left:
        return Zone.LEFT
    if at_right:
        return Zone.RIGHT
    if at_top:
        return Zone.TOP
    if at_bottom:
        return Zone.BOTTOM
    return Zone.CENTER
