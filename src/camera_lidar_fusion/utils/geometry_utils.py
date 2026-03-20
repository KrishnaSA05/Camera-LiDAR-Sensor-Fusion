def rect_contains(rect, pt, shrink_factor: float = 0.0) -> bool:
    """
    Check if point pt lies inside rect [x1, y1, x2, y2].
    shrink_factor shrinks the box inward on all sides (0.0 = no shrink).
    """
    x1 = rect[0] + (rect[2] - rect[0]) * shrink_factor
    y1 = rect[1] + (rect[3] - rect[1]) * shrink_factor
    x2 = rect[2] - (rect[2] - rect[0]) * shrink_factor
    y2 = rect[3] - (rect[3] - rect[1]) * shrink_factor
    return x1 < pt[0] < x2 and y1 < pt[1] < y2
