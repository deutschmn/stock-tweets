from typing import List

from src.data.movement import ClassifiedMovement


def movement_stats(movements: List[ClassifiedMovement]) -> str:
    """Returns direction stats for a list of movements"""
    dirs = [m.direction for m in movements]
    total = len(dirs)
    up = sum(dirs)
    down = total - up
    return f"{up} ({(up / total * 100):.2f}%) up, {down} ({(down / total * 100):.2f}%) down"
