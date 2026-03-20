import statistics
import random
from typing import List


def filter_outliers(distances: List[float]) -> List[float]:
    """Remove values more than 1 standard deviation from the mean."""
    if len(distances) < 2:
        return distances
    mu  = statistics.mean(distances)
    std = statistics.stdev(distances)
    return [x for x in distances if abs(x - mu) < std]


def get_best_distance(distances: List[float], technique: str = "closest") -> float:
    """
    Aggregate a list of distances using a chosen technique.
    Options: 'closest', 'average', 'random', 'median'
    """
    if not distances:
        raise ValueError("Distance list is empty.")
    techniques = {
        "closest": min,
        "average": statistics.mean,
        "random":  random.choice,
        "median":  lambda d: statistics.median(sorted(d)),
    }
    func = techniques.get(technique)
    if func is None:
        raise ValueError(
            f"Unknown technique: '{technique}'. Choose from {list(techniques.keys())}"
        )
    return func(distances)
