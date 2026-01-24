from __future__ import annotations

import math
from statistics import median
from typing import List, Optional, Tuple


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))


def _typical_step_m(points: List[Tuple[float, float]], *, round_decimals: int = 5) -> float:
    """
    Estimate a "typical" consecutive-point distance (meters) robustly.

    We collapse consecutive duplicates after rounding to avoid a median of ~0 for long stationary segments.
    """
    if len(points) < 2:
        return 1.0

    collapsed: List[Tuple[float, float]] = []
    last_key: Optional[Tuple[float, float]] = None
    for lat, lon in points:
        key = (round(lat, round_decimals), round(lon, round_decimals))
        if key == last_key:
            continue
        collapsed.append((lat, lon))
        last_key = key

    if len(collapsed) < 2:
        return 1.0

    dists = [haversine_m(collapsed[i - 1], collapsed[i]) for i in range(1, len(collapsed))]
    nontrivial = [d for d in dists if d > 0.5]
    if not nontrivial:
        return 1.0

    # Use the median of the lower half to avoid a single spike (often producing 2 huge legs)
    # dominating when the track is short.
    nontrivial.sort()
    lower = nontrivial[: max(1, len(nontrivial) // 2)]
    return max(1.0, float(median(lower)))


def drop_single_spike_point(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Remove at most one "spike" point: a single outlier coordinate far from its neighbors.

    Heuristic: an interior point is a spike if both legs (prev->p and p->next) are huge while skipping it
    (prev->next) is small.
    """
    n = len(points)
    if n < 4:
        return points

    typical = _typical_step_m(points)
    big_thr = max(250.0, typical * 20.0)
    skip_thr = max(100.0, typical * 5.0)

    best_score = 0.0
    best_idx: Optional[int] = None

    # Interior candidates
    for i in range(1, n - 1):
        d_prev = haversine_m(points[i - 1], points[i])
        d_next = haversine_m(points[i], points[i + 1])
        local_max = max(d_prev, d_next)
        if local_max <= big_thr:
            continue

        d_skip = haversine_m(points[i - 1], points[i + 1])
        if d_skip >= local_max * 0.2:
            continue
        if d_skip >= skip_thr:
            continue

        score = local_max / max(1.0, d_skip)
        if score > best_score:
            best_score = score
            best_idx = i

    # Edge candidates
    d01 = haversine_m(points[0], points[1])
    d12 = haversine_m(points[1], points[2])
    if d01 > big_thr and d12 < skip_thr:
        score = d01 / max(1.0, d12)
        if score > best_score:
            best_score = score
            best_idx = 0

    dnm1 = haversine_m(points[-2], points[-1])
    dnm2 = haversine_m(points[-3], points[-2])
    if dnm1 > big_thr and dnm2 < skip_thr:
        score = dnm1 / max(1.0, dnm2)
        if score > best_score:
            best_score = score
            best_idx = n - 1

    if best_idx is None:
        return points

    return points[:best_idx] + points[best_idx + 1 :]


def drop_spike_points(
    points: List[Tuple[float, float]],
    *,
    max_passes: int = 4,
) -> List[Tuple[float, float]]:
    """
    Remove multiple "spike" points (outliers) that create very large jumps.

    Runs a few passes because removing one spike can expose the next one.
    The detection remains conservative: it prefers removing points that create huge legs while skipping them
    yields a comparatively small step.
    """
    if len(points) < 4:
        return points

    out = list(points)
    for _ in range(max(1, int(max_passes))):
        n = len(out)
        if n < 4:
            break

        typical = _typical_step_m(out)
        big_thr = max(250.0, typical * 20.0)
        skip_thr = max(100.0, typical * 5.0)

        to_drop: List[int] = []

        # Interior candidates
        for i in range(1, n - 1):
            d_prev = haversine_m(out[i - 1], out[i])
            d_next = haversine_m(out[i], out[i + 1])
            d_skip = haversine_m(out[i - 1], out[i + 1])

            # Classic spike: both legs huge, but skipping is small.
            if d_prev > big_thr and d_next > big_thr and d_skip < min(skip_thr, min(d_prev, d_next) * 0.2):
                to_drop.append(i)
                continue

            # One-sided spike-ish: one leg huge, the other normal-ish, and skipping is small.
            local_max = max(d_prev, d_next)
            local_min = min(d_prev, d_next)
            if local_max > big_thr and local_min < skip_thr and d_skip < min(skip_thr, local_max * 0.05):
                to_drop.append(i)

        # Edge candidates (start/end)
        d01 = haversine_m(out[0], out[1])
        d12 = haversine_m(out[1], out[2])
        if d01 > big_thr and d12 < skip_thr:
            to_drop.append(0)

        dnm1 = haversine_m(out[-2], out[-1])
        dnm2 = haversine_m(out[-3], out[-2])
        if dnm1 > big_thr and dnm2 < skip_thr:
            to_drop.append(n - 1)

        if not to_drop:
            break

        # Remove (dedup + keep order)
        drop_set = set(to_drop)
        out = [p for idx, p in enumerate(out) if idx not in drop_set]

    return out
