"""Synthetic walk generator, for demos and tests.

Real GPS traces are the point of this project, and they are also personal.
Nothing in the repository ships someone's home coordinates, so the demo, the
tests and the screenshots all run on generated walks around a fictional
anchor. The movement model is deliberately simple but not random: it produces
the behavioural signatures the classifier keys on -- a fast straight commute
lane, a slow looping park path, a long stationary bench stop -- so the
pipeline is exercised on shapes it would actually meet in the field.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from .geo import Fix, Projection, Walk, haversine_m

#: Roughly Harare's altitude, since the demo anchor is a Harare street corner.
BASE_ELEVATION_M = 1480.0


def _walk_along(
    projection: Projection,
    path: list[tuple[float, float]],
    start: datetime,
    speed_ms: float,
    jitter_m: float,
    rng: random.Random,
    pause_at: tuple[int, float] | None = None,
    hz: float = 1.0,
) -> list[Fix]:
    """Convert a metre-space polyline into timed fixes at a realistic rate.

    A phone logging a walk writes roughly one fix per second, and consumer GPS
    noise is a few metres regardless of how fast you are going. Sampling
    coarsely instead (one fix per ten metres, say) makes the trace look clean
    and the derived features wrong, so this walks the clock rather than
    walking the distance.
    """
    fixes: list[Fix] = []
    clock = start
    step_s = 1.0 / max(hz, 0.1)
    step_m = max(speed_ms, 0.05) * step_s
    x, y = path[0]

    def emit(px: float, py: float) -> None:
        lat, lon = projection.to_latlon(px + rng.gauss(0, jitter_m),
                                        py + rng.gauss(0, jitter_m))
        # A gentle rise to the north, so walks that head out in that direction
        # accumulate real ascent and the elevation feature has something to say.
        ele = BASE_ELEVATION_M + py * 0.018 + rng.gauss(0, 0.6)
        fixes.append(Fix(lat=lat, lon=lon, t=clock, ele=ele))

    emit(x, y)
    for index in range(len(path) - 1):
        tx, ty = path[index + 1]
        while math.dist((x, y), (tx, ty)) > step_m:
            remaining = math.dist((x, y), (tx, ty))
            x += (tx - x) / remaining * step_m
            y += (ty - y) / remaining * step_m
            clock += timedelta(seconds=step_s)
            emit(x, y)
        x, y = tx, ty
        clock += timedelta(seconds=step_s)
        emit(x, y)
        if pause_at is not None and pause_at[0] == index:
            # standing still: the fix stays put and GPS wanders around it
            for _ in range(int(pause_at[1] * hz)):
                clock += timedelta(seconds=step_s)
                emit(x, y)
    return fixes


def _commute_lane(rng: random.Random, reach_m: float) -> list[tuple[float, float]]:
    """A near-straight out-and-back road heading east."""
    points = []
    y = rng.uniform(-20, 20)
    steps = 22
    for i in range(steps + 1):
        x = reach_m * i / steps
        points.append((x, y + rng.gauss(0, 3.0)))
    return points + list(reversed(points[:-1]))


def _park_loop(rng: random.Random, center: tuple[float, float], radius: float) -> list[tuple[float, float]]:
    """A closed loop that wanders -- slow, curving, low straightness."""
    cx, cy = center
    points = []
    steps = 48
    for i in range(steps + 1):
        angle = 2 * math.pi * i / steps
        wobble = radius * (1.0 + rng.gauss(0, 0.08))
        points.append((cx + wobble * math.cos(angle), cy + wobble * math.sin(angle) * 0.75))
    return points


def _out_and_back(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return points + list(reversed(points[:-1]))


def generate_neighbourhood(
    anchor_lat: float = -17.8255,
    anchor_lon: float = 31.0522,
    days: int = 12,
    seed: int = 7,
) -> list[Walk]:
    """Generate a plausible fortnight of walking around a fictional anchor."""
    rng = random.Random(seed)
    projection = Projection(anchor_lat, anchor_lon)
    day0 = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
    walks: list[Walk] = []

    for day in range(days):
        date = day0 + timedelta(days=day)
        weekend = date.weekday() >= 5

        # The weekday commute: fast, straight, the same lane every morning.
        if not weekend or rng.random() < 0.3:
            start = date.replace(hour=7, minute=rng.randint(0, 25))
            lane = _commute_lane(rng, reach_m=rng.uniform(850, 1150))
            walks.append(
                Walk(
                    name=f"commute-{date.strftime('%m%d')}",
                    fixes=_walk_along(projection, lane, start, 1.45, 3.2, rng),
                    source="synthetic",
                )
            )

        # An evening park loop with a stop at a bench.
        if rng.random() < 0.75:
            start = date.replace(hour=17, minute=rng.randint(0, 50))
            center = (rng.uniform(-120, 260), rng.uniform(140, 420))
            loop = _park_loop(rng, center, radius=rng.uniform(70, 120))
            walks.append(
                Walk(
                    name=f"park-{date.strftime('%m%d')}",
                    fixes=_walk_along(
                        projection, loop, start, 0.85, 3.2, rng,
                        pause_at=(len(loop) // 2, rng.uniform(300, 900)),
                    ),
                    source="synthetic",
                )
            )

        # On weekends, one longer exploration in a new direction.
        if weekend and rng.random() < 0.8:
            start = date.replace(hour=rng.randint(9, 14), minute=rng.randint(0, 55))
            bearings = [(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if (x, y) != (0, 0)]
            bx, by = rng.choice(bearings)
            reach = rng.uniform(1400, 2600)
            points = []
            for i in range(26):
                frac = i / 25
                points.append((
                    bx * reach * frac + rng.gauss(0, 18),
                    by * reach * frac + rng.gauss(0, 18),
                ))
            walks.append(
                Walk(
                    name=f"ramble-{date.strftime('%m%d')}",
                    fixes=_walk_along(projection, _out_and_back(points), start, 1.15, 3.2, rng),
                    source="synthetic",
                )
            )

    return walks


def walk_distance_m(walk: Walk) -> float:
    total = 0.0
    for a, b in zip(walk.fixes, walk.fixes[1:]):
        total += haversine_m(a.lat, a.lon, b.lat, b.lon)
    return total
