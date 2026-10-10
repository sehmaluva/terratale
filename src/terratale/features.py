"""Turn raw GPS traces into one tabular row per hex.

This matrix is the only thing the classifier ever sees. Every column is a
physical quantity measured from the trace, so a legend the model writes about
a cell can always be traced back to something that actually happened on the
walk. Missing values stay as NaN rather than being filled with a fake zero,
because "we do not know how long you stayed" and "you did not stay" are
different facts, and TabPFN can tell them apart.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .geo import Fix, Walk, haversine_m
from .grid import Hex, HexGrid

#: Feature order is a compatibility surface: saved atlases and TabPFN support
#: sets are aligned by position, so append new features rather than inserting.
FEATURES: tuple[str, ...] = (
    "a_points",
    "b_visits",
    "c_days",
    "d_dwell_s",
    "e_median_speed",
    "f_p90_speed",
    "g_path_len_m",
    "h_straightness",
    "i_turn_rate",
    "j_distance_from_anchor_m",
    "k_hour_mean",
    "l_hour_spread",
    "m_weekend_frac",
    "n_night_frac",
    "o_climb_per_100m",
    "p_revisit_span_days",
)

FEATURE_DOCS: dict[str, str] = {
    "a_points": "GPS fixes recorded inside the hex",
    "b_visits": "separate arrivals (a run of fixes counted once)",
    "c_days": "distinct calendar days the hex was entered",
    "d_dwell_s": "seconds spent inside the hex",
    "e_median_speed": "median speed while inside, m/s",
    "f_p90_speed": "90th percentile speed inside, m/s",
    "g_path_len_m": "distance walked inside the hex",
    "h_straightness": "net displacement / path length, 0 = looped, 1 = straight through",
    "i_turn_rate": "degrees of heading change per metre walked",
    "j_distance_from_anchor_m": "metres from the anchor point",
    "k_hour_mean": "circular mean hour of day the hex was visited",
    "l_hour_spread": "circular std-dev of visit hour, hours",
    "m_weekend_frac": "fraction of visits on Sat/Sun",
    "n_night_frac": "fraction of visits before 06:00 or after 21:00",
    "o_climb_per_100m": "metres of ascent per 100 m walked, so it does not grow with time spent",
    "p_revisit_span_days": "days between first and last visit",
}


@dataclass
class CellStats:
    """One row of the tabular dataset, plus the raw material for a legend."""

    cell: Hex
    a_points: int = 0
    b_visits: int = 0
    c_days: int = 0
    d_dwell_s: float = math.nan
    e_median_speed: float = math.nan
    f_p90_speed: float = math.nan
    g_path_len_m: float = 0.0
    h_straightness: float = math.nan
    i_turn_rate: float = math.nan
    j_distance_from_anchor_m: float = math.nan
    k_hour_mean: float = math.nan
    l_hour_spread: float = math.nan
    m_weekend_frac: float = math.nan
    n_night_frac: float = math.nan
    o_climb_per_100m: float = math.nan
    p_revisit_span_days: float = math.nan

    def vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in FEATURES]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cell"] = self.cell.key
        return data


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = fraction * (len(ordered) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _median(values: list[float]) -> float:
    return _percentile(values, 0.5)


def clean_path(fixes: list[Fix], spacing_m: float = 5.0) -> list[Fix]:
    """Drop fixes that add no information, so jitter stops inflating lengths.

    Consumer GPS wanders by several metres while standing still. Untreated,
    that wander becomes path length, and path length is what half these
    features are ratios of.
    """
    kept: list[Fix] = []
    for fix in fixes:
        if not kept:
            kept.append(fix)
            continue
        if haversine_m(kept[-1].lat, kept[-1].lon, fix.lat, fix.lon) >= spacing_m:
            kept.append(fix)
    if fixes and (not kept or kept[-1] is not fixes[-1]):
        if len(kept) < 2 or haversine_m(kept[-1].lat, kept[-1].lon, fixes[-1].lat,
                                         fixes[-1].lon) > 0.5:
            kept.append(fixes[-1])
    return kept


def smooth_path(fixes: list[Fix], window: int = 5) -> list[Fix]:
    """Moving average of position over a short time window.

    Phone GPS at 1 Hz wanders by a few metres per fix. A walker covers barely
    one metre in a second, so the wander is larger than the movement and any
    speed taken from two consecutive fixes measures the noise, not the person.
    Averaging a handful of fixes pulls the noise down without moving the path.
    """
    if len(fixes) < 3 or window < 2:
        return list(fixes)
    half = window // 2
    out: list[Fix] = []
    for index, fix in enumerate(fixes):
        lo = max(0, index - half)
        hi = min(len(fixes), index + half + 1)
        chunk = fixes[lo:hi]
        elevations = [c.ele for c in chunk if c.ele is not None]
        out.append(Fix(
            lat=sum(c.lat for c in chunk) / len(chunk),
            lon=sum(c.lon for c in chunk) / len(chunk),
            t=fix.t,
            ele=sum(elevations) / len(elevations) if elevations else None,
        ))
    return out


#: A rise smaller than this between two gated points is barometric and GPS
#: noise, not hill. Without a floor, the noise alone integrates into tens of
#: metres of imaginary climbing over a long walk.
ELEVATION_NOISE_M = 1.0


def climb_per_100m(fixes: list[Fix], path_len_m: float) -> float:
    """Ascent per 100 m walked, in metres.

    Reported per distance rather than per visit so that a hex you spent an
    hour in and a hex you crossed in a minute are on the same scale.
    """
    if path_len_m <= 0 or len(fixes) < 2:
        return math.nan
    seen_elevation = False
    rises = 0.0
    for a, b in zip(fixes, fixes[1:]):
        if a.ele is None or b.ele is None:
            continue
        seen_elevation = True
        delta = b.ele - a.ele
        if delta > ELEVATION_NOISE_M:
            rises += delta
    if not seen_elevation:
        return math.nan
    return rises / path_len_m * 100.0


#: Geometry is measured on points this far apart. At walking pace this is a
#: good few seconds of travel, so the residual GPS noise is small next to the
#: distance actually covered and length stops being inflated.
GEOMETRY_SPACING_M = 8.0

#: Speed is measured across a baseline at least this long, for the same reason.
SPEED_BASELINE_S = 4.0


def baseline_speeds(fixes: list[Fix], baseline_s: float = SPEED_BASELINE_S) -> list[float]:
    """Speeds in m/s, each measured across at least `baseline_s`."""
    speeds: list[float] = []
    timed = [f for f in fixes if f.t is not None]
    if len(timed) < 2:
        return speeds
    index = 0
    for start in range(len(timed) - 1):
        if index <= start:
            index = start + 1
        while index < len(timed) and (timed[index].t - timed[start].t).total_seconds() < baseline_s:
            index += 1
        if index >= len(timed):
            break
        dt = (timed[index].t - timed[start].t).total_seconds()
        if dt > 0:
            speeds.append(
                haversine_m(timed[start].lat, timed[start].lon,
                            timed[index].lat, timed[index].lon) / dt
            )
    return speeds


def _turn_rate(fixes: list[Fix], path_len_m: float, baseline_m: float = 25.0) -> float:
    """Degrees of heading change per metre walked.

    Headings are measured between points a *baseline* apart rather than
    between consecutive samples. Over 25 m of walking the GPS noise is small
    next to the actual direction of travel, so the number means something.
    """
    if path_len_m <= 0 or len(fixes) < 3:
        return math.nan
    anchors: list[Fix] = [fixes[0]]
    accumulated = 0.0
    for fix in fixes[1:]:
        accumulated += haversine_m(anchors[-1].lat, anchors[-1].lon, fix.lat, fix.lon)
        if accumulated >= baseline_m:
            anchors.append(fix)
            accumulated = 0.0
    if len(anchors) < 3:
        return math.nan
    total_turn = 0.0
    for index in range(1, len(anchors) - 1):
        a, b, c = anchors[index - 1], anchors[index], anchors[index + 1]
        vectors = []
        for p, q in ((a, b), (b, c)):
            vectors.append((haversine_m(p.lat, p.lon, q.lat, q.lon), p, q))
        (_, p1, q1), (_, p2, q2) = vectors
        # bearings are computed on the sphere, so this is independent of the
        # local projection
        bearing1 = _bearing_deg(p1.lat, p1.lon, q1.lat, q1.lon)
        bearing2 = _bearing_deg(p2.lat, p2.lon, q2.lat, q2.lon)
        delta = abs((bearing2 - bearing1 + 180.0) % 360.0 - 180.0)
        total_turn += delta
    return total_turn / path_len_m


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _circular_hour_stats(hours: list[float]) -> tuple[float, float]:
    """Mean hour and spread on a 24-hour circle (23:00 is next to 01:00)."""
    if not hours:
        return math.nan, math.nan
    angles = [h / 24.0 * 2 * math.pi for h in hours]
    sin_sum = sum(math.sin(a) for a in angles) / len(angles)
    cos_sum = sum(math.cos(a) for a in angles) / len(angles)
    mean = math.atan2(sin_sum, cos_sum)
    mean_hour = (mean / (2 * math.pi) * 24.0) % 24.0
    resultant = math.hypot(sin_sum, cos_sum)
    if resultant <= 1e-9:
        return mean_hour, math.nan
    spread = math.sqrt(max(0.0, -2.0 * math.log(min(1.0, resultant)))) / (2 * math.pi) * 24.0
    return mean_hour, spread


def _run_geometry(fixes: list[Fix]) -> dict:
    """Geometry and speed for one continuous stay inside one hex."""
    cleaned = clean_path(smooth_path(fixes), GEOMETRY_SPACING_M)
    out = {
        "path_len": 0.0,
        "straightness": math.nan,
        "turn_rate": math.nan,
        "speeds": [],
        "climb_per_100m": math.nan,
        "dwell": 0.0,
        "hours": [],
        "weekend": 0,
        "night": 0,
        "timed": 0,
        "first": None,
        "last": None,
    }
    if len(cleaned) >= 2:
        path_len = 0.0
        for a, b in zip(cleaned, cleaned[1:]):
            path_len += haversine_m(a.lat, a.lon, b.lat, b.lon)
        out["path_len"] = path_len
        if path_len > 0:
            net = haversine_m(cleaned[0].lat, cleaned[0].lon, cleaned[-1].lat, cleaned[-1].lon)
            out["straightness"] = min(1.0, net / path_len)
        out["turn_rate"] = _turn_rate(cleaned, path_len)
        out["speeds"] = baseline_speeds(cleaned)
        out["climb_per_100m"] = climb_per_100m(cleaned, path_len)

    stamps = [f.t for f in fixes if f.t is not None]
    if len(stamps) >= 2:
        out["dwell"] = (max(stamps) - min(stamps)).total_seconds()
    for fix in fixes:
        if fix.t is None:
            continue
        out["timed"] += 1
        hour = fix.t.hour + fix.t.minute / 60.0
        out["hours"].append(hour)
        if fix.t.weekday() >= 5:
            out["weekend"] += 1
        if hour < 6.0 or hour >= 21.0:
            out["night"] += 1
        if out["first"] is None or fix.t < out["first"]:
            out["first"] = fix.t
        if out["last"] is None or fix.t > out["last"]:
            out["last"] = fix.t
    return out


#: An excursion into a neighbouring hex shorter than this is treated as the
#: walker still being in the hex they came from. Standing near a boundary, GPS
#: noise alone will flip the assignment many times a minute, and counting each
#: flip as an arrival turns "one visit" into "eighty visits".
MIN_RUN_S = 25.0


def _span_seconds(fixes: list[Fix]) -> float | None:
    stamps = [f.t for f in fixes if f.t is not None]
    if len(stamps) < 2:
        return None
    return (max(stamps) - min(stamps)).total_seconds()


def _is_brief(fixes: list[Fix], min_run_s: float) -> bool:
    """Is this stay short enough to be boundary noise?

    A single-fix stay has no measurable duration, but it is by construction
    the briefest possible thing, so it counts as brief rather than unknown.
    """
    span = _span_seconds(fixes)
    if span is None:
        return len(fixes) <= 2
    return span < min_run_s


def _segments(fixes: list[Fix], grid: HexGrid, min_run_s: float = MIN_RUN_S):
    """Split a walk into per-hex stays, returning (cell, start, end) ranges.

    Merging a short excursion joins three consecutive ranges, so every range
    stays a contiguous slice of the original walk.
    """
    ranges: list[list] = []
    for index, fix in enumerate(fixes):
        cell = grid.to_hex(fix.lat, fix.lon)
        if ranges and ranges[-1][0] == cell:
            ranges[-1][2] = index + 1
        else:
            ranges.append([cell, index, index + 1])

    changed = True
    while changed and len(ranges) > 2:
        changed = False
        for index in range(1, len(ranges) - 1):
            if ranges[index - 1][0] != ranges[index + 1][0]:
                continue
            if _is_brief(fixes[ranges[index][1]:ranges[index][2]], min_run_s):
                ranges[index - 1][2] = ranges[index + 1][2]
                del ranges[index:index + 2]
                changed = True
                break
    return [(cell, start, end) for cell, start, end in ranges]


def _accumulate(acc: dict, walk: Walk, grid: HexGrid) -> None:
    """Fold one walk into per-hex stays.

    Hex assignment runs on the smoothed track, so standing beside a boundary
    does not stampede between cells, but each stay keeps the *original* fixes
    so nothing downstream is measured on already-averaged data. Smoothing
    preserves index alignment, so a range found on one applies to both.
    """
    smoothed = smooth_path(walk.fixes)
    if len(smoothed) != len(walk.fixes):  # pragma: no cover - defensive
        smoothed = walk.fixes
    for cell, start, end in _segments(smoothed, grid):
        acc.setdefault(cell.key, []).append(walk.fixes[start:end])


def extract_cells(walks: list[Walk], grid: HexGrid) -> dict[Hex, CellStats]:
    """Build the tabular dataset: one :class:`CellStats` per visited hex."""
    acc: dict[str, list[list[Fix]]] = {}
    for walk in walks:
        _accumulate(acc, walk, grid)

    cells: dict[Hex, CellStats] = {}
    for key, runs in acc.items():
        cell = Hex.parse(key)
        stats = CellStats(cell=cell)
        speeds: list[float] = []
        hours: list[float] = []
        days: set[str] = set()
        weekend = night = timed_points = 0
        dwell = path_len = 0.0
        climb_weighted = climb_len = 0.0
        turn_weighted = 0.0
        turn_len = 0.0
        straightness_samples: list[float] = []
        first_ts = last_ts = None

        for fixes in runs:
            stats.b_visits += 1
            stats.a_points += len(fixes)
            geometry = _run_geometry(fixes)
            path_len += geometry["path_len"]
            if not math.isnan(geometry["climb_per_100m"]) and geometry["path_len"] > 0:
                climb_weighted += geometry["climb_per_100m"] * geometry["path_len"]
                climb_len += geometry["path_len"]
            dwell += geometry["dwell"]
            speeds.extend(geometry["speeds"])
            hours.extend(geometry["hours"])
            weekend += geometry["weekend"]
            night += geometry["night"]
            timed_points += geometry["timed"]
            if not math.isnan(geometry["straightness"]):
                straightness_samples.append(geometry["straightness"])
            if not math.isnan(geometry["turn_rate"]) and geometry["path_len"] > 0:
                turn_weighted += geometry["turn_rate"] * geometry["path_len"]
                turn_len += geometry["path_len"]
            if geometry["first"] is not None:
                first_ts = geometry["first"] if first_ts is None else min(first_ts, geometry["first"])
            if geometry["last"] is not None:
                last_ts = geometry["last"] if last_ts is None else max(last_ts, geometry["last"])
            for fix in fixes:
                if fix.t is not None:
                    days.add(fix.t.strftime("%Y-%m-%d"))

        stats.c_days = len(days)
        stats.g_path_len_m = path_len
        if speeds:
            stats.e_median_speed = _median(speeds)
            stats.f_p90_speed = _percentile(speeds, 0.9)
        if turn_len > 0:
            stats.i_turn_rate = turn_weighted / turn_len
        if straightness_samples:
            stats.h_straightness = sum(straightness_samples) / len(straightness_samples)
        if timed_points:
            stats.d_dwell_s = dwell
            stats.m_weekend_frac = weekend / timed_points
            stats.n_night_frac = night / timed_points
            mean_hour, spread = _circular_hour_stats(hours)
            stats.k_hour_mean = mean_hour
            stats.l_hour_spread = spread
            if first_ts is not None and last_ts is not None:
                stats.p_revisit_span_days = (last_ts - first_ts).total_seconds() / 86400.0
        if climb_len > 0:
            stats.o_climb_per_100m = climb_weighted / climb_len
        cx, cy = grid.center_xy(cell)
        stats.j_distance_from_anchor_m = math.hypot(cx, cy)
        cells[cell] = stats
    return cells


@dataclass
class Evidence:
    """How much of a hex was actually walked, and whether that is enough."""

    min_points: int = 6
    min_path_m: float = 25.0
    min_dwell_s: float = 60.0
    min_days: int = 1

    def passes(self, stats: CellStats) -> tuple[bool, str]:
        if stats.a_points < self.min_points:
            return False, f"only {stats.a_points} GPS fixes"
        if stats.g_path_len_m < self.min_path_m:
            return False, f"only {stats.g_path_len_m:.0f} m walked"
        dwell = stats.d_dwell_s
        if not math.isnan(dwell) and dwell < self.min_dwell_s:
            return False, f"only {dwell:.0f} s inside"
        if stats.c_days < self.min_days:
            return False, "no dated visits"
        return True, ""


def charted_cells(
    stats: dict[Hex, CellStats], evidence: Evidence | None = None
) -> tuple[dict[Hex, CellStats], dict[Hex, str]]:
    """Split cells into charted and too-thin-to-chart.

    A map that inks in a hex because someone's GPS clipped the corner of it
    would be a map that lies. Crossing a place is not visiting it, so a hex
    has to accumulate real walking before it earns its outline.
    """
    evidence = evidence or Evidence()
    charted: dict[Hex, CellStats] = {}
    rejected: dict[Hex, str] = {}
    for cell, cell_stats in stats.items():
        ok, why = evidence.passes(cell_stats)
        if ok:
            charted[cell] = cell_stats
        else:
            rejected[cell] = why
    return charted, rejected
