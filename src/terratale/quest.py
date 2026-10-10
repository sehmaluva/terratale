"""Choosing where the walker should go next.

The map is only half of the mechanic. A blank region with no instruction is
just an unfinished document; the quest is what turns it into a reason to put
shoes on. Candidates are the uncharted hexes touching chartered ground, scored
to prefer ground the walker has not been anywhere near before, so the map
spreads out instead of thickening in one corner.
"""

from __future__ import annotations

import math

from .grid import Hex, hex_distance
from .render import wrap_text  # re-exported for callers building cards

COMPASS_16 = (
    "north", "north-north-east", "north-east", "east-north-east",
    "east", "east-south-east", "south-east", "south-south-east",
    "south", "south-south-west", "south-west", "west-south-west",
    "west", "west-north-west", "north-west", "north-north-west",
)


def compass_label(bearing_deg: float) -> str:
    index = int(((bearing_deg % 360.0) + 11.25) // 22.5) % 16
    return COMPASS_16[index]


def _frontier(atlas, ring: int = 1) -> list[Hex]:
    revealed = {Hex.parse(key) for key in atlas.cells}
    candidates: set[Hex] = set()
    for cell in revealed:
        for neighbour in cell.ring(ring):
            if neighbour not in revealed:
                candidates.add(neighbour)
    return sorted(candidates)


def _bearing(origin_lat: float, origin_lon: float, lat: float, lon: float) -> float:
    p1, p2 = math.radians(origin_lat), math.radians(lat)
    dl = math.radians(lon - origin_lon)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def suggest_quests(atlas, count: int = 1, max_km: float = 3.0) -> list[dict]:
    """Return up to `count` suggestions, nearest-and-freshest first."""
    from .geo import haversine_m

    grid = atlas.grid
    revealed = [Hex.parse(key) for key in atlas.cells]
    candidates = _frontier(atlas, ring=1)

    if not candidates:
        # Nothing charted at all: send them out on the first leg in a
        # direction, rather than demanding they understand the grid.
        if not revealed:
            candidates = Hex(0, 0).ring(2)
        else:
            candidates = _frontier(atlas, ring=2)

    anchor_lat, anchor_lon = grid.center(Hex(0, 0))
    scored: list[tuple[float, Hex, float]] = []
    for cell in candidates:
        lat, lon = grid.center(cell)
        distance = haversine_m(anchor_lat, anchor_lon, lat, lon)
        if distance > max_km * 1000.0:
            continue
        # freshness: how empty the neighbourhood of this candidate is
        charted_nearby = sum(
            1 for neighbour in [cell] + cell.neighbours()
            if neighbour.key in atlas.cells
        )
        # a small random-free tiebreak on hex parity keeps the pick stable
        score = charted_nearby * 1.0 + hex_distance(Hex(0, 0), cell) * 0.35
        scored.append((score + abs(distance - 600.0) / 1000.0, cell, distance))

    if not scored:
        scored = [
            (0.0, cell, haversine_m(anchor_lat, anchor_lon, *grid.center(cell)))
            for cell in candidates[:8]
        ]

    scored.sort(key=lambda item: item[0])
    out: list[dict] = []
    for _score, cell, distance in scored[:count]:
        lat, lon = grid.center(cell)
        bearing = _bearing(anchor_lat, anchor_lon, lat, lon)
        label = compass_label(bearing)
        out.append({
            "cell": cell.key,
            "lat": lat,
            "lon": lon,
            "distance_m": distance,
            "bearing_deg": bearing,
            "bearing_compass": label,
            "headline": f"Chart the hex {distance / 1000.0:.2f} km {label}",
            "instruction": (
                f"Walk to roughly {lat:.5f}, {lon:.5f} — about "
                f"{distance / 1000.0:.2f} km {label} of your anchor. "
                "Stay a few minutes and move around so the trace records a real visit, "
                "then bring the map home."
            ),
        })
    return out


def summarise_progress(atlas) -> str:
    """One line on how much of the neighbourhood is charted."""
    charted = atlas.revealed_count()
    if charted == 0:
        return "The map is blank. Nothing is charted."
    km = atlas.total_walk_distance_m() / 1000.0
    named = sum(1 for record in atlas.cells.values() if record.name)
    return (f"{charted} hexes charted across {len(atlas.walks)} walks ({km:.1f} km on foot); "
            f"{named} places named.")
