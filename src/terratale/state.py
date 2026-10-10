"""The atlas: the on-disk state that survives between walks.

Kept as plain JSON next to your own data on purpose. A map of everywhere you
walk is about as personal as data gets, and there is no account, no sync and
no server in this project to leak it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .grid import HexGrid
from .geo import Projection

DEFAULT_STATE_NAME = "atlas.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CellRecord:
    """A hex that has been revealed, and everything the map knows about it."""

    cell: str
    archetype: str
    confidence: float = 0.0
    name: str = ""
    legend: str = ""
    inscription: str = ""
    author: str = ""
    first_seen: str = ""
    last_seen: str = ""
    revisions: int = 0
    walks: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cell": self.cell,
            "archetype": self.archetype,
            "confidence": round(self.confidence, 4),
            "name": self.name,
            "legend": self.legend,
            "inscription": self.inscription,
            "author": self.author,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "revisions": self.revisions,
            "walks": self.walks,
            "stats": self.stats,
        }

    @staticmethod
    def from_dict(data: dict) -> "CellRecord":
        return CellRecord(
            cell=data["cell"],
            archetype=data.get("archetype", "unknown"),
            confidence=float(data.get("confidence", 0.0)),
            name=data.get("name", ""),
            legend=data.get("legend", ""),
            inscription=data.get("inscription", ""),
            author=data.get("author", ""),
            first_seen=data.get("first_seen", ""),
            last_seen=data.get("last_seen", ""),
            revisions=int(data.get("revisions", 0)),
            walks=list(data.get("walks", [])),
            stats=dict(data.get("stats", {})),
        )


@dataclass
class Atlas:
    """Everything the map has learned, in one place."""

    anchor_lat: float = 0.0
    anchor_lon: float = 0.0
    hex_size_m: float = 120.0
    title: str = "The Unwalked Neighbourhood"
    cells: dict[str, CellRecord] = field(default_factory=dict)
    labelled: dict[str, str] = field(default_factory=dict)
    walks: list[dict] = field(default_factory=list)
    model: str = "gemma4:e2b"
    backend: str = "auto"
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)

    @property
    def grid(self) -> HexGrid:
        return HexGrid(Projection(self.anchor_lat, self.anchor_lon), self.hex_size_m)

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "title": self.title,
            "anchor": {"lat": self.anchor_lat, "lon": self.anchor_lon},
            "hex_size_m": self.hex_size_m,
            "model": self.model,
            "backend": self.backend,
            "created": self.created,
            "updated": self.updated,
            "labelled": self.labelled,
            "walks": self.walks,
            "cells": [record.to_dict() for record in self.cells.values()],
        }

    @staticmethod
    def from_dict(data: dict) -> "Atlas":
        anchor = data.get("anchor", {})
        atlas = Atlas(
            anchor_lat=float(anchor.get("lat", 0.0)),
            anchor_lon=float(anchor.get("lon", 0.0)),
            hex_size_m=float(data.get("hex_size_m", 120.0)),
            title=data.get("title", "The Unwalked Neighbourhood"),
            model=data.get("model", "gemma4:e2b"),
            backend=data.get("backend", "auto"),
            created=data.get("created", _now()),
            updated=data.get("updated", _now()),
            labelled=dict(data.get("labelled", {})),
            walks=list(data.get("walks", [])),
        )
        for raw in data.get("cells", []):
            record = CellRecord.from_dict(raw)
            atlas.cells[record.cell] = record
        return atlas

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self.updated = _now()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> "Atlas":
        return Atlas.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def load_or_create(path: str | Path) -> "Atlas":
        path = Path(path)
        if path.exists():
            return Atlas.load(path)
        return Atlas()

    # -- queries ---------------------------------------------------------

    def has_walk(self, name: str) -> bool:
        return any(entry.get("name") == name for entry in self.walks)

    def taken_names(self) -> list[str]:
        return [record.name for record in self.cells.values() if record.name]

    def revealed_count(self) -> int:
        return len(self.cells)

    def total_walk_distance_m(self) -> float:
        return sum(float(entry.get("distance_m", 0.0)) for entry in self.walks)


#: Trace points are stored at roughly this spacing. Fine enough that the drawn
#: line still matches the walk, coarse enough that a year of daily walks stays
#: a file you can email.
TRACE_SPACING_M = 6.0


def store_walk(atlas: Atlas, walk, min_spacing_m: float = TRACE_SPACING_M) -> dict:
    """Downsample a walk into the atlas and return the stored record.

    The atlas, not the GPX file, is the source of truth afterwards: features
    and the printed map are both rebuilt from this, so importing is done once
    and the original export can be archived or deleted.
    """
    from .features import GEOMETRY_SPACING_M, clean_path, smooth_path
    from .geo import haversine_m

    # Distance is reported from the smoothed track. Summing a raw 1 Hz GPS
    # trace measures the receiver's noise as much as the walk -- on real
    # traces it can overstate the distance several times over.
    measured = clean_path(smooth_path(walk.fixes), GEOMETRY_SPACING_M)
    total = 0.0
    for a, b in zip(measured, measured[1:]):
        total += haversine_m(a.lat, a.lon, b.lat, b.lon)

    points: list[list[float]] = []
    last_lat = last_lon = None
    for fix in walk.fixes:
        # Always keep the first fix, then one per `min_spacing_m`; the final
        # fix is added below so a walk never loses its endpoint.
        if last_lat is not None and haversine_m(last_lat, last_lon, fix.lat, fix.lon) < min_spacing_m:
            continue
        stamp = int(fix.t.timestamp()) if fix.t is not None else None
        points.append([round(fix.lat, 6), round(fix.lon, 6), stamp])
        last_lat, last_lon = fix.lat, fix.lon

    if walk.fixes:
        tail = walk.fixes[-1]
        if not points or points[-1][0] != round(tail.lat, 6) or points[-1][1] != round(tail.lon, 6):
            stamp = int(tail.t.timestamp()) if tail.t is not None else None
            points.append([round(tail.lat, 6), round(tail.lon, 6), stamp])

    started = walk.start()
    record = {
        "name": walk.name,
        "source": walk.source or "imported",
        "imported": _now(),
        "started": started.isoformat() if started else None,
        "fixes": len(walk.fixes),
        "stored_points": len(points),
        "distance_m": round(total, 1),
        "points": points,
    }
    atlas.walks.append(record)
    return record


def walks_from_atlas(atlas: Atlas) -> list:
    """Rebuild timed :class:`Walk` objects from stored traces."""
    from datetime import datetime as _dt

    from .geo import Fix, Walk

    walks = []
    for entry in atlas.walks:
        fixes = []
        for lat, lon, stamp in entry.get("points", []):
            when = _dt.fromtimestamp(stamp, tz=timezone.utc) if stamp else None
            fixes.append(Fix(lat=lat, lon=lon, t=when))
        walks.append(Walk(name=entry.get("name", "walk"), fixes=fixes,
                          source=entry.get("source")))
    return walks
