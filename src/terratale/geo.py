"""Geodesy and GPS trace loading.

Everything here is dependency-free on purpose: a map that only works when you
have a package index reachable is a map that does not work in the field.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class Fix:
    """A single GPS sample, with time normalised to UTC-aware."""

    lat: float
    lon: float
    t: datetime | None = None
    ele: float | None = None
    accuracy_m: float | None = None

    @property
    def has_time(self) -> bool:
        return self.t is not None


@dataclass
class Walk:
    """One recorded outing."""

    name: str
    fixes: list[Fix] = field(default_factory=list)
    source: str | None = None

    def __len__(self) -> int:
        return len(self.fixes)

    @property
    def timed(self) -> bool:
        return sum(1 for f in self.fixes if f.has_time)

    def start(self) -> datetime | None:
        for f in self.fixes:
            if f.t is not None:
                return f.t
        return None

    def duration_s(self) -> float:
        stamps = [f.t for f in self.fixes if f.t is not None]
        if len(stamps) < 2:
            return 0.0
        return (stamps[-1] - stamps[0]).total_seconds()


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


@dataclass(frozen=True)
class Projection:
    """Local equirectangular projection to metres around an anchor.

    A neighbourhood is small enough that the error from treating the Earth as
    flat over a few kilometres is well under GPS noise, and the resulting
    coordinates are metres, which makes every downstream threshold legible.
    """

    lat0: float
    lon0: float

    def to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        x = math.radians(lon - self.lon0) * math.cos(math.radians(self.lat0)) * EARTH_RADIUS_M
        y = math.radians(lat - self.lat0) * EARTH_RADIUS_M
        return x, y

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        lat = self.lat0 + math.degrees(y / EARTH_RADIUS_M)
        lon = self.lon0 + math.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(self.lat0))))
        return lat, lon


def _norm_time(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(node: ET.Element, *names: str) -> str | None:
    for child in node:
        if _localname(child.tag) in names and child.text:
            return child.text
    return None


def _fix_from_trackpoint(node: ET.Element) -> Fix | None:
    try:
        lat = float(node.attrib["lat"])
        lon = float(node.attrib["lon"])
    except (KeyError, ValueError):
        return None

    ele_raw = _child_text(node, "ele")
    try:
        ele = float(ele_raw) if ele_raw is not None else None
    except ValueError:
        ele = None

    acc_raw = _child_text(node, "hdop")
    try:
        # hdop is a unitless dilution factor; ~5 m per unit is the usual
        # consumer-grade rule of thumb, and it only ever gates plausibility.
        accuracy = float(acc_raw) * 5.0 if acc_raw is not None else None
    except ValueError:
        accuracy = None

    return Fix(lat=lat, lon=lon, t=_norm_time(_child_text(node, "time")), ele=ele, accuracy_m=accuracy)


def load_gpx(path: str | Path) -> Walk:
    """Load a GPX track (the format every phone fitness app can export)."""
    path = Path(path)
    root = ET.fromstring(path.read_text(encoding="utf-8", errors="replace"))
    fixes: list[Fix] = []
    for node in root.iter():
        if _localname(node.tag) in {"trkpt", "wpt", "rtept"}:
            fix = _fix_from_trackpoint(node)
            if fix is not None:
                fixes.append(fix)
    if not fixes:
        raise ValueError(f"{path}: no track points found")
    return Walk(name=path.stem, fixes=fixes, source=str(path))


def load_walk(path: str | Path) -> Walk:
    """Load any supported trace format."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".gpx":
        return load_gpx(path)
    raise ValueError(f"{path}: unsupported trace format {suffix!r} (supported: .gpx)")


def segment_speeds(walk: Walk) -> list[float]:
    """Per-segment speed in m/s, skipping segments with no usable timing."""
    speeds: list[float] = []
    previous: Fix | None = None
    for fix in walk.fixes:
        if previous is not None:
            if previous.t is not None and fix.t is not None:
                dt = (fix.t - previous.t).total_seconds()
                if dt > 0:
                    speeds.append(haversine_m(previous.lat, previous.lon, fix.lat, fix.lon) / dt)
        previous = fix
    return speeds
