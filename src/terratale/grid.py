"""Flat-top hexagonal grid over a local metre projection.

Hexes are used rather than square tiles because a hex has a single
neighbourhood radius in every direction: "did the walker actually get here"
means the same thing no matter which way the street runs. Axial coordinates
keep the storage and the distance function trivial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geo import Projection

SQRT3 = math.sqrt(3.0)


@dataclass(frozen=True, order=True)
class Hex:
    q: int
    r: int

    @property
    def key(self) -> str:
        return f"{self.q}:{self.r}"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.key

    @staticmethod
    def parse(text: str) -> "Hex":
        q, _, r = text.partition(":")
        return Hex(int(q), int(r))

    def neighbours(self) -> list["Hex"]:
        return [
            Hex(self.q + 1, self.r),
            Hex(self.q + 1, self.r - 1),
            Hex(self.q, self.r - 1),
            Hex(self.q - 1, self.r),
            Hex(self.q - 1, self.r + 1),
            Hex(self.q, self.r + 1),
        ]

    def ring(self, radius: int) -> list["Hex"]:
        if radius <= 0:
            return [self]
        out: list[Hex] = []
        cursor = Hex(self.q + radius, self.r - radius)
        for direction in (0, 1, 2, 3, 4, 5):
            for _ in range(radius):
                out.append(cursor)
                cursor = cursor._step(direction)
        return out

    def _step(self, direction: int) -> "Hex":
        moves = [
            (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0),
        ][direction % 6]
        return Hex(self.q + moves[0], self.r + moves[1])


def hex_distance(a: Hex, b: Hex) -> int:
    dq, dr = a.q - b.q, a.r - b.r
    return (abs(dq) + abs(dq + dr) + abs(dr)) // 2


def _cube_round(qf: float, rf: float) -> Hex:
    """Round fractional axial coordinates to the nearest hex (cube rounding)."""
    xf, zf = qf, rf
    yf = -xf - zf
    rx, ry, rz = round(xf), round(yf), round(zf)
    dx, dy, dz = abs(rx - xf), abs(ry - yf), abs(rz - zf)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return Hex(int(rx), int(rz))


@dataclass(frozen=True)
class HexGrid:
    """A fixed lattice of hexes, `size_m` from centre to vertex."""

    projection: Projection
    size_m: float = 120.0

    def to_hex(self, lat: float, lon: float) -> Hex:
        x, y = self.projection.to_xy(lat, lon)
        qf = (2.0 / 3.0 * x) / self.size_m
        rf = (-1.0 / 3.0 * x + SQRT3 / 3.0 * y) / self.size_m
        return _cube_round(qf, rf)

    def center(self, cell: Hex) -> tuple[float, float]:
        """Return the (lat, lon) of a hex centre."""
        x = self.size_m * 1.5 * cell.q
        y = self.size_m * (SQRT3 / 2.0 * cell.q + SQRT3 * cell.r)
        return self.projection.to_latlon(x, y)

    def center_xy(self, cell: Hex) -> tuple[float, float]:
        x = self.size_m * 1.5 * cell.q
        y = self.size_m * (SQRT3 / 2.0 * cell.q + SQRT3 * cell.r)
        return x, y

    def corners_xy(self, cell: Hex) -> list[tuple[float, float]]:
        """Flat-top hexagon vertices, in order, in metres."""
        cx, cy = self.center_xy(cell)
        return [
            (
                cx + self.size_m * math.cos(math.radians(60 * i)),
                cy + self.size_m * math.sin(math.radians(60 * i)),
            )
            for i in range(6)
        ]

    def viewport(self, cells) -> tuple[float, float, float, float]:
        """Bounding box (min_x, min_y, max_x, max_y) in metres over `cells`."""
        xs: list[float] = []
        ys: list[float] = []
        for cell in cells:
            for x, y in self.corners_xy(cell):
                xs.append(x)
                ys.append(y)
        if not xs:
            return -1.0, -1.0, 1.0, 1.0
        return min(xs), min(ys), max(xs), max(ys)
