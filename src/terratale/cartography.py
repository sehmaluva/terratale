"""Composing the map sheet from an atlas.

Layout is fixed rather than adaptive: the title band at the top, the drawn
map in the middle, the key and the cartouche in the bottom band. A map people
print every week should not rearrange itself between printings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .archetypes import ARCHETYPE_BY_ID
from .geo import haversine_m
from .grid import Hex, HexGrid
from .render import (
    INK, INK_FAINT, PARCHMENT, PARCHMENT_DEEP, Canvas, Dot, Ellipse, Poly, Text,
    marker_for, marker_prims, to_pdf, to_png, to_svg, wobble, wrap_paragraph,
)

MM = 2.834645669  # points per millimetre

PAGE_SIZES = {
    "a4": (210.0, 297.0),
    "a3": (297.0, 420.0),
    "a5": (148.0, 210.0),
    "letter": (215.9, 279.4),
    "tabloid": (279.4, 431.8),
}


@dataclass
class MapOptions:
    page: str = "a3"
    #: "auto" picks whichever orientation wastes less paper for the shape of
    #: the ground actually charted. A fortnight of walks is rarely the same
    #: shape twice, and a fixed orientation strands a lot of blank parchment.
    orientation: str = "auto"
    margin_mm: float = 12.0
    title: str | None = None
    show_traces: bool = True
    show_ghosts: bool = True
    dpi: float = 300.0

    def page_points(self, landscape: bool) -> tuple[float, float]:
        width_mm, height_mm = PAGE_SIZES.get(self.page, PAGE_SIZES["a3"])
        if landscape:
            width_mm, height_mm = max(width_mm, height_mm), min(width_mm, height_mm)
        return width_mm * MM, height_mm * MM


@dataclass
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def span_x(self) -> float:
        return max(self.max_x - self.min_x, 1.0)

    def span_y(self) -> float:
        return max(self.max_y - self.min_y, 1.0)


def _transform(bounds: Bounds, rect: tuple[float, float, float, float]):
    rx, ry, rw, rh = rect
    scale = min(rw / bounds.span_x(), rh / bounds.span_y())
    mid_x = (bounds.min_x + bounds.max_x) / 2.0
    mid_y = (bounds.min_y + bounds.max_y) / 2.0
    cx = rx + rw / 2.0
    cy = ry + rh / 2.0

    def project(x: float, y: float) -> tuple[float, float]:
        return cx + (x - mid_x) * scale, cy - (y - mid_y) * scale

    return project, scale


def _bounds_for(grid: HexGrid, cells: list[Hex], pad_m: float) -> Bounds:
    if not cells:
        cells = [Hex(0, 0)]
        for radius in (1, 2):
            cells.extend(Hex(0, 0).ring(radius))
    pts: list[tuple[float, float]] = []
    for cell in cells:
        pts.extend(grid.corners_xy(cell))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return Bounds(min(xs) - pad_m, min(ys) - pad_m, max(xs) + pad_m, max(ys) + pad_m)


def _compass(canvas: Canvas, cx: float, cy: float, radius: float) -> None:
    canvas.add(Ellipse(cx, cy, radius, radius, fill=None, stroke=INK_FAINT, width=0.8, opacity=0.7))
    canvas.add(Ellipse(cx, cy, radius * 0.72, radius * 0.72, fill=None, stroke=INK_FAINT,
                       width=0.5, opacity=0.5))
    for index, (dx, dy, length) in enumerate((
        (0.0, -1.0, 1.0), (1.0, 0.0, 0.62), (0.0, 1.0, 0.62), (-1.0, 0.0, 0.62),
    )):
        tip = (cx + dx * radius * length, cy + dy * radius * length)
        left = (cx - dy * radius * 0.11, cy + dx * radius * 0.11)
        right = (cx + dy * radius * 0.11, cy - dx * radius * 0.11)
        canvas.add(Poly([tip, left, right], fill=INK if index == 0 else PARCHMENT,
                        stroke=INK, width=0.7, opacity=0.9 if index == 0 else 0.7))
    canvas.add(Text(cx, cy - radius - 7.0, "N", size=9, anchor="middle", bold=True, fill=INK))


def _scale_bar(canvas: Canvas, project, rect, scale: float) -> None:
    lengths_m = [100, 200, 500, 1000, 2000, 5000]
    target_px = 150.0
    chosen = min(lengths_m, key=lambda m: abs(m * scale - target_px))
    segments = 4 if chosen >= 500 else 2
    seg_px = chosen * scale / segments
    x0 = rect[0] + 14
    y0 = rect[1] + rect[3] - 26
    label = f"{chosen} m" if chosen < 1000 else f"{chosen // 1000} km"
    for index in range(segments):
        fill = INK if index % 2 == 0 else PARCHMENT
        canvas.add(Poly([(x0 + index * seg_px, y0), (x0 + (index + 1) * seg_px, y0),
                         (x0 + (index + 1) * seg_px, y0 + 5), (x0 + index * seg_px, y0 + 5)],
                        fill=fill, stroke=INK, width=0.7, opacity=0.9))
    canvas.add(Text(x0, y0 - 4, label, size=7.5, fill=INK_FAINT))
    canvas.add(Text(x0, y0 + 15, "walked distances, measured from GPS", size=6.5, fill=INK_FAINT))


def _title_band(canvas: Canvas, atlas, options: MapOptions, rect) -> None:
    title = options.title or atlas.title
    canvas.add(Text(canvas.width / 2, rect[1] + 26, title, size=26, anchor="middle",
                    bold=True, fill=INK))
    cells = atlas.revealed_count()
    km = atlas.total_walk_distance_m() / 1000.0
    subtitle = (
        f"{cells} hex{'es' if cells != 1 else ''} charted  ·  "
        f"{len(atlas.walks)} walk{'s' if len(atlas.walks) != 1 else ''} logged  ·  "
        f"{km:.1f} km on foot"
    )
    canvas.add(Text(canvas.width / 2, rect[1] + 46, subtitle, size=10.5, anchor="middle",
                    fill=INK_FAINT))
    y = rect[1] + 62
    canvas.add(Poly([(rect[0] + rect[2] * 0.28, y), (rect[0] + rect[2] * 0.72, y)],
                    stroke=INK_FAINT, width=0.8, opacity=0.6, close=False))


def _draw_hex(canvas: Canvas, prims_points, fill, stroke, width, opacity, dash=None):
    canvas.add(Poly(prims_points, fill=fill, stroke=stroke, width=width,
                    opacity=opacity, dash=dash))


#: Reserved bands on the sheet, in points: the title at the top, the key and
#: cartouche along the bottom.
TITLE_BAND = 74.0
BOTTOM_BAND = 186.0


def _map_rect(page_w: float, page_h: float, margin: float):
    return (margin + 16, margin + TITLE_BAND,
            page_w - 2 * margin - 32, page_h - 2 * margin - TITLE_BAND - BOTTOM_BAND)


def _best_orientation(candidates, bounds, page: str, margin: float) -> bool:
    """Choose the orientation that draws this ground largest."""
    best_scale, best_landscape = -1.0, True
    for landscape in (True, False):
        width_mm, height_mm = PAGE_SIZES.get(page, PAGE_SIZES["a3"])
        if landscape:
            width_mm, height_mm = max(width_mm, height_mm), min(width_mm, height_mm)
        rect = _map_rect(width_mm * MM, height_mm * MM, margin)
        scale = min(rect[2] / bounds.span_x(), rect[3] / bounds.span_y())
        if scale > best_scale:
            best_scale, best_landscape = scale, landscape
    del candidates
    return best_landscape


def build_map(atlas, options: MapOptions | None = None) -> Canvas:
    """Render the atlas into a printable map sheet."""
    options = options or MapOptions()
    margin = options.margin_mm * MM
    grid = atlas.grid

    revealed = [Hex.parse(key) for key in sorted(atlas.cells)]
    ghost_cells: list[Hex] = []
    if options.show_ghosts:
        seen = set(revealed)
        for cell in revealed:
            for neighbour in cell.neighbours():
                if neighbour not in seen and neighbour not in ghost_cells:
                    ghost_cells.append(neighbour)
        if not revealed:
            ghost_cells = Hex(0, 0).ring(1) + Hex(0, 0).ring(2)

    bounds = _bounds_for(grid, revealed or ghost_cells, pad_m=grid.size_m * 0.6)

    if options.orientation == "auto":
        landscape = _best_orientation(revealed or ghost_cells, bounds, options.page, margin)
    else:
        landscape = options.orientation != "portrait"
    page_w, page_h = options.page_points(landscape)

    canvas = Canvas(page_w, page_h, background=PARCHMENT)
    inner = (margin, margin, page_w - 2 * margin, page_h - 2 * margin)
    map_rect = _map_rect(page_w, page_h, margin)

    # outer rule, the way a drawn sheet is bordered
    canvas.add(Poly([(inner[0], inner[1]), (inner[0] + inner[2], inner[1]),
                     (inner[0] + inner[2], inner[1] + inner[3]), (inner[0], inner[1] + inner[3])],
                    fill=None, stroke=INK, width=1.6, opacity=0.55, close=True))
    inset = 5.0
    canvas.add(Poly([(inner[0] + inset, inner[1] + inset),
                     (inner[0] + inner[2] - inset, inner[1] + inset),
                     (inner[0] + inner[2] - inset, inner[1] + inner[3] - inset),
                     (inner[0] + inset, inner[1] + inner[3] - inset)],
                    fill=None, stroke=INK, width=0.6, opacity=0.4, close=True))

    _title_band(canvas, atlas, options, inner)
    project, scale = _transform(bounds, map_rect)

    for cell in ghost_cells:
        corners = [project(*pt) for pt in grid.corners_xy(cell)]
        # On an empty sheet the frontier is the only thing on the page, so it
        # is drawn a shade more firmly than later on.
        _draw_hex(canvas, corners, None, INK_FAINT, 0.7,
                  0.45 if not atlas.cells else 0.28, dash=(2.0, 3.5))

    # Footfall goes down first, the way pencil under ink works on a real
    # sheet. Traces overlap heavily on a route walked every day, and drawing
    # them on top of the chart buries it under a black smear; underneath, the
    # hex fills tint the accumulated ink and the chart still reads.
    if options.show_traces:
        _draw_traces(canvas, atlas, grid, project)

    for key in sorted(atlas.cells):
        record = atlas.cells[key]
        cell = Hex.parse(key)
        archetype = ARCHETYPE_BY_ID.get(record.archetype)
        corners = [project(*pt) for pt in grid.corners_xy(cell)]
        _draw_hex(canvas, corners, archetype.ink if archetype else INK, INK, 0.9, 0.22)
        _draw_hex(canvas, wobble(corners + [corners[0]], 0.7, f"hex-{key}"),
                  None, INK, 1.0, 0.75)

    _draw_labels(canvas, atlas, grid, project)

    ax, ay = project(*grid.center_xy(Hex(0, 0)))
    canvas.add(Dot(ax, ay, 3.4, fill=INK))
    canvas.add(Ellipse(ax, ay, 7.5, 7.5, fill=None, stroke=INK, width=1.0, opacity=0.8))
    canvas.add(Text(ax, ay + 20, "anchor", size=7.5, anchor="middle", fill=INK_FAINT))

    _compass(canvas, map_rect[0] + map_rect[2] - 34, map_rect[1] + 36, 22.0)
    _scale_bar(canvas, project, map_rect, scale)

    if not atlas.cells:
        canvas.add(Text(
            canvas.width / 2, map_rect[1] + map_rect[3] / 2 - 58,
            "Nothing charted yet. Walk somewhere.",
            size=15, anchor="middle", fill=INK_FAINT,
        ))

    bottom_y = inner[1] + inner[3] - BOTTOM_BAND + 8
    bottom_h = BOTTOM_BAND - 20
    _key_box(canvas, atlas, (inner[0] + 16, bottom_y, min(260.0, inner[2] * 0.4), bottom_h))
    _cartouche(canvas, atlas, (inner[0] + inner[2] * 0.46, bottom_y,
                               inner[2] * 0.54 - 16, bottom_h))
    return canvas


def _key_box(canvas: Canvas, atlas, rect) -> None:
    x, y, w, h = rect
    canvas.add(Poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fill=PARCHMENT_DEEP,
                    stroke=INK, width=0.9, opacity=0.55))
    canvas.add(Text(x + 10, y + 17, "KEY", size=9.5, bold=True, fill=INK))
    counts: dict[str, int] = {}
    for record in atlas.cells.values():
        counts[record.archetype] = counts.get(record.archetype, 0) + 1
    cursor = y + 32
    if not counts:
        canvas.add(Text(x + 10, cursor, "no hexes charted", size=8, fill=INK_FAINT))
        return
    for archetype_id in sorted(counts, key=lambda a: (-counts[a], a)):
        archetype = ARCHETYPE_BY_ID.get(archetype_id)
        if archetype is None:
            continue
        prims = marker_prims(x + 17, cursor - 3, marker_for(archetype_id), 3.4,
                             archetype.ink, fill=PARCHMENT)
        for prim in prims:
            canvas.add(prim)
        canvas.add(Text(x + 30, cursor, f"{archetype.label}  ({counts[archetype_id]})",
                        size=8, fill=INK))
        cursor += 14
        if cursor > y + h - 6:
            break


def _cartouche(canvas: Canvas, atlas, rect) -> None:
    x, y, w, h = rect
    canvas.add(Poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fill=PARCHMENT_DEEP,
                    stroke=INK, width=1.0, opacity=0.6))
    cursor = y + 19
    canvas.add(Text(x + 12, cursor, "FIELD NOTES", size=9.5, bold=True, fill=INK))
    cursor += 15

    walks = atlas.walks
    span = ""
    dates = [walk.get("started") for walk in walks if walk.get("started")]
    if dates:
        span = f"{min(dates)[:10]} to {max(dates)[:10]}"
    lines = [
        f"walks logged: {len(walks)}",
        f"distance walked: {atlas.total_walk_distance_m() / 1000.0:.1f} km",
        f"hexes charted: {atlas.revealed_count()}",
    ]
    if span:
        lines.append(f"period: {span}")
    lines.append(f"model: {atlas.model}")
    lines.append(f"classifier: {atlas.backend}")
    for line in lines:
        canvas.add(Text(x + 12, cursor, line, size=8, fill=INK))
        cursor += 12

    named = [record for record in atlas.cells.values() if record.name]
    if named:
        cursor += 4
        canvas.add(Text(x + 12, cursor, "named places:", size=8, bold=True, fill=INK))
        cursor += 12
        # The box has a fixed height on purpose; spill the names rather than
        # run them through the border of a sheet someone is going to print.
        room = int((y + h - 8 - cursor) // 10.5)
        if room < 2:
            canvas.add(Text(x + 12, cursor,
                            f"{len(named)} places named; see the gazetteer",
                            size=7.5, fill=INK_FAINT))
            return
        room = min(room - 1, 8)
        for record in sorted(named, key=lambda r: r.name)[:room]:
            canvas.add(Text(x + 12, cursor, f"· {record.name}", size=7.5, fill=INK))
            cursor += 10.5
        if len(named) > room:
            canvas.add(Text(x + 12, cursor, f"· and {len(named) - room} more", size=7.5,
                            fill=INK_FAINT))


def build_quest_card(atlas, quest: dict, options: MapOptions | None = None) -> Canvas:
    """A small printable card carrying one instruction: go here next."""
    options = options or MapOptions(page="a5", orientation="landscape", dpi=300.0)
    page_w, page_h = options.page_points(landscape=True)
    canvas = Canvas(page_w, page_h, background=PARCHMENT)
    margin = 10 * MM
    canvas.add(Poly([(margin, margin), (page_w - margin, margin), (page_w - margin, page_h - margin),
                     (margin, page_h - margin)], fill=None, stroke=INK, width=1.4, opacity=0.7,
                    close=True))

    # Start the block a fixed fraction down the card so the words sit in the
    # middle of the paper rather than clinging to the top edge.
    y = page_h * 0.24
    canvas.add(Text(page_w / 2, y, "A QUEST", size=11, anchor="middle",
                    bold=True, fill=INK_FAINT))
    y += 30
    canvas.add(Text(page_w / 2, y, quest.get("headline", "Walk somewhere new"),
                    size=19, anchor="middle", bold=True, fill=INK))

    body = wrap_paragraph(quest.get("instruction", ""), 10.0, page_w - 2 * margin - 40)
    y += 24
    for line in body:
        canvas.add(Text(page_w / 2, y, line, size=10, anchor="middle", fill=INK))
        y += 14

    _compass(canvas, page_w - margin - 38, margin + 40, 20.0)

    bearing = quest.get("bearing_deg")
    distance = quest.get("distance_m")
    if bearing is not None and distance is not None:
        y += 10
        canvas.add(Text(page_w / 2, y, f"{distance / 1000.0:.2f} km at {bearing:.0f}° "
                                       f"from the anchor", size=10.5, anchor="middle", fill=INK))
        y += 16
        label = quest.get("bearing_compass", "")
        if label:
            canvas.add(Text(page_w / 2, y, f"that is roughly {label}", size=9.5,
                            anchor="middle", fill=INK_FAINT))

    canvas.add(Text(page_w / 2, page_h - margin - 16,
                    "the map only grows where you actually stand",
                    size=8.5, anchor="middle", fill=INK_FAINT))
    return canvas


def gazetteer_markdown(atlas) -> str:
    """The written half of the map: every place the model named, in order."""
    lines = [f"# Gazetteer of {atlas.title}", ""]
    if not atlas.cells:
        lines.append("_Nothing charted yet._")
        return "\n".join(lines)
    named = [record for record in atlas.cells.values() if record.name]
    named.sort(key=lambda r: r.first_seen)
    for record in named:
        archetype = ARCHETYPE_BY_ID.get(record.archetype)
        label = archetype.label if archetype else record.archetype
        lines.append(f"## {record.name}")
        lines.append("")
        lines.append(f"*{label}* — hex `{record.cell}` — first charted {record.first_seen[:10]}")
        lines.append("")
        lines.append(record.legend)
        if record.inscription:
            lines.append("")
            lines.append(f"> {record.inscription}")
        lines.append("")
        lines.append(
            f"<sub>written by {record.author}, revised {record.revisions} time(s), "
            f"confidence {record.confidence:.2f}</sub>"
        )
        lines.append("")
    unnamed = [record for record in atlas.cells.values() if not record.name]
    if unnamed:
        lines.append(f"_{len(unnamed)} hex(es) charted but not yet named._")
    return "\n".join(lines)


def write_outputs(atlas, output_dir, options: MapOptions | None = None, stem: str = "map") -> dict:
    """Write SVG, PNG and PDF for the map, plus the gazetteer."""
    from pathlib import Path

    options = options or MapOptions()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas = build_map(atlas, options)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    svg_path.write_text(to_svg(canvas), encoding="utf-8")
    to_png(canvas, png_path, scale=options.dpi / 72.0)
    to_pdf(png_path, pdf_path, dpi=options.dpi)
    gazetteer = output_dir / "gazetteer.md"
    gazetteer.write_text(gazetteer_markdown(atlas), encoding="utf-8")
    return {"svg": svg_path, "png": png_path, "pdf": pdf_path, "gazetteer": gazetteer}


#: Where a label may sit relative to its hex, in preference order. Above first,
#: because a name reads as belonging to the thing just beneath it.
_LABEL_SLOTS = ((0.0, -13.0), (0.0, 19.0), (26.0, -2.0), (-26.0, -2.0),
                (24.0, -15.0), (-24.0, -15.0), (0.0, -27.0), (0.0, 32.0))


def _draw_traces(canvas: Canvas, atlas, grid, project) -> None:
    """Lay down every walk as a thin wobbling line.

    The atlas keeps the full trace, because the feature matrix is rebuilt from
    it on every reveal; only the drawing decimates.
    """
    for walk in atlas.walks:
        points = walk.get("points")
        if not points:
            continue
        if len(points) > 400:
            step = len(points) / 400.0
            points = [points[int(i * step)] for i in range(400)] + [points[-1]]
        projected = [project(*grid.projection.to_xy(lat, lon)) for lat, lon, *_ in points]
        if len(projected) < 2:
            continue
        path = wobble(projected, 0.9, f"trace-{walk.get('name', '')}", subdivisions=3)
        canvas.add(Poly(path, fill=None, stroke=INK, width=0.7, opacity=0.30, close=False))



def _draw_labels(canvas: Canvas, atlas, grid, project) -> None:
    """Place names so they do not collide.

    Cartographers label by hand and drop what will not fit; a machine should
    do the same rather than stack names on top of each other. Every hex still
    gets its symbol, so nothing is lost from the map -- only the words for
    places that had no room.
    """
    entries = []
    for key, record in atlas.cells.items():
        if not record.name and not record.inscription:
            continue
        cell = Hex.parse(key)
        cx, cy = project(*grid.center_xy(cell))
        priority = -int(record.stats.get("b_visits", 0) or 0)
        entries.append((priority, key, record, cell, cx, cy))
    entries.sort(key=lambda entry: (entry[0], entry[1]))

    label_size = 8.5 if len(entries) <= 16 else 7.0
    if len(entries) > 30:
        label_size = 6.2
    caption_size = label_size * 0.78
    placed: list[tuple[float, float, float, float]] = []

    for _priority, _key, record, cell, cx, cy in entries:
        archetype = ARCHETYPE_BY_ID.get(record.archetype)
        ink = archetype.ink if archetype else INK
        for prim in marker_prims(cx, cy, marker_for(record.archetype), 4.2, ink,
                                 fill=PARCHMENT):
            canvas.add(prim)

        if not record.name:
            continue
        caption = record.inscription[:44]
        width = max(canvas.text_width(record.name, label_size, bold=True),
                    canvas.text_width(caption, caption_size) if caption else 0.0)
        height = label_size + (caption_size + 3.0 if caption else 0.0)

        for offset_x, offset_y in _LABEL_SLOTS:
            box = (cx + offset_x - width / 2.0, cy + offset_y - height * 0.72,
                   width, height)
            if any(_overlaps(box, other) for other in placed):
                continue
            if (box[0] < 6 or box[1] < 6
                    or box[0] + box[2] > canvas.width - 6
                    or box[1] + box[3] > canvas.height - 6):
                continue
            placed.append(box)
            canvas.add(Text(cx + offset_x, cy + offset_y, record.name, size=label_size,
                            anchor="middle", bold=True, fill=INK))
            if caption:
                canvas.add(Text(cx + offset_x, cy + offset_y + caption_size + 2.0,
                                caption, size=caption_size, anchor="middle", fill=INK_FAINT))
            break


def _overlaps(a, b, padding: float = 2.5) -> bool:
    return not (
        a[0] + a[2] + padding < b[0]
        or b[0] + b[2] + padding < a[0]
        or a[1] + a[3] + padding < b[1]
        or b[1] + b[3] + padding < a[1]
    )
