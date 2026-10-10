"""Map rendering: an intermediate draw list with SVG, PNG and PDF backends.

The map is the artefact people actually keep, so it has to come out printable
rather than merely viewable. Builders push primitives into a :class:`Canvas`
in pixel space and each backend serialises them; the SVG and the raster output
therefore cannot drift apart.

The hand-drawn look is deliberate and deterministic: every wobble comes from a
seeded generator keyed on the thing being drawn, so the same atlas always
produces the same map. A map that re-scribbles itself on each run stops
feeling like an object.
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from .archetypes import ARCHETYPE_BY_ID, Archetype

PARCHMENT = "#f4e8d0"
PARCHMENT_DEEP = "#e8d7b6"
INK = "#3b2f23"
INK_FAINT = "#8d7a5f"

FONT_STACK = "Noto Serif, Liberation Serif, Georgia, serif"
_SERIF_CANDIDATES = (
    "/usr/share/fonts/google-noto-vf/NotoSerif[wght].ttf",
    "/usr/share/fonts/liberation-serif-fonts/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/msttcore/times.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
)
_BOLD_CANDIDATES = (
    "/usr/share/fonts/google-noto-vf/NotoSerif[wght].ttf",
    "/usr/share/fonts/liberation-serif-fonts/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/msttcore/timesbd.ttf",
)

#: Drawn markers, one per archetype. Chosen so a printed map key is readable
#: without relying on glyph coverage in whatever font the printer has.
MARKERS = ("disc", "bar", "cross", "leaf", "ring", "triangle", "chevron", "diamond")


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


@dataclass
class Poly:
    points: list[tuple[float, float]]
    fill: str | None = None
    stroke: str | None = None
    width: float = 1.0
    opacity: float = 1.0
    dash: tuple[float, float] | None = None
    close: bool = True


@dataclass
class Text:
    x: float
    y: float
    text: str
    size: float = 12.0
    fill: str = INK
    anchor: str = "start"
    bold: bool = False
    opacity: float = 1.0
    rotation: float = 0.0


@dataclass
class Dot:
    x: float
    y: float
    r: float
    fill: str = INK
    stroke: str | None = None
    width: float = 1.0
    opacity: float = 1.0


@dataclass
class Ellipse:
    x: float
    y: float
    rx: float
    ry: float
    fill: str | None = None
    stroke: str | None = None
    width: float = 1.0
    opacity: float = 1.0
    rotation: float = 0.0


@dataclass
class Canvas:
    width: float
    height: float
    background: str = PARCHMENT
    prims: list = field(default_factory=list)

    def add(self, prim):
        self.prims.append(prim)
        return prim

    def text_width(self, text: str, size: float, bold: bool = False) -> float:
        return measure_text(text, size, bold)


# --------------------------------------------------------------------------
# font metrics -- PIL is used for measurement only, so SVG and PNG agree
# --------------------------------------------------------------------------

_FONT_CACHE: dict[tuple[str, int, bool], object] = {}
_HAS_PIL = True
try:  # pragma: no cover - exercised implicitly
    from PIL import ImageFont
except Exception:  # pragma: no cover
    ImageFont = None
    _HAS_PIL = False


def _font_path(bold: bool) -> str | None:
    for candidate in (_BOLD_CANDIDATES if bold else _SERIF_CANDIDATES):
        if Path(candidate).exists():
            return candidate
    return None


def load_font(size: int, bold: bool = False):
    if not _HAS_PIL:
        return None
    key = ("font", size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = _font_path(bold)
    font = None
    if path:
        try:
            font = ImageFont.truetype(path, size)
        except Exception:
            font = None
    _FONT_CACHE[key] = font
    return font


def measure_text(text: str, size: float, bold: bool = False) -> float:
    """Width in pixels, measured when possible and estimated when not."""
    if not text:
        return 0.0
    font = load_font(max(1, int(round(size))), bold)
    if font is not None:
        try:
            return float(font.getlength(text))
        except Exception:
            pass
    return len(text) * size * 0.5


def wrap_text(text: str, size: float, max_width: float, bold: bool = False) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if measure_text(candidate, size, bold) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def wrap_paragraph(text: str, size: float, max_width: float, bold: bool = False) -> list[str]:
    """Wrap honouring explicit newlines in the source."""
    out: list[str] = []
    for block in text.split("\n"):
        out.extend(wrap_text(block, size, max_width, bold))
    return out


# --------------------------------------------------------------------------
# hand-drawn helpers
# --------------------------------------------------------------------------


def wobble(points, amplitude: float, seed: str, subdivisions: int = 6):
    """Perturb a polyline deterministically so it reads as pen on paper."""
    if len(points) < 2:
        return list(points)
    rng = random.Random(seed)
    out = [points[0]]
    for index in range(len(points) - 1):
        x0, y0 = points[index]
        x1, y1 = points[index + 1]
        span = math.dist((x0, y0), (x1, y1))
        if span < 1e-6:
            continue
        for step in range(1, subdivisions + 1):
            frac = step / subdivisions
            x = x0 + (x1 - x0) * frac
            y = y0 + (y1 - y0) * frac
            if step != subdivisions:
                fade = math.sin(math.pi * frac)
                x += rng.uniform(-amplitude, amplitude) * fade
                y += rng.uniform(-amplitude, amplitude) * fade
            out.append((x, y))
    return out


def marker_prims(cx: float, cy: float, kind: str, size: float, ink: str, fill: str | None = None):
    """Return primitives for one archetype marker."""
    prims = []
    if kind == "disc":
        prims.append(Dot(cx, cy, size, fill=ink))
    elif kind == "ring":
        prims.append(Ellipse(cx, cy, size, size, fill=fill, stroke=ink, width=1.4))
    elif kind == "bar":
        prims.append(Poly(
            [(cx - size, cy - size * 0.34), (cx + size, cy - size * 0.34),
             (cx + size, cy + size * 0.34), (cx - size, cy + size * 0.34)],
            fill=fill, stroke=ink, width=1.3))
    elif kind == "cross":
        prims.append(Poly([(cx - size, cy - size), (cx + size, cy + size)], stroke=ink,
                          width=1.6, close=False))
        prims.append(Poly([(cx - size, cy + size), (cx + size, cy - size)], stroke=ink,
                          width=1.6, close=False))
    elif kind == "triangle":
        prims.append(Poly([(cx, cy - size * 1.15), (cx + size, cy + size * 0.8),
                           (cx - size, cy + size * 0.8)], fill=fill, stroke=ink, width=1.3))
    elif kind == "diamond":
        prims.append(Poly([(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)],
                          fill=fill, stroke=ink, width=1.3))
    elif kind == "leaf":
        prims.append(Ellipse(cx, cy, size, size * 0.55, fill=fill, stroke=ink,
                             width=1.3, rotation=-28.0))
    elif kind == "chevron":
        prims.append(Poly([(cx - size, cy + size * 0.7), (cx, cy - size * 0.7),
                           (cx + size, cy + size * 0.7)], stroke=ink, width=1.6, close=False))
    return prims


def marker_for(archetype_id: str) -> str:
    order = list(ARCHETYPE_BY_ID)
    try:
        return MARKERS[order.index(archetype_id) % len(MARKERS)]
    except ValueError:
        return "disc"


# --------------------------------------------------------------------------
# serialisers
# --------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def to_svg(canvas: Canvas, texture: bool = True) -> str:
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt(canvas.width)}" '
        f'height="{_fmt(canvas.height)}" viewBox="0 0 {_fmt(canvas.width)} {_fmt(canvas.height)}">'
    )
    if texture:
        parts.append(
            "<defs>"
            '<filter id="paper" x="0" y="0" width="100%" height="100%">'
            '<feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="4" '
            'seed="11" result="noise"/>'
            '<feColorMatrix in="noise" type="saturate" values="0"/>'
            '<feComponentTransfer><feFuncA type="linear" slope="0.055"/></feComponentTransfer>'
            "</filter>"
            "</defs>"
        )
    parts.append(
        f'<rect width="100%" height="100%" fill="{canvas.background}"/>'
    )
    if texture:
        parts.append(
            f'<rect width="100%" height="100%" filter="url(#paper)" opacity="0.85"/>'
        )
    for prim in canvas.prims:
        if isinstance(prim, Poly):
            points = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in prim.points)
            shape = "polygon" if prim.close else "polyline"
            attrs = [f'points="{points}"']
            attrs.append(f'fill="{prim.fill}"' if prim.fill else 'fill="none"')
            if prim.stroke:
                attrs.append(f'stroke="{prim.stroke}"')
                attrs.append(f'stroke-width="{_fmt(prim.width)}"')
                attrs.append('stroke-linejoin="round" stroke-linecap="round"')
            if prim.dash:
                attrs.append(f'stroke-dasharray="{_fmt(prim.dash[0])} {_fmt(prim.dash[1])}"')
            if prim.opacity < 1.0:
                attrs.append(f'opacity="{_fmt(prim.opacity)}"')
            parts.append(f"<{shape} {' '.join(attrs)}/>")
        elif isinstance(prim, Text):
            attrs = [
                f'x="{_fmt(prim.x)}"', f'y="{_fmt(prim.y)}"',
                f'font-size="{_fmt(prim.size)}"', f'fill="{prim.fill}"',
                f'font-family="{FONT_STACK}"',
            ]
            if prim.bold:
                attrs.append('font-weight="bold"')
            if prim.anchor != "start":
                attrs.append(f'text-anchor="{prim.anchor}"')
            if prim.opacity < 1.0:
                attrs.append(f'opacity="{_fmt(prim.opacity)}"')
            if prim.rotation:
                attrs.append(f'transform="rotate({_fmt(prim.rotation)} {_fmt(prim.x)} {_fmt(prim.y)})"')
            escaped = (
                prim.text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            parts.append(f"<text {' '.join(attrs)}>{escaped}</text>")
        elif isinstance(prim, Dot):
            attrs = [f'cx="{_fmt(prim.x)}"', f'cy="{_fmt(prim.y)}"', f'r="{_fmt(prim.r)}"',
                     f'fill="{prim.fill}"']
            if prim.stroke:
                attrs.append(f'stroke="{prim.stroke}"')
                attrs.append(f'stroke-width="{_fmt(prim.width)}"')
            if prim.opacity < 1.0:
                attrs.append(f'opacity="{_fmt(prim.opacity)}"')
            parts.append(f"<circle {' '.join(attrs)}/>")
        elif isinstance(prim, Ellipse):
            transform = ""
            if prim.rotation:
                transform = f' transform="rotate({_fmt(prim.rotation)} {_fmt(prim.x)} {_fmt(prim.y)})"'
            attrs = [f'cx="{_fmt(prim.x)}"', f'cy="{_fmt(prim.y)}"',
                     f'rx="{_fmt(prim.rx)}"', f'ry="{_fmt(prim.ry)}"']
            attrs.append(f'fill="{prim.fill}"' if prim.fill else 'fill="none"')
            if prim.stroke:
                attrs.append(f'stroke="{prim.stroke}"')
                attrs.append(f'stroke-width="{_fmt(prim.width)}"')
            if prim.opacity < 1.0:
                attrs.append(f'opacity="{_fmt(prim.opacity)}"')
            parts.append(f"<ellipse {' '.join(attrs)}{transform}/>")
    parts.append("</svg>")
    return "\n".join(parts)


def _hex_rgb(color: str) -> tuple[int, int, int]:
    """Parse #rgb, #rrggbb or #rrggbbaa into an (r, g, b) triple."""
    text = color.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) == 8:
        text = text[:6]
    if len(text) != 6:
        raise ValueError(f"unrecognised colour {color!r}")
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))


def to_png(canvas: Canvas, path: str | Path, scale: float = 2.0, texture: bool = True) -> Path:
    """Rasterise with Pillow, supersampled so print output stays crisp."""
    from PIL import Image, ImageDraw

    path = Path(path)
    width = int(round(canvas.width * scale))
    height = int(round(canvas.height * scale))
    image = Image.new("RGB", (width, height), _hex_rgb(canvas.background))
    draw = ImageDraw.Draw(image, "RGBA")

    if texture:
        noise = Image.effect_noise((width, height), 22).convert("L")
        overlay = Image.new("RGB", (width, height), (120, 96, 60))
        image = Image.composite(overlay, image, noise.point(lambda v: int(v * 0.10)))
        draw = ImageDraw.Draw(image, "RGBA")

    for prim in canvas.prims:
        if isinstance(prim, Poly):
            points = [(x * scale, y * scale) for x, y in prim.points]
            alpha = int(255 * prim.opacity)
            fill = (*_hex_rgb(prim.fill), alpha) if prim.fill else None
            outline = (*_hex_rgb(prim.stroke), alpha) if prim.stroke else None
            if prim.close:
                draw.polygon(points, fill=fill)
            if prim.stroke:
                line = points + [points[0]] if prim.close else points
                draw.line(line, fill=outline, width=max(1, int(prim.width * scale)), joint="curve")
        elif isinstance(prim, Text):
            font = load_font(max(1, int(round(prim.size * scale))), prim.bold)
            if font is None:
                continue
            anchor = {"start": "ls", "middle": "ms", "end": "rs"}[prim.anchor]
            fill = (*_hex_rgb(prim.fill), int(255 * prim.opacity))
            if prim.rotation:
                pad = int(prim.size * scale * 3)
                box = draw.textbbox((0, 0), prim.text, font=font)
                tile_w = max(1, box[2] - box[0] + pad * 2)
                tile_h = max(1, box[3] - box[1] + pad * 2)
                tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
                ImageDraw.Draw(tile).text((pad, pad), prim.text, font=font, fill=fill)
                tile = tile.rotate(prim.rotation, expand=True, resample=Image.BICUBIC)
                image.paste(tile, (int(prim.x * scale) - tile.width // 2,
                                   int(prim.y * scale) - tile.height // 2), tile)
            else:
                draw.text((prim.x * scale, prim.y * scale), prim.text, font=font,
                          fill=fill, anchor=anchor)
        elif isinstance(prim, Dot):
            r = prim.r * scale
            box = [prim.x * scale - r, prim.y * scale - r, prim.x * scale + r, prim.y * scale + r]
            draw.ellipse(box, fill=(*_hex_rgb(prim.fill), int(255 * prim.opacity)),
                         outline=(*_hex_rgb(prim.stroke), int(255 * prim.opacity)) if prim.stroke else None,
                         width=max(1, int(prim.width * scale)))
        elif isinstance(prim, Ellipse):
            r = (prim.rx * scale, prim.ry * scale)
            box = [prim.x * scale - r[0], prim.y * scale - r[1],
                   prim.x * scale + r[0], prim.y * scale + r[1]]
            fill = (*_hex_rgb(prim.fill), int(255 * prim.opacity)) if prim.fill else None
            outline = (*_hex_rgb(prim.stroke), int(255 * prim.opacity)) if prim.stroke else None
            if prim.rotation:
                pad = int(max(r) * 2 + 4)
                tile = Image.new("RGBA", (pad, pad), (0, 0, 0, 0))
                ImageDraw.Draw(tile).ellipse([pad / 2 - r[0], pad / 2 - r[1],
                                              pad / 2 + r[0], pad / 2 + r[1]],
                                             fill=fill, outline=outline,
                                             width=max(1, int(prim.width * scale)))
                tile = tile.rotate(prim.rotation, expand=True, resample=Image.BICUBIC)
                image.paste(tile, (int(prim.x * scale) - tile.width // 2,
                                   int(prim.y * scale) - tile.height // 2), tile)
            else:
                draw.ellipse(box, fill=fill, outline=outline,
                             width=max(1, int(prim.width * scale)))

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=92)
    else:
        image.save(path)
    return path


def to_pdf(png_path: str | Path, pdf_path: str | Path, dpi: float = 300.0) -> Path:
    """Wrap a PNG into a single-page PDF at a real physical size.

    Written by hand so printing a map needs no extra dependency; the page is
    sized from the image at `dpi`, which means an A3 render prints at A3.
    """
    from PIL import Image

    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    with Image.open(png_path) as probe:
        pixel_w, pixel_h = probe.size
        rgb = probe.convert("RGB")
        raw = rgb.tobytes()

    page_w = pixel_w / dpi * 72.0
    page_h = pixel_h / dpi * 72.0
    stream = zlib.compress(raw, 9)

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.2f} {page_h:.2f}] "
        f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>".encode()
    )
    objects.append(
        f"<< /Type /XObject /Subtype /Image /Width {pixel_w} /Height {pixel_h} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
        f"/Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
    )
    # The drawing operator stream must declare its own exact byte length; a
    # wrong /Length is the classic hand-rolled-PDF corruption.
    content = f"q {page_w:.2f} 0 0 {page_h:.2f} 0 0 cm /Im0 Do Q\n".encode()
    objects.append(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
                   + content + b"endstream")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(bytes(out))
    return pdf_path
