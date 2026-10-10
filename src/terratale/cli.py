"""Command line interface: the short screen in this whole project.

The intended rhythm is: walk with a phone in your pocket, run one command when
you get home, print the sheet. The CLI is deliberately a handful of verbs so
that nothing about using TerraTale requires sitting at a computer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .archetypes import ARCHETYPE_BY_ID, ARCHETYPE_IDS, ARCHETYPE_BY_LABEL_PROXY  # noqa: F401
from .cartography import MapOptions, build_quest_card, gazetteer_markdown, write_outputs
from .grid import Hex
from .legend import build_cartographer
from .render import to_pdf, to_png, to_svg
from .state import DEFAULT_STATE_NAME, Atlas, store_walk, walks_from_atlas

DEFAULT_ATLAS = Path(DEFAULT_STATE_NAME)

try:  # pragma: no cover - cosmetic only
    from rich.console import Console

    _console = Console()
    _HEAD = lambda text: _console.print(f"\n[bold]{text}[/bold]")  # noqa: E731
    _SAY = lambda text: _console.print(text)  # noqa: E731
    _WARN = lambda text: _console.print(f"[yellow]{text}[/yellow]")  # noqa: E731
    _OK = lambda text: _console.print(f"[green]{text}[/green]")  # noqa: E731
except Exception:  # pragma: no cover
    _HEAD = lambda text: click.echo(f"\n{text}")  # noqa: E731
    _SAY = click.echo
    _WARN = lambda text: click.echo(f"warning: {text}", err=True)  # noqa: E731
    _OK = click.echo


def _atlas_path(ctx, param, value) -> Path:
    return Path(value)


pass_atlas = click.make_pass_decorator(Atlas, ensure=True)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--atlas", default=str(DEFAULT_ATLAS), callback=_atlas_path, show_default=True,
              help="Path to the atlas file.")
@click.version_option(package_name="terratale", prog_name="terratale")
@click.pass_context
def main(ctx: click.Context, atlas: Path) -> None:
    """TerraTale — a treasure map that only grows where you have walked."""
    ctx.obj = Atlas.load_or_create(atlas)
    ctx.meta["atlas_path"] = atlas


@main.command()
@click.option("--lat", type=float, required=True, help="Anchor latitude (usually home).")
@click.option("--lon", type=float, required=True, help="Anchor longitude (usually home).")
@click.option("--name", default=None, help="Title for the finished map.")
@click.option("--hex-size", type=float, default=120.0, show_default=True,
              help="Hex radius in metres.")
@click.option("--force", is_flag=True, help="Overwrite an existing atlas.")
@click.pass_obj
def init(atlas: Atlas, lat: float, lon: float, name: str | None, hex_size: float,
         force: bool) -> None:
    """Start a new map around an anchor point."""
    if atlas.cells and not force:
        raise click.ClickException("this atlas already has charted hexes; pass --force to reset")
    atlas.anchor_lat = lat
    atlas.anchor_lon = lon
    atlas.hex_size_m = hex_size
    if name:
        atlas.title = name
    atlas.cells.clear()
    _OK(f"anchor set to {lat:.5f}, {lon:.5f} with {hex_size:.0f} m hexes")
    _SAY("next: terratale add <your.gpx>")


@main.command()
@click.argument("traces", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def add(ctx: click.Context, traces) -> None:
    """Import GPS traces (GPX exports) into the atlas."""
    from .geo import load_walk

    atlas: Atlas = ctx.obj
    added = 0
    for trace in traces:
        walk = load_walk(trace)
        if atlas.has_walk(walk.name):
            walk.name = f"{walk.name}-{len(atlas.walks)}"
        record = store_walk(atlas, walk)
        added += 1
        _SAY(f"  {walk.name}: {record['stored_points']} points, "
             f"{record['distance_m'] / 1000.0:.2f} km")
    atlas.save(ctx.meta["atlas_path"])
    _OK(f"imported {added} trace(s); atlas saved")
    _SAY("next: terratale reveal")


@main.command()
@click.option("--model", default=None, help="Gemma model tag to use via Ollama.")
@click.option("--backend", type=click.Choice(["auto", "tabpfn", "knn"]), default=None,
              help="Classifier backend.")
@click.option("--no-revise", is_flag=True, help="Only name new hexes; never rewrite a legend.")
@click.option("--offline", is_flag=True, help="Use the template writer instead of Gemma.")
@click.option("--limit", type=int, default=0, help="Cap how many hexes to write up (0 = all).")
@click.option("--rewrite", is_flag=True,
              help="Rewrite every legend, e.g. after switching model or backend.")
@click.pass_context
def reveal(ctx: click.Context, model: str | None, backend: str | None, no_revise: bool,
           offline: bool, limit: int, rewrite: bool) -> None:
    """Classify every visited hex and write legends for it."""
    from .archetypes import classify_cells
    from .features import extract_cells
    from .state import CellRecord, _now

    atlas: Atlas = ctx.obj
    if not atlas.walks:
        raise click.ClickException("no walks in the atlas yet; run `terratale add` first")

    walks = walks_from_atlas(atlas)
    grid = atlas.grid
    stats = extract_cells(walks, grid)
    _SAY(f"{len(walks)} walks cover {len(stats)} hexes of "
         f"{grid.size_m:.0f} m radius")

    if backend:
        atlas.backend = backend
    predictions, used_backend, _model = classify_cells(
        stats, atlas.labelled, atlas.backend or "auto"
    )
    atlas.backend = used_backend
    _SAY(f"classifier backend: {used_backend}"
         + ("  (install tabpfn for the real model)" if used_backend == "knn" else ""))

    writer_mode = "offline" if offline else "auto"
    cartographer, detail = build_cartographer(mode=writer_mode, model=model or atlas.model)
    if model:
        atlas.model = model
    _SAY(f"legend writer: {detail}")
    if detail.startswith("offline"):
        _WARN("Gemma is not reachable, so legends come from the offline template writer. "
              "They are honest but they are not model output.")

    fresh = 0
    revised = 0
    written = 0
    for key, cell_stats in sorted(stats.items()):
        if limit and written >= limit:
            break
        cell_key = key.key
        archetype_id, confidence = predictions.get(key, ("hidden_hollow", 0.0))
        record = atlas.cells.get(cell_key)
        walk_names = [w.name for w in walks]
        is_new = record is None

        if record is None:
            record = CellRecord(
                cell=cell_key,
                archetype=archetype_id,
                confidence=confidence,
                first_seen=_now(),
                stats=cell_stats.to_dict(),
            )
            atlas.cells[cell_key] = record

        gained = [name for name in set(walk_names) if name not in record.walks]
        if not is_new and not rewrite and (not gained or no_revise):
            record.last_seen = _now()
            record.stats = cell_stats.to_dict()
            record.archetype = archetype_id
            record.confidence = confidence
            continue

        taken = [name for name in atlas.taken_names() if name != record.name]
        try:
            legend = cartographer.name_cell(cell_stats, ARCHETYPE_BY_ID[archetype_id],
                                            taken, walk_names)
        except Exception as exc:  # noqa: BLE001 - one bad hex must not kill the run
            _WARN(f"{cell_key}: legend rejected ({exc}); leaving unnamed")
            continue

        record.archetype = archetype_id
        record.confidence = confidence
        record.name = legend.name
        record.legend = legend.text
        record.inscription = legend.inscription
        record.author = legend.author
        record.stats = cell_stats.to_dict()
        record.last_seen = _now()
        record.walks = sorted(set(record.walks) | set(walk_names))
        if is_new:
            fresh += 1
        else:
            record.revisions += 1
            revised += 1
        written += 1

    atlas.save(ctx.meta["atlas_path"])
    _OK(f"named {fresh} new hex(es), revised {revised}; {len(atlas.cells)} charted in total")
    _SAY("next: terratale map")


@main.command()
@click.option("--out", default="map", show_default=True, type=click.Path(file_okay=False),
              help="Output directory.")
@click.option("--page", default="a3", show_default=True,
              type=click.Choice(["a4", "a3", "a5", "letter", "tabloid"]))
@click.option("--orientation", default="auto", show_default=True,
              type=click.Choice(["auto", "landscape", "portrait"]),
              help="Sheet orientation; auto fits the ground actually charted.")
@click.option("--dpi", default=300.0, show_default=True, help="Raster resolution for PNG/PDF.")
@click.option("--title", default=None, help="Override the map title.")
@click.option("--stem", default="map", show_default=True, help="Output filename stem.")
@click.pass_context
def map(ctx: click.Context, out: str, page: str, orientation: str, dpi: float,
        title: str | None, stem: str) -> None:
    """Render the printable map sheet (SVG, PNG and PDF) plus the gazetteer."""
    atlas: Atlas = ctx.obj
    options = MapOptions(page=page, orientation=orientation, dpi=dpi, title=title)
    paths = write_outputs(atlas, out, options, stem=stem)
    _OK("map written:")
    for kind, path in paths.items():
        _SAY(f"  {kind:9s} {path}")
    _SAY("print the PDF at 100% scale; it is sized to the real page.")


@main.command()
@click.option("--count", default=1, show_default=True, help="How many destinations to suggest.")
@click.option("--card", is_flag=True, help="Also render a printable quest card.")
@click.option("--out", default="map", show_default=True, type=click.Path(file_okay=False))
@click.option("--dpi", default=200.0, show_default=True, help="Raster resolution for the card.")
@click.pass_context
def quest(ctx: click.Context, count: int, card: bool, out: str, dpi: float) -> None:
    """Ask where to walk next."""
    from .quest import suggest_quests, summarise_progress

    atlas: Atlas = ctx.obj
    _SAY(summarise_progress(atlas))
    suggestions = suggest_quests(atlas, count=count)
    if not suggestions:
        _WARN("no reachable uncharted hexes; extend the atlas or walk further out")
        return
    for index, suggestion in enumerate(suggestions, start=1):
        _HEAD(f"Quest {index}: {suggestion['headline']}")
        _SAY(f"  hex {suggestion['cell']} at {suggestion['lat']:.5f}, {suggestion['lon']:.5f}")
        _SAY("  " + suggestion["instruction"])
    if card:
        options = MapOptions(page="a5", orientation="landscape", dpi=dpi)
        canvas = build_quest_card(atlas, suggestions[0], options)
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)
        svg_path = out_dir / "quest.svg"
        png_path = out_dir / "quest.png"
        svg_path.write_text(to_svg(canvas), encoding="utf-8")
        to_png(canvas, png_path, scale=options.dpi / 72.0)
        to_pdf(png_path, out_dir / "quest.pdf", dpi=options.dpi)
        _OK(f"quest card written to {svg_path}, {png_path} and {out_dir / 'quest.pdf'}")


@main.command()
@click.argument("cell")
@click.argument("archetype", type=click.Choice(sorted(ARCHETYPE_IDS)))
@click.pass_context
def label(ctx: click.Context, cell: str, archetype: str) -> None:
    """Teach the classifier: declare what a hex really is.

    Labels join the TabPFN support set alongside the built-in primer, so a
    couple of corrections re-shape every other prediction on the map.
    """
    atlas: Atlas = ctx.obj
    try:
        Hex.parse(cell)
    except Exception as exc:
        raise click.ClickException(f"{cell!r} is not a hex coordinate like '3:-1'") from exc
    atlas.labelled[cell] = archetype
    if cell in atlas.cells:
        atlas.cells[cell].archetype = archetype
    atlas.save(ctx.meta["atlas_path"])
    _OK(f"{cell} labelled {archetype}")
    _SAY("run `terratale reveal` to re-classify with your correction in the support set")


@main.command()
@click.option("--out", default=None, help="Write the gazetteer to this file.")
@click.pass_context
def gazetteer(ctx: click.Context, out: str | None) -> None:
    """Print the written legends for every named place."""
    atlas: Atlas = ctx.obj
    text = gazetteer_markdown(atlas)
    if out:
        Path(out).write_text(text, encoding="utf-8")
        _OK(f"gazetteer written to {out}")
    else:
        click.echo(text)


@main.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Check what this machine can actually run."""
    import platform

    atlas: Atlas = ctx.obj
    _HEAD("environment")
    _SAY(f"  python           {platform.python_version()}")
    for module, purpose in (("numpy", "feature matrix"), ("PIL", "raster and PDF output"),
                            ("tabpfn", "place classification"), ("click", "the CLI itself")):
        try:
            __import__(module)
            _SAY(f"  {module:16s} ok       ({purpose})")
        except ImportError:
            _WARN(f"  {module:16s} missing  ({purpose})")
    from .legend import OllamaClient, OllamaUnavailable

    _HEAD("gemma")
    client = OllamaClient()
    try:
        models = client.models()
        _OK(f"  Ollama reachable, {len(models)} model(s): {', '.join(models) or 'none'}")
        if atlas.model not in models:
            _WARN(f"  atlas is set to {atlas.model!r}, which is not pulled")
    except OllamaUnavailable as exc:
        _WARN(f"  {exc}")
        _SAY("  start it with `ollama serve` and pull a model, e.g. `ollama pull gemma4:e2b`")
    _HEAD("atlas")
    _SAY(f"  {len(atlas.walks)} walks, {atlas.revealed_count()} hexes charted, "
         f"model {atlas.model}")


@main.command()
@click.option("--days", default=12, show_default=True, help="Days of synthetic walking to invent.")
@click.option("--seed", default=7, show_default=True)
@click.option("--lat", default=-17.8255, show_default=True)
@click.option("--lon", default=31.0522, show_default=True)
@click.option("--out", default="map", show_default=True, type=click.Path(file_okay=False))
@click.option("--offline", is_flag=True, help="Skip Gemma and use the template writer.")
@click.option("--dpi", default=300.0, show_default=True, help="Raster resolution for PNG/PDF.")
@click.pass_context
def demo(ctx: click.Context, days: int, seed: int, lat: float, lon: float, out: str,
         offline: bool, dpi: float) -> None:
    """Run the whole pipeline on generated walks — no personal data needed."""
    from .synth import generate_neighbourhood, walk_distance_m

    atlas: Atlas = ctx.obj
    atlas.anchor_lat, atlas.anchor_lon = lat, lon
    atlas.title = "The Unwalked Neighbourhood"
    atlas.cells.clear()
    atlas.walks.clear()

    walks = generate_neighbourhood(anchor_lat=lat, anchor_lon=lon, days=days, seed=seed)
    for walk in walks:
        store_walk(atlas, walk)
    _OK(f"generated {len(walks)} synthetic walks, "
        f"{sum(walk_distance_m(w) for w in walks) / 1000.0:.1f} km")
    _WARN("these traces are generated for demonstration; no real location data is involved")

    ctx.invoke(reveal, model=None, backend=None, no_revise=False, offline=offline,
               limit=0, rewrite=False)
    ctx.invoke(map, out=out, page="a3", orientation="auto", dpi=dpi, title=None,
               stem="map")
    ctx.invoke(quest, count=2, card=True, out=out, dpi=min(dpi, 200.0))
    _OK("demo complete — open map/map.png or print map/map.pdf")


if __name__ == "__main__":  # pragma: no cover
    main()
