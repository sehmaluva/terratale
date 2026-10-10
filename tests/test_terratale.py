"""Tests for TerraTale.

Written with the standard library's unittest so that the suite runs on a
machine with no package index reachable, which is the same constraint the
project is built for. Most of these guard specific things that were measurably
wrong at some point: distance inflated by GPS noise, one visit counted as
eighty, a feature that grew with time spent, a support set with holes in it.
"""

from __future__ import annotations

import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

from terratale.archetypes import (
    ARCHETYPE_IDS, NearestNeighbourClassifier, classify_cells, primer_matrix,
)
from terratale.cartography import (
    MapOptions, build_map, build_quest_card, gazetteer_markdown, write_outputs,
)
from terratale.features import (
    FEATURES, GEOMETRY_SPACING_M, Evidence, CellStats, _segments, charted_cells,
    clean_path, climb_per_100m, extract_cells, smooth_path,
)
from terratale.geo import Fix, Projection, Walk, haversine_m, load_gpx
from terratale.grid import Hex, HexGrid, hex_distance
from terratale.legend import Legend, OfflineCartographer, _parse_json_object, _validate, _Validation
from terratale.quest import compass_label, suggest_quests
from terratale.render import Canvas, Text, _hex_rgb, to_pdf, to_png, to_svg
from terratale.state import Atlas, store_walk, walks_from_atlas
from terratale.synth import generate_neighbourhood

LAT0, LON0 = -17.8255, 31.0522


def make_walk(fixes, name="w") -> Walk:
    return Walk(name=name, fixes=fixes, source="test")


def straight_line(n=200, spacing_m=1.4, jitter_m=0.0, speed=1.4, seed=3, t0=None):
    """A due-east walk with optional GPS noise, sampled at 1 Hz."""
    import random

    rng = random.Random(seed)
    projection = Projection(LAT0, LON0)
    t0 = t0 or datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
    fixes = []
    for i in range(n):
        x = i * spacing_m
        lat, lon = projection.to_latlon(x + rng.gauss(0, jitter_m), rng.gauss(0, jitter_m))
        fixes.append(Fix(lat=lat, lon=lon, t=t0 + timedelta(seconds=i * spacing_m / speed)))
    return fixes


class TestGeo(unittest.TestCase):
    def test_haversine_against_known_offset(self):
        # one hundredth of a degree of latitude is about 1.11 km
        self.assertAlmostEqual(haversine_m(LAT0, LON0, LAT0 + 0.01, LON0), 1112, delta=15)

    def test_haversine_is_symmetric_and_zero_at_identity(self):
        self.assertEqual(haversine_m(LAT0, LON0, LAT0, LON0), 0.0)
        a = haversine_m(LAT0, LON0, LAT0 + 0.004, LON0 + 0.004)
        b = haversine_m(LAT0 + 0.004, LON0 + 0.004, LAT0, LON0)
        self.assertAlmostEqual(a, b, places=9)

    def test_projection_roundtrip(self):
        projection = Projection(LAT0, LON0)
        for x, y in ((0, 0), (500, -700), (-1200, 60)):
            lat, lon = projection.to_latlon(float(x), float(y))
            back_x, back_y = projection.to_xy(lat, lon)
            self.assertAlmostEqual(back_x, x, places=6)
            self.assertAlmostEqual(back_y, y, places=6)

    def test_loads_gpx_with_namespaces_and_missing_time(self):
        gpx = """<?xml version="1.0"?>
<gpx xmlns="http://www.topografix.com/GPX/1/1" version="1.1">
  <trk><name>evening loop</name><trkseg>
    <trkpt lat="-17.8255" lon="31.0522"><ele>1481.5</ele>
      <time>2026-01-05T17:04:00Z</time></trkpt>
    <trkpt lat="-17.8256" lon="31.0523"><ele>1483.0</ele></trkpt>
    <trkpt lat="-17.8257" lon="31.0524"><ele>1482.0</ele>
      <time>2026-01-05T17:06:00Z</time></trkpt>
  </trkseg></trk>
</gpx>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "walk.gpx"
            path.write_text(gpx)
            walk = load_gpx(path)
        self.assertEqual(len(walk), 3)
        self.assertEqual(walk.timed, 2)
        self.assertAlmostEqual(walk.fixes[0].ele, 1481.5)
        self.assertIsNone(walk.fixes[1].t)
        self.assertEqual(walk.duration_s(), 120.0)

    def test_gpx_without_points_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.gpx"
            path.write_text('<?xml version="1.0"?><gpx version="1.1"></gpx>')
            with self.assertRaises(ValueError):
                load_gpx(path)


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.grid = HexGrid(Projection(LAT0, LON0), 120.0)

    def test_hex_assignment_is_stable_across_the_lattice(self):
        import random

        rng = random.Random(11)
        for _ in range(500):
            lat = LAT0 + rng.uniform(-0.02, 0.02)
            lon = LON0 + rng.uniform(-0.02, 0.02)
            cell = self.grid.to_hex(lat, lon)
            centre_lat, centre_lon = self.grid.center(cell)
            self.assertEqual(self.grid.to_hex(centre_lat, centre_lon), cell)

    def test_neighbours_are_adjacent_and_ring_size_is_correct(self):
        origin = Hex(0, 0)
        self.assertEqual({hex_distance(origin, n) for n in origin.neighbours()}, {1})
        self.assertEqual(len(origin.ring(2)), 12)
        self.assertEqual(len(origin.ring(0)), 1)

    def test_hex_key_roundtrip(self):
        for cell in (Hex(0, 0), Hex(-4, 7), Hex(12, -3)):
            self.assertEqual(Hex.parse(cell.key), cell)


class TestFeatureExtraction(unittest.TestCase):
    def test_smoothing_recovers_distance_that_raw_noise_inflates(self):
        """The regression that mattered most: raw GPS distance is fiction."""
        fixes = straight_line(n=600, spacing_m=1.4, jitter_m=3.2)
        truth = 599 * 1.4

        raw = sum(haversine_m(a.lat, a.lon, b.lat, b.lon) for a, b in zip(fixes, fixes[1:]))
        cleaned = clean_path(smooth_path(fixes), GEOMETRY_SPACING_M)
        measured = sum(haversine_m(a.lat, a.lon, b.lat, b.lon)
                       for a, b in zip(cleaned, cleaned[1:]))

        self.assertGreater(raw, truth * 1.5, "raw trace should overstate the walk")
        self.assertLess(abs(measured - truth), truth * 0.08,
                        f"cleaned {measured:.1f} m should be near the true {truth:.1f} m")

    def test_clean_path_keeps_the_endpoint(self):
        fixes = straight_line(n=40, spacing_m=1.0)
        cleaned = clean_path(fixes, GEOMETRY_SPACING_M)
        self.assertEqual(cleaned[-1].lat, fixes[-1].lat)

    def test_border_flapping_is_one_visit_not_many(self):
        """Standing near a hex edge must not read as dozens of arrivals."""
        grid = HexGrid(Projection(LAT0, LON0), 120.0)
        inside = grid.center(Hex(0, 0))
        across = grid.center(Hex(1, 0))
        t0 = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
        fixes = []
        for i in range(24):
            lat, lon = across if i % 2 else inside
            fixes.append(Fix(lat=lat, lon=lon, t=t0 + timedelta(seconds=5 * i)))
        fixes.append(Fix(lat=inside[0], lon=inside[1], t=t0 + timedelta(seconds=125)))

        segments = _segments(fixes, grid)
        self.assertEqual(len(segments), 1, "short excursions should collapse into one stay")

        stats = extract_cells([make_walk(fixes)], grid)
        self.assertEqual(sum(s.b_visits for s in stats.values()), 1)

    def test_climb_is_per_distance_not_per_visit(self):
        """A long flat walk along a slope-free line gains nothing."""
        projection = Projection(LAT0, LON0)
        fixes = []
        for i in range(120):
            lat, lon = projection.to_latlon(i * 5.0, 0.0)
            fixes.append(Fix(lat=lat, lon=lon, ele=1500.0))
        self.assertEqual(climb_per_100m(fixes, 595.0), 0.0)

        hillside = []
        for i in range(120):
            lat, lon = projection.to_latlon(i * 5.0, 0.0)
            hillside.append(Fix(lat=lat, lon=lon, ele=1500.0 + i * 2.0))
        self.assertAlmostEqual(climb_per_100m(hillside, 595.0), 40.0, delta=1.0)

    def test_climb_ignores_sub_metre_barometric_noise(self):
        projection = Projection(LAT0, LON0)
        fixes = []
        for i in range(200):
            lat, lon = projection.to_latlon(i * 5.0, 0.0)
            fixes.append(Fix(lat=lat, lon=lon, ele=1500.0 + (0.4 if i % 2 else -0.4)))
        self.assertEqual(climb_per_100m(fixes, 995.0), 0.0)

    def test_evidence_gate_rejects_a_clipped_corner(self):
        """Crossing a hex is not visiting it."""
        grid = HexGrid(Projection(LAT0, LON0), 120.0)
        projection = grid.projection
        centre_lat, centre_lon = grid.center(Hex(3, -2))
        clipped = []
        for i in range(3):
            lat, lon = projection.to_latlon(projection.to_xy(centre_lat, centre_lon)[0]
                                            + i * 1.0, 0.0)
            clipped.append(Fix(lat=lat, lon=lon))
        stats = extract_cells([make_walk(clipped)], grid)
        charted, rejected = charted_cells(stats, Evidence())
        self.assertEqual(charted, {})
        self.assertEqual(len(rejected), 1)

    def test_features_are_all_produced_and_finite_where_expected(self):
        walks = generate_neighbourhood(days=3, seed=5)
        grid = HexGrid(Projection(LAT0, LON0), 120.0)
        stats = extract_cells(walks, grid)
        self.assertTrue(stats)
        for cell_stats in stats.values():
            vector = cell_stats.vector()
            self.assertEqual(len(vector), len(FEATURES))
            self.assertTrue(all(isinstance(v, float) for v in vector))


class TestArchetypes(unittest.TestCase):
    def test_primer_specifies_every_column(self):
        """A half-filled support set makes 'unknown' look like signal."""
        X, y = primer_matrix()
        self.assertEqual(len(X), len(y))
        for row in X:
            self.assertEqual(len(row), len(FEATURES))
            self.assertFalse(any(math.isnan(v) for v in row),
                             "every primer row must specify every feature")
        self.assertTrue(set(y).issubset(set(ARCHETYPE_IDS)))

    def test_reference_classifier_survives_nan_input(self):
        X, y = primer_matrix()
        model = NearestNeighbourClassifier().fit(X, y)
        queried = [[math.nan] * len(FEATURES), [0.0] * len(FEATURES)]
        labels = model.predict(queried)
        self.assertEqual(len(labels), 2)
        self.assertTrue(all(label in ARCHETYPE_IDS for label in labels))

    def test_anchor_hex_is_always_the_threshold(self):
        grid = HexGrid(Projection(LAT0, LON0), 120.0)
        walks = generate_neighbourhood(days=4, seed=2)
        stats = extract_cells(walks, grid)
        charted, _ = charted_cells(stats)
        predictions, _backend, _model = classify_cells(charted, {}, "knn")
        self.assertEqual(predictions[Hex(0, 0)][0], "threshold")

    def test_user_labels_enter_the_support_set(self):
        grid = HexGrid(Projection(LAT0, LON0), 120.0)
        stats = extract_cells(generate_neighbourhood(days=3, seed=4), grid)
        charted, _ = charted_cells(stats)
        target = sorted(charted, key=lambda c: c.key)[-1]
        predictions, _backend, _model = classify_cells(
            charted, {target.key: "gathering_place"}, "knn"
        )
        self.assertEqual(predictions[target][0], "gathering_place")


class TestLegend(unittest.TestCase):
    def setUp(self):
        self.stats = CellStats(cell=Hex(0, 0))
        self.stats.b_visits = 4
        self.stats.d_dwell_s = 620.0
        self.stats.j_distance_from_anchor_m = 240.0
        from terratale.archetypes import ARCHETYPE_BY_ID

        self.archetype = ARCHETYPE_BY_ID["quiet_grove"]

    def test_validation_rejects_a_name_already_on_the_map(self):
        with self.assertRaises(_Validation):
            _validate({"a_name": "Thornleigh", "b_legend": "It is 4 visits. More text here.",
                       "c_inscription": "quiet"}, ["Thornleigh"], self.stats)

    def test_validation_requires_the_legend_to_cite_a_measurement(self):
        with self.assertRaises(_Validation):
            _validate({"a_name": "Fernwick", "b_legend": "A pleasant place. Quite nice indeed.",
                       "c_inscription": "quiet"}, [], self.stats)

    def test_validation_accepts_a_well_formed_reply(self):
        legend = _validate(
            {"a_name": "Fernwick",
             "b_legend": "A slow corner. The trace records 4 arrivals and 10.3 minutes inside it.",
             "c_inscription": "where the walking slows"},
            [], self.stats,
        )
        self.assertEqual(legend.name, "Fernwick")
        self.assertEqual(legend.author, "gemma")

    def test_parses_json_wrapped_in_prose_or_fences(self):
        self.assertEqual(_parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(_parse_json_object('Sure thing: {"a": 1} there you go'), {"a": 1})
        with self.assertRaises(ValueError):
            _parse_json_object("no json at all")

    def test_offline_writer_is_deterministic_and_grounded(self):
        writer = OfflineCartographer()
        first = writer.name_cell(self.stats, self.archetype, [], ["walk-a"])
        second = writer.name_cell(self.stats, self.archetype, [], ["walk-a"])
        self.assertEqual(first.name, second.name)
        self.assertIn("4", first.text, "the legend must cite the measured visits")
        self.assertEqual(first.author, "offline-template")

    def test_offline_writer_avoids_names_already_taken(self):
        writer = OfflineCartographer()
        taken = [writer.name_cell(self.stats, self.archetype, [], []).name]
        fresh = writer.name_cell(self.stats, self.archetype, taken, [])
        self.assertNotIn(fresh.name, taken)


class TestState(unittest.TestCase):
    def test_roundtrip_preserves_anchor_walks_and_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "atlas.json"
            atlas = Atlas(anchor_lat=LAT0, anchor_lon=LON0, title="Test Map")
            for walk in generate_neighbourhood(days=3, seed=9):
                store_walk(atlas, walk)
            atlas.labelled["2:1"] = "crossroads"
            atlas.save(path)

            reloaded = Atlas.load(path)
            self.assertEqual(reloaded.anchor_lat, LAT0)
            self.assertEqual(reloaded.title, "Test Map")
            self.assertEqual(len(reloaded.walks), len(atlas.walks))
            self.assertEqual(reloaded.labelled["2:1"], "crossroads")
            self.assertAlmostEqual(reloaded.total_walk_distance_m(),
                                   atlas.total_walk_distance_m(), places=3)

    def test_stored_distance_matches_the_walk_not_the_noise(self):
        walk = make_walk(straight_line(n=900, spacing_m=1.4, jitter_m=3.2))
        atlas = Atlas(anchor_lat=LAT0, anchor_lon=LON0)
        record = store_walk(atlas, walk)
        truth = 899 * 1.4
        self.assertLess(abs(record["distance_m"] - truth), truth * 0.10)

    def test_rebuilt_walks_keep_their_timestamps(self):
        walk = make_walk(straight_line(n=300, spacing_m=1.4))
        atlas = Atlas(anchor_lat=LAT0, anchor_lon=LON0)
        store_walk(atlas, walk)
        rebuilt = walks_from_atlas(atlas)
        self.assertEqual(len(rebuilt), 1)
        self.assertTrue(all(f.t is not None for f in rebuilt[0].fixes))


class TestRendering(unittest.TestCase):
    def test_colour_parsing_handles_shorthand_and_long_form(self):
        self.assertEqual(_hex_rgb("#fff"), (255, 255, 255))
        self.assertEqual(_hex_rgb("#3b2f23"), (59, 47, 35))
        self.assertEqual(_hex_rgb("#3b2f23ff"), (59, 47, 35))
        with self.assertRaises(ValueError):
            _hex_rgb("not a colour")

    def test_fresh_atlas_renders_a_blank_sheet(self):
        canvas = build_map(Atlas(anchor_lat=LAT0, anchor_lon=LON0))
        self.assertGreater(canvas.width, 0)
        svg = to_svg(canvas)
        self.assertIn("<svg", svg)

    def test_all_outputs_are_written_and_the_pdf_is_well_formed(self):
        atlas = Atlas(anchor_lat=LAT0, anchor_lon=LON0, title="Output Test")
        for walk in generate_neighbourhood(days=2, seed=8):
            store_walk(atlas, walk)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_outputs(atlas, tmp, MapOptions(page="a5", dpi=36.0))
            for kind in ("svg", "png", "pdf"):
                path = paths[kind]
                self.assertTrue(Path(path).exists(), f"{kind} missing")
                self.assertGreater(Path(path).stat().st_size, 100, f"{kind} is empty")
            pdf = Path(paths["pdf"]).read_bytes()
            self.assertTrue(pdf.startswith(b"%PDF-"))
            self.assertIn(b"startxref", pdf)
            # no reveal has run, so the gazetteer is legitimately near-empty
            gazetteer = Path(paths["gazetteer"])
            self.assertTrue(gazetteer.exists())
            self.assertIn("# Gazetteer of Output Test", gazetteer.read_text())

    def test_gazetteer_says_nothing_is_charted_when_it_is(self):
        self.assertIn("_Nothing charted yet._",
                      gazetteer_markdown(Atlas(anchor_lat=LAT0, anchor_lon=LON0)))


class TestQuest(unittest.TestCase):
    def test_compass_labels_are_sane(self):
        self.assertEqual(compass_label(0.0), "north")
        self.assertEqual(compass_label(90.0), "east")
        self.assertEqual(compass_label(180.0), "south")
        self.assertEqual(compass_label(270.0), "west")
        self.assertEqual(compass_label(359.0), "north")

    def test_quest_targets_ground_that_is_not_charted(self):
        atlas = Atlas(anchor_lat=LAT0, anchor_lon=LON0)
        for walk in generate_neighbourhood(days=3, seed=6):
            store_walk(atlas, walk)
        grid = atlas.grid
        stats = extract_cells(walks_from_atlas(atlas), grid)
        charted, _ = charted_cells(stats)
        for cell, cell_stats in charted.items():
            atlas.cells[cell.key] = atlas.cells.get(cell.key) or __import__(
                "terratale.state", fromlist=["CellRecord"]
            ).CellRecord(cell=cell.key, archetype="quiet_grove", stats=cell_stats.to_dict())

        quests = suggest_quests(atlas, count=3)
        self.assertTrue(quests)
        for quest in quests:
            self.assertNotIn(quest["cell"], atlas.cells)
            self.assertLessEqual(quest["distance_m"], 3000.0)
            self.assertTrue(quest["instruction"])

    def test_quest_card_renders_every_page_size(self):
        """The card is a separate code path from the map and must not rot."""
        atlas = Atlas(anchor_lat=LAT0, anchor_lon=LON0)
        quest = {
            "headline": "Chart the hex 0.62 km west",
            "instruction": "Walk west and stay a few minutes.",
            "bearing_deg": 271.0,
            "bearing_compass": "west",
            "distance_m": 620.0,
        }
        for page in ("a5", "a4", "letter"):
            canvas = build_quest_card(atlas, quest, MapOptions(page=page, dpi=36.0))
            self.assertGreater(canvas.width, 0)
            self.assertIn("A QUEST", to_svg(canvas))


class TestEvidence(unittest.TestCase):
    def test_gate_reports_why_a_hex_was_refused(self):
        stats = CellStats(cell=Hex(0, 0), a_points=2, g_path_len_m=4.0, c_days=1)
        ok, why = Evidence().passes(stats)
        self.assertFalse(ok)
        self.assertIn("fixes", why)


if __name__ == "__main__":
    unittest.main(verbosity=2)
