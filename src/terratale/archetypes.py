"""Place archetypes, and the model that assigns them.

TabPFN does the load-bearing work here, and it is a genuinely good fit rather
than a bolt-on: the dataset is a few dozen rows, there is a real label budget
of near zero, and the taxonomy is personal. A walker labels two or three
places they care about and expects the rest of the neighbourhood to follow --
that is in-context learning on tiny tabular data, which is the one thing
TabPFN is better at than anything else in the open ecosystem.

The support set is the interface. `primer_rows()` ships a portable prior that
works on a cold start, and every user label is appended to the same table, so
a correction immediately re-shapes every other prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .features import FEATURES, CellStats
from .grid import Hex


@dataclass(frozen=True)
class Archetype:
    """A kind of place the map knows how to draw and write about."""

    id: str
    label: str
    glyph: str
    ink: str
    signal: str


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        "threshold", "Threshold", "⌂", "#7a3b2e",
        "at the anchor point, entered on nearly every walk, high dwell, "
        "almost no through-movement",
    ),
    Archetype(
        "thoroughfare", "Thoroughfare", "═", "#3c4f6b",
        "passed straight through at walking or cycling pace, high "
        "straightness, short dwell, visited very often",
    ),
    Archetype(
        "crossroads", "Crossroads", "✕", "#6b4a2f",
        "lots of heading change while still moving fast, medium dwell, "
        "many separate visits",
    ),
    Archetype(
        "quiet_grove", "Quiet Grove", "❦", "#2f5d3a",
        "slow, wandering, low straightness, no urgency, middling dwell",
    ),
    Archetype(
        "gathering_place", "Gathering Place", "◉", "#8a6d1f",
        "the walker stopped here: very high dwell against a very short path "
        "length, low straightness, often revisited",
    ),
    Archetype(
        "overlook", "Overlook", "▲", "#5b3a7a",
        "real elevation gain inside the hex and a long way from the anchor",
    ),
    Archetype(
        "lane_revisited", "Revisited Lane", "↺", "#4a6b6b",
        "the habit path: many distinct days, steady speed, moderate "
        "straightness, stable hour of day",
    ),
    Archetype(
        "hidden_hollow", "Hidden Hollow", "◆", "#4d4d4d",
        "reached rarely or once, low dwell before leaving, far from the "
        "anchor, no repeat pattern",
    ),
)

ARCHETYPE_BY_ID: dict[str, Archetype] = {a.id: a for a in ARCHETYPES}
ARCHETYPE_IDS: tuple[str, ...] = tuple(a.id for a in ARCHETYPES)
#: Convenience for help text and human-facing listings.
ARCHETYPE_BY_LABEL_PROXY: dict[str, str] = {a.label: a.id for a in ARCHETYPES}


#: Column order for the compact primer table below. Keeping the table
#: positional guarantees every row specifies every feature: a half-filled
#: support set is worse than a small one, because a model that meets "unknown"
#: only where it expected a number will lean on that gap as if it were signal.
_PRIMER_COLUMNS = (
    "a_points", "b_visits", "c_days", "d_dwell_s", "e_median_speed", "f_p90_speed",
    "g_path_len_m", "h_straightness", "i_turn_rate", "j_distance_from_anchor_m",
    "k_hour_mean", "l_hour_spread", "m_weekend_frac", "n_night_frac",
    "o_climb_per_100m", "p_revisit_span_days",
)

#                 points visits days  dwell  med  p90  path  str  turn  dist   hour spread wknd night climb span
_PRIMER_TABLE = (
    ("threshold",       (120,  10,   9,  1500, 0.45, 0.9,   90, 0.20, 1.6,   12,  7.4, 1.4, 0.25, 0.05, 0.5, 40)),
    ("threshold",       ( 70,   6,   5,   900, 0.50, 1.0,   70, 0.28, 1.9,   22, 17.8, 1.9, 0.50, 0.10, 0.8, 20)),
    ("thoroughfare",    ( 40,  12,  10,    55, 1.45, 1.7,  340, 0.94, 0.35, 900,  8.1, 2.2, 0.15, 0.05, 0.3, 55)),
    ("thoroughfare",    ( 28,   7,   6,    40, 2.40, 3.1,  380, 0.96, 0.22, 600, 18.2, 1.8, 0.30, 0.20, 0.2, 30)),
    ("crossroads",      ( 35,   8,   7,   130, 1.15, 1.5,  240, 0.72, 2.1,  450,  9.0, 2.6, 0.25, 0.08, 0.6, 35)),
    ("crossroads",      ( 20,   4,   4,   100, 1.05, 1.4,  190, 0.80, 1.7,  300, 16.5, 2.0, 0.60, 0.05, 0.5, 12)),
    ("quiet_grove",     ( 70,   4,   4,   520, 0.80, 1.1,  200, 0.50, 2.2,  220, 15.0, 2.4, 0.55, 0.00, 1.0, 18)),
    ("quiet_grove",     ( 45,   3,   3,   430, 0.70, 1.0,  170, 0.42, 2.6,  340, 10.5, 1.6, 0.80, 0.00, 1.2,  8)),
    ("gathering_place", (150,   5,   5,  2100, 0.18, 0.5,   45, 0.10, 1.0,  500, 12.0, 3.0, 0.60, 0.00, 0.8, 25)),
    ("gathering_place", ( 95,   3,   3,  1400, 0.22, 0.6,   40, 0.15, 1.3,  260, 17.0, 2.2, 0.40, 0.33, 1.0, 14)),
    ("overlook",        ( 55,   3,   3,   560, 0.90, 1.2,  230, 0.58, 2.2, 1800, 16.0, 2.0, 0.70, 0.00, 6.2, 10)),
    ("overlook",        ( 38,   2,   2,   440, 1.00, 1.3,  210, 0.62, 2.5, 2400,  8.0, 1.5, 0.50, 0.00, 9.5,  7)),
    ("lane_revisited",  ( 30,  15,  12,   110, 1.35, 1.6,  250, 0.92, 0.40, 700,  7.2, 0.7, 0.10, 0.00, 0.4, 60)),
    ("lane_revisited",  ( 22,  10,   9,    90, 1.30, 1.5,  220, 0.90, 0.50, 420, 17.6, 0.9, 0.20, 0.11, 0.6, 45)),
    ("hidden_hollow",   ( 14,   1,   1,    70, 1.25, 1.6,  180, 0.85, 0.90,2600, 14.0, 0.05, 1.00, 0.00, 2.0,  0)),
    ("hidden_hollow",   ( 10,   1,   1,    45, 1.40, 1.8,  210, 0.88, 0.70,3300, 11.0, 0.05, 0.00, 0.00, 2.5,  0)),
)


def _row(values: tuple) -> list[float]:
    return [float(value) for value in values]


def primer_rows() -> list[tuple[str, list[float]]]:
    """The cold-start prior: what each archetype physically looks like.

    These are not fitted values. Walking pace is about 1.2 m/s, a straight
    road barely changes heading per metre, a bench is a large dwell over a
    very short path, and a hill is metres of climb per hundred metres walked.
    Expressing the prior in those units is what lets it travel to any city,
    and it is what makes a walker's own labels meaningful next to it.
    """
    if _PRIMER_COLUMNS != FEATURES:
        raise RuntimeError(
            "primer column order has drifted from the feature list; "
            "the prior is positional and would be silently misaligned"
        )
    return [(label, _row(values)) for label, values in _PRIMER_TABLE]


def primer_matrix() -> tuple[list[list[float]], list[str]]:
    rows = primer_rows()
    return [row for _label, row in rows], [label for label, _row_ in rows]


class NearestNeighbourClassifier:
    """Standardised, distance-weighted k-NN over the same support set.

    This exists so the pipeline is testable and offline-safe on a machine with
    no torch, and so the two backends can be compared on identical inputs. It
    is *not* presented as a TabPFN substitute: it is a reference point. On the
    real neighbourhood dataset TabPFN is the intended backend.
    """

    name = "knn-reference"

    def __init__(self, k: int = 5):
        self.k = k
        self.classes_: list[str] = []
        self._centers: list[float] = []
        self._scales: list[float] = []
        self._x: list[list[float]] = []
        self._y: list[str] = []

    def _prepare(self, X: list[list[float]]) -> None:
        width = len(X[0])
        self._centers, self._scales = [], []
        for col in range(width):
            values = [row[col] for row in X if not math.isnan(row[col])]
            if values:
                center = sum(values) / len(values)
                var = sum((v - center) ** 2 for v in values) / len(values)
                scale = math.sqrt(var) or 1.0
            else:
                center, scale = 0.0, 1.0
            self._centers.append(center)
            self._scales.append(scale)

    def _transform(self, X: list[list[float]]) -> list[list[float]]:
        """Scale the data and append a missing-indicator per feature.

        A real GPX export may carry no timestamps and no elevation at all, so
        "not measured" is a normal state here. Imputing quietly would make an
        unmeasured climb look like a measured flat one; the indicator keeps
        those two cases distinguishable.
        """
        out: list[list[float]] = []
        for row in X:
            scaled: list[float] = []
            for i, value in enumerate(row):
                if math.isnan(value):
                    scaled.append(0.0)
                    scaled.append(1.0)
                else:
                    scaled.append((value - self._centers[i]) / self._scales[i])
                    scaled.append(0.0)
            out.append(scaled)
        return out

    def fit(self, X: list[list[float]], y: list[str]) -> "NearestNeighbourClassifier":
        self._prepare(X)
        self._x = self._transform(X)
        self._y = list(y)
        self.classes_ = sorted(set(y))
        return self

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        query = self._transform(X)
        index = {name: i for i, name in enumerate(self.classes_)}
        out: list[list[float]] = []
        for row in query:
            distances = []
            for i, support in enumerate(self._x):
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(row, support)))
                distances.append((dist, self._y[i]))
            distances.sort(key=lambda pair: pair[0])
            weights = {}
            for dist, name in distances[: self.k]:
                weights[name] = weights.get(name, 0.0) + 1.0 / (dist + 1e-6)
            total = sum(weights.values()) or 1.0
            probs = [0.0] * len(self.classes_)
            for name, weight in weights.items():
                probs[index[name]] = weight / total
            out.append(probs)
        return out

    def predict(self, X: list[list[float]]) -> list[str]:
        return [
            self.classes_[max(range(len(probs)), key=lambda i: probs[i])]
            for probs in self.predict_proba(X)
        ]


def build_classifier(backend: str = "auto"):
    """Return a fitted classifier plus the name of the backend actually used.

    `backend` is one of ``auto``, ``tabpfn`` or ``knn``. ``auto`` prefers
    TabPFN and silently falls back, but always reports which one ran so a
    legend is never mistaken for model output when it is not.
    """
    X, y = primer_matrix()
    if backend in {"auto", "tabpfn"}:
        try:
            from tabpfn import TabPFNClassifier  # noqa: PLC0415 - optional heavy dep

            for kwargs in ({"device": "cpu", "n_estimators": 8}, {"device": "cpu"}, {}):
                try:
                    model = TabPFNClassifier(**kwargs)
                    break
                except TypeError:
                    continue
            else:  # pragma: no cover - defensive
                model = TabPFNClassifier()
            model.fit(X, y)
            return model, "tabpfn"
        except Exception as exc:  # noqa: BLE001 - any import/runtime failure degrades
            if backend == "tabpfn":
                raise
            fallback = NearestNeighbourClassifier().fit(X, y)
            fallback.name = f"knn-reference (tabpfn unavailable: {type(exc).__name__})"
            return fallback, "knn"
    return NearestNeighbourClassifier().fit(X, y), "knn"


def label_cells(stats: dict, labelled: dict) -> tuple[list[list[float]], list[str]]:
    """Build the support set: portable primer plus the walker's own labels."""
    X, y = primer_matrix()
    for cell_key, archetype in labelled.items():
        # Labels are stored as text keys so they survive JSON; the feature
        # table is keyed by Hex objects. Getting this wrong drops the user's
        # teaching silently, which is the worst possible failure here.
        cell_stats = stats.get(cell_key)
        if cell_stats is None:
            try:
                cell_stats = stats.get(Hex.parse(cell_key))
            except (ValueError, AttributeError):
                cell_stats = None
        if cell_stats is None or archetype not in ARCHETYPE_BY_ID:
            continue
        X.append(cell_stats.vector())
        y.append(archetype)
    return X, y


def classify_cells(
    stats: dict,
    labelled: dict | None = None,
    backend: str = "auto",
):
    """Classify every cell, returning (predictions, backend_name, model)."""
    X, y = label_cells(stats, labelled or {})
    model, used = build_classifier(backend)
    model.fit(X, y)
    keys = list(stats.keys())
    matrix = [stats[key].vector() for key in keys]
    if not matrix:
        return {}, used, model
    probas = model.predict_proba(matrix)
    names = list(getattr(model, "classes_", sorted(set(y))))
    predictions = {}
    for key, row in zip(keys, probas):
        best = max(range(len(row)), key=lambda i: row[i])
        predictions[key] = (names[best], float(row[best]))

    # The anchor hex is where the walker lives: it is a definition, not an
    # inference, so it does not get to be overruled by its own speed profile.
    anchor = Hex(0, 0)
    if anchor in predictions and anchor.key not in (labelled or {}):
        predictions[anchor] = ("threshold", 1.0)
    return predictions, used, model
