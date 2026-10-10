"""Legend writing: Gemma, running locally, describing places you really went.

Two Ollama behaviours found in the wild shape this module, and both are silent
when they bite:

1. On the "native" chat path (Ollama 0.35.x), the request schema is decoded
   into a Go map before reaching llama-server, and Go writes map keys in
   alphabetical order. The grammar then enforces *that* order, so a model
   answering in the order the prompt asked puts values in the wrong fields and
   still returns valid JSON. The fix we apply is the documented one: name keys
   so alphabetical order *is* the order we want.

2. With a thinking model, the format schema is applied as a grammar wrapped
   around the thinking block. A reply that skips thinking is "text before the
   closing", which the grammar leaves unconstrained -- so the shape is not
   guaranteed at all. We therefore send ``think: false`` and validate every
   reply in this process, retrying with a blunter instruction.

Neither behaviour is worth trusting, so nothing here is trusted: a legend only
reaches the map after it validates, and every record carries the backend that
actually wrote it.
"""

from __future__ import annotations

import json
import math
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .archetypes import Archetype

# Keys are prefixed so that alphabetical order matches the order we want the
# model to fill them in: a_name -> b_legend -> c_inscription.
LEGEND_SCHEMA = {
    "type": "object",
    "properties": {
        "a_name": {"type": "string"},
        "b_legend": {"type": "string"},
        "c_inscription": {"type": "string"},
    },
    "required": ["a_name", "b_legend", "c_inscription"],
}


@dataclass
class Legend:
    """What the map knows about one place."""

    name: str
    text: str
    inscription: str = ""
    author: str = "unknown"
    revised: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "text": self.text,
            "inscription": self.inscription,
            "author": self.author,
            "revised": self.revised,
        }


class OllamaUnavailable(RuntimeError):
    pass


class OllamaClient:
    """Minimal local client for the Ollama HTTP API (stdlib only)."""

    def __init__(self, host: str = "http://127.0.0.1:11434", timeout: float = 120.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaUnavailable(f"cannot reach Ollama at {self.host}: {exc}") from exc

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(f"{self.host}{path}", timeout=10.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaUnavailable(f"cannot reach Ollama at {self.host}: {exc}") from exc

    def models(self) -> list[str]:
        payload = self._get("/api/tags")
        return [entry.get("name", "") for entry in payload.get("models", [])]

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict | None = None,
        temperature: float = 0.8,
    ) -> str:
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
            # Required: with a thinking model the schema is only enforced on the
            # part of the reply after the thinking block, and a reply that skips
            # thinking escapes the grammar entirely.
            "think": False,
        }
        if schema is not None:
            payload["format"] = schema
        body = self._post("/api/chat", payload)
        message = body.get("message", {})
        return message.get("content", "") or ""


def _feature_summary(stats, archetype: Archetype) -> str:
    """Render the measured facts as a compact sheet for the prompt.

    Only quantities that were actually measured are listed. A GPX export with
    no timestamps produces a shorter sheet rather than invented numbers, and
    the prompt tells the model to leave time-dependent detail out.
    """
    lines = [f"- this kind of place reads as: {archetype.signal}"]
    if stats.a_points:
        lines.append(f"- GPS fixes recorded here: {stats.a_points}")
    if stats.b_visits:
        lines.append(f"- separate arrivals: {stats.b_visits}")
    if stats.c_days:
        lines.append(f"- distinct days: {stats.c_days}")
    if not math.isnan(stats.d_dwell_s):
        lines.append(f"- time spent inside: {stats.d_dwell_s / 60.0:.1f} minutes")
    if not math.isnan(stats.e_median_speed):
        lines.append(f"- typical speed: {stats.e_median_speed * 3.6:.1f} km/h")
    if stats.g_path_len_m:
        lines.append(f"- distance walked inside: {stats.g_path_len_m:.0f} m")
    if not math.isnan(stats.h_straightness):
        shape = "straight through" if stats.h_straightness > 0.7 else "looping or lingering"
        lines.append(f"- movement shape: {shape} ({stats.h_straightness:.2f} of a straight line)")
    if not math.isnan(stats.i_turn_rate):
        lines.append(f"- turning: {stats.i_turn_rate:.1f} degrees per metre")
    if not math.isnan(stats.j_distance_from_anchor_m):
        lines.append(f"- distance from the anchor point: {stats.j_distance_from_anchor_m:.0f} m")
    if not math.isnan(stats.k_hour_mean):
        lines.append(f"- usual hour: about {int(stats.k_hour_mean) % 24:02d}:00")
    if not math.isnan(stats.n_night_frac) and stats.n_night_frac > 0:
        lines.append(f"- after dark on {stats.n_night_frac * 100:.0f}% of visits")
    if not math.isnan(stats.m_weekend_frac) and stats.m_weekend_frac > 0:
        lines.append(f"- weekends: {stats.m_weekend_frac * 100:.0f}% of visits")
    if not math.isnan(stats.o_elevation_gain_m) and stats.o_elevation_gain_m > 0:
        lines.append(f"- climbing done inside: {stats.o_elevation_gain_m:.0f} m of ascent")
    if not math.isnan(stats.p_revisit_span_days) and stats.p_revisit_span_days > 1:
        lines.append(f"- first and last visit {stats.p_revisit_span_days:.0f} days apart")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are the cartographer of a hand-drawn treasure map of one real "
    "neighbourhood. Walkers carry the map, and the map only fills in where "
    "they have physically been. Given the measurements taken in one hex of "
    "that map, you name the place and write its legend.\n\n"
    "Rules:\n"
    "1. The legend must be grounded in the measurements you are given. Quote "
    "at least one real number from the sheet.\n"
    "2. Never invent a landmark, business, street name, or animal that is not "
    "implied by the measurements. If the sheet is thin, write a shorter legend "
    "rather than inventing detail.\n"
    "3. Tone: an old field journal. Plain, warm, concrete. Not flowery.\n"
    "4. The name is one to three words and must not already be taken.\n"
    "5. The inscription is at most eight words, like a line carved on a stone."
)


def _build_user_prompt(stats, archetype: Archetype, taken: list[str], walk_names: list[str]) -> str:
    taken_text = ", ".join(taken[-40:]) if taken else "(none yet)"
    walk_text = ", ".join(sorted(set(walk_names))) or "(unnamed walks)"
    return (
        f"Archetype assigned to this hex by the classifier: {archetype.label}.\n\n"
        f"Measurements from the GPS traces:\n{_feature_summary(stats, archetype)}\n\n"
        f"Walks this hex was recorded on: {walk_text}\n"
        f"Names already used elsewhere on this map (do not reuse): {taken_text}\n\n"
        "Write the name, the legend, and the inscription."
    )


class _Validation(Exception):
    pass


def _validate(payload: dict, taken: list[str], stats) -> Legend:
    """Trust nothing that comes back over the wire."""
    missing = [key for key in ("a_name", "b_legend", "c_inscription") if not payload.get(key)]
    if missing:
        raise _Validation(f"missing keys: {missing}")

    name = str(payload["a_name"]).strip().strip('"').strip()
    text = str(payload["b_legend"]).strip()
    inscription = str(payload["c_inscription"]).strip().strip('"').strip()
    notes: list[str] = []

    if not name:
        raise _Validation("empty name")
    if len(name) > 48:
        raise _Validation(f"name too long ({len(name)} chars)")
    if name.casefold() in {taken_name.casefold() for taken_name in taken}:
        raise _Validation(f"name already used: {name!r}")

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) < 2:
        raise _Validation("legend is shorter than two sentences")
    if len(text) > 700:
        raise _Validation(f"legend too long ({len(text)} chars)")

    has_number = bool(re.search(r"\d", text))
    if not has_number and not math.isnan(stats.d_dwell_s):
        raise _Validation("legend cites no measurement")
    if not has_number:
        notes.append("no measurable number available for this hex")

    if len(inscription.split()) > 10:
        raise _Validation("inscription too long")

    return Legend(name=name, text=text, inscription=inscription, author="gemma", notes=notes)


class GemmaCartographer:
    """Writes legends with a local open-weight Gemma via Ollama."""

    def __init__(
        self,
        model: str = "gemma4:e2b",
        client: OllamaClient | None = None,
        temperature: float = 0.85,
        attempts: int = 3,
    ):
        self.model = model
        self.client = client or OllamaClient()
        self.temperature = temperature
        self.attempts = attempts

    def available(self) -> tuple[bool, str]:
        try:
            models = self.client.models()
        except OllamaUnavailable as exc:
            return False, str(exc)
        if not models:
            return False, "Ollama is running but has no models pulled"
        if self.model not in models:
            return False, f"model {self.model!r} not pulled; available: {', '.join(models)}"
        return True, f"{self.model} via Ollama"

    def name_cell(self, stats, archetype: Archetype, taken: list[str], walk_names: list[str]) -> Legend:
        prompt = _build_user_prompt(stats, archetype, taken, walk_names)
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            instruction = prompt
            if attempt:
                instruction += (
                    "\n\nYour previous reply was rejected. Reply again with all three "
                    "fields filled, the legend at least two sentences, and at least one "
                    "number copied from the measurements. Keep every string short."
                )
            raw = self.client.chat(
                self.model,
                SYSTEM_PROMPT,
                instruction,
                schema=LEGEND_SCHEMA,
                temperature=self.temperature,
            )
            try:
                return _validate(_parse_json_object(raw), taken, stats)
            except (_Validation, ValueError) as exc:
                last_error = exc
        raise _Validation(f"no valid legend after {self.attempts} attempts: {last_error}")


def _parse_json_object(raw: str) -> dict:
    """Parse a reply that should be JSON, tolerating stray prose or fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


_SYLLABLES_A = (
    "Ash", "Bram", "Cold", "Dun", "Elm", "Fern", "Glas", "Haw", "Iron", "Kel",
    "Lark", "Moss", "Nether", "Oak", "Pike", "Quar", "Rook", "Salt", "Thorn", "Vale",
)
_SYLLABLES_B = (
    "bourne", "combe", "crag", "dale", "fen", "garth", "hollow", "kirk", "leigh",
    "marsh", "ness", "reach", "ridge", "shaw", "stead", "stone", "well", "wick",
)

_SHAPE_WORDS = {
    "threshold": "the place every walk begins and ends",
    "thoroughfare": "a channel others pass through without stopping",
    "crossroads": "a meeting of ways",
    "quiet_grove": "a slow corner that rewards dawdling",
    "gathering_place": "somewhere the walker halted and stayed",
    "overlook": "high ground reached by climbing",
    "lane_revisited": "a road worn smooth by habit",
    "hidden_hollow": "a place the map caught once and barely knows",
}

#: Inscriptions are carved on stone, so they stay short enough to fit on one
#: line of a printed map without being clipped.
_INSCRIPTIONS = {
    "threshold": "every walk begins here",
    "thoroughfare": "the way through",
    "crossroads": "where the ways meet",
    "quiet_grove": "walk slowly here",
    "gathering_place": "here the walker stopped",
    "overlook": "high ground, hard won",
    "lane_revisited": "worn smooth by habit",
    "hidden_hollow": "seen once, not yet known",
}


class OfflineCartographer:
    """Deterministic legend writer used when Gemma is not reachable.

    It composes from the same measurement sheet and the same archetype signal,
    so the map is complete and honest offline -- but it is a template, and
    every record it writes says so in ``author``. Nothing here should ever be
    presented as model output.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def available(self) -> tuple[bool, str]:
        return True, "offline template composer (no model)"

    def name_cell(self, stats, archetype: Archetype, taken: list[str], walk_names: list[str]) -> Legend:
        taken_lower = {name.casefold() for name in taken}
        rng = random.Random(f"{archetype.id}:{stats.cell.key}")
        name = ""
        for _ in range(200):
            candidate = f"{rng.choice(_SYLLABLES_A)}{rng.choice(_SYLLABLES_B)}"
            if candidate.casefold() not in taken_lower:
                name = candidate
                break
        if not name:
            name = f"{rng.choice(_SYLLABLES_A)}{rng.choice(_SYLLABLES_B)}{stats.cell.q}{stats.cell.r}"

        facts: list[str] = []
        if stats.b_visits:
            noun = "arrival" if stats.b_visits == 1 else "arrivals"
            facts.append(f"{stats.b_visits} separate {noun}")
        if not math.isnan(stats.d_dwell_s):
            if stats.d_dwell_s < 60.0:
                facts.append("well under a minute inside it")
            else:
                facts.append(f"{stats.d_dwell_s / 60.0:.1f} minutes inside it")
        if not math.isnan(stats.j_distance_from_anchor_m):
            facts.append(f"{stats.j_distance_from_anchor_m:.0f} m from the anchor")
        if stats.g_path_len_m:
            facts.append(f"{stats.g_path_len_m:.0f} m walked within it")

        opening = f"{name} is {_SHAPE_WORDS.get(archetype.id, 'a hex on this map')}."
        if facts:
            body = "The trace records " + ", ".join(facts) + "."
        else:
            body = "The trace is thin here, so the map keeps this place short."
        note = ""
        if not math.isnan(stats.h_straightness) and stats.h_straightness < 0.4:
            note = " Whoever walks it tends to circle rather than leave."
        elif not math.isnan(stats.h_straightness) and stats.h_straightness > 0.8:
            note = " Nobody lingers; the line runs clean through."

        inscription = _INSCRIPTIONS.get(archetype.id, "charted")[:40]
        return Legend(
            name=name,
            text=f"{opening} {body}{note}",
            inscription=inscription,
            author="offline-template",
        )


def build_cartographer(
    mode: str = "auto",
    model: str = "gemma4:e2b",
    client: OllamaClient | None = None,
):
    """Return (cartographer, description). ``auto`` prefers Gemma."""
    if mode in {"auto", "gemma"}:
        cartographer = GemmaCartographer(model=model, client=client)
        ok, detail = cartographer.available()
        if ok:
            return cartographer, detail
        if mode == "gemma":
            raise OllamaUnavailable(detail)
        fallback = OfflineCartographer()
        return fallback, f"offline template composer (Gemma unavailable: {detail})"
    return OfflineCartographer(), "offline template composer (requested)"
