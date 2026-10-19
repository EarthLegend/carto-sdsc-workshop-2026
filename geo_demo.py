"""The demo's interface: a chat dock beside a live map, inside a Jupyter notebook.

`GeoChat(agent)` is an app. Ask a question, and the agent's reasoning, each move it makes through
LGND Geo's tools (in plain words) and its findings stream into the chat, while the map draws what
each move looked at and found. Every imagery check the agent runs becomes a numbered finding:
green pins where the imagery supported it, red where it did not, cited in the answer as [1], [2].

Everything is ipywidgets and ipyleaflet, so it runs in the notebook with no server of its own.
"""

from __future__ import annotations

import asyncio
import base64
import html
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ipywidgets as W
import mistune
from ipyleaflet import (
    Circle,
    CircleMarker,
    DivIcon,
    LayerGroup,
    Map,
    Marker,
    Polygon,
    Rectangle,
    TileLayer,
    WidgetControl,
    basemaps,
)
from IPython.display import display
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.run import AgentRunResultEvent

# ----------------------------------------------------------------------------- look

ACCENT = "#38bdf8"
PLACE = "#fbbf24"  # a search match
CHANGE = "#fb7185"  # a change
CELL = "#38bdf8"  # a cell or a place looked at
FINDING = "#34d399"  # supported by the imagery
REJECTED = "#f87171"  # the imagery showed something else
CLASSES = ["#38bdf8", "#fbbf24", "#f472b6", "#4ade80", "#c084fc", "#fb923c", "#a3e635", "#e879f9"]

LABELS_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/"
    "World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
)

_LOGO_PATH = Path(__file__).with_name("assets") / "lgnd_logo_white.png"
_LOGO = (
    "data:image/png;base64," + base64.b64encode(_LOGO_PATH.read_bytes()).decode()
    if _LOGO_PATH.exists()
    else ""
)

CSS = """
<style>
.geo-app, .geo-app * { box-sizing: border-box; }
.geo-dock {
  --bg:#0b1120; --panel:#0f172a; --panel2:#172033; --line:rgba(255,255,255,.08); --line2:rgba(255,255,255,.16);
  --text:#e5ecf5; --muted:#94a3b8; --faint:#64748b; --accent:#38bdf8; --finding:#34d399; --rejected:#f87171; --checking:#fbbf24;
  --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: var(--panel); color: var(--text); border: 1px solid var(--line2); border-radius: 14px; overflow: hidden;
  font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; font-size: 13.5px; line-height: 1.5;
  box-shadow: 0 14px 50px rgba(0,0,0,.45);
}
.geo-dock .widget-html-content, .geo-dock .widget-html { color: var(--text); font-size: inherit; line-height: inherit; }
.geo-head { display:flex; align-items:center; gap:10px; padding: 12px 14px; border-bottom: 1px solid var(--line); }
.geo-head img { height: 18px; }
.geo-head .geo-title { font-weight: 600; font-size: 15px; }
.geo-head .geo-model { margin-left: 6px; color: var(--faint); font-size: 11.5px; }
.geo-dock .geo-new { margin-left: auto; }
.geo-dock button.geo-new, .geo-dock button.geo-example, .geo-dock button.geo-ask {
  font-family: inherit; border-radius: 999px; border: 1px solid var(--line2); background: transparent; color: var(--muted);
  font-size: 12px; height: 28px; padding: 0 12px; box-shadow: none;
}
.geo-dock button.geo-new:hover, .geo-dock button.geo-example:hover { color: var(--text); border-color: var(--accent); background: transparent; }
.geo-dock button.geo-example {
  width: 100%; height: auto; min-height: 40px; text-align: left; white-space: normal; line-height: 1.4; padding: 9px 12px;
  border-radius: 10px; background: var(--panel2); color: var(--text); font-size: 13px;
}
.geo-examples { padding: 12px 14px 4px; }
.geo-examples .geo-ex-h { color: var(--muted); font-size: 10.5px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; margin-bottom: 8px; }
.geo-dock .geo-scroll { display: flex; flex-direction: column-reverse; overflow-y: auto; padding: 12px 14px; }
.geo-dock .geo-scroll::-webkit-scrollbar { width: 8px; }
.geo-dock .geo-scroll::-webkit-scrollbar-thumb { background: var(--line2); border-radius: 4px; }
.geo-turns { display:flex; flex-direction:column; gap: 20px; }
.geo-turn { display:flex; flex-direction:column; gap: 10px; }
.geo-q { align-self:flex-end; max-width: 92%; padding: 8px 13px; border-radius: 14px 14px 4px 14px; background: rgba(56,189,248,.14); border: 1px solid rgba(56,189,248,.3); }
.geo-rail { position: relative; list-style: none; margin: 0; padding: 0; display:flex; flex-direction:column; gap: 8px; }
.geo-rail::before { content:''; position:absolute; left: 11px; top: 12px; bottom: 12px; width: 1.5px; background: var(--line2); }
.geo-beat { position: relative; display:flex; align-items:flex-start; gap: 10px; }
.geo-node { position: relative; z-index: 1; flex: 0 0 24px; height: 24px; margin-top: 8px; border-radius: 50%; background: var(--panel); border: 1.5px solid var(--line2); display:flex; align-items:center; justify-content:center; }
.geo-node i { display:block; width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
.geo-beat.live .geo-node { border-style: dashed; }
.geo-beat.running .geo-node i { width: 12px; height: 12px; border: 1.5px solid var(--line2); border-top-color: var(--accent); background: transparent; animation: geo-spin .8s linear infinite; }
.geo-beat.done .geo-node { border-color: var(--accent); } .geo-beat.done .geo-node i { background: var(--accent); }
.geo-beat.error .geo-node { border-color: var(--rejected); } .geo-beat.error .geo-node i { background: var(--rejected); }
.geo-beat.found .geo-node { border-color: var(--finding); } .geo-beat.found .geo-node i { background: var(--finding); }
.geo-card { flex: 1; min-width: 0; display:flex; flex-direction:column; gap: 6px; padding: 9px 12px; border: 1px solid var(--line); border-radius: 10px; background: rgba(23,32,51,.6); }
.geo-beat.live .geo-card { background: transparent; border-style: dashed; }
.geo-says { margin: 0; white-space: pre-wrap; }
.geo-think { color: var(--faint); font-size: 12px; }
.geo-think summary { cursor: pointer; list-style: none; }
.geo-think summary::-webkit-details-marker { display:none; }
.geo-think summary::before { content: '▸ '; } .geo-think[open] summary::before { content: '▾ '; }
.geo-think div { margin-top: 4px; padding-left: 10px; border-left: 1.5px solid var(--line2); color: #a8b3c7; font-style: italic; font-size: 12.5px; white-space: pre-wrap; }
.geo-dots { display:inline-flex; gap: 4px; vertical-align: middle; } .geo-dots i { width:5px; height:5px; border-radius:50%; background: var(--faint); animation: geo-blink 1.2s infinite; }
.geo-dots i:nth-child(2){animation-delay:.2s} .geo-dots i:nth-child(3){animation-delay:.4s}
.geo-line { font-size: 13px; } .geo-result { color: var(--muted); font-size: 12px; } .geo-err { color: var(--rejected); font-size: 12px; white-space: pre-wrap; }
.geo-says + .geo-did { padding-top: 6px; border-top: 1px solid var(--line); }
.geo-finding { display:grid; grid-template-columns: auto 1fr; gap: 8px; padding: 8px 9px; border-radius: 8px; font-size: 12.5px; border: 1px solid rgba(251,191,36,.4); background: rgba(251,191,36,.08); }
.geo-finding.ok { border-color: rgba(52,211,153,.35); background: rgba(52,211,153,.08); }
.geo-finding.rejected { border-color: rgba(248,113,113,.4); background: rgba(248,113,113,.08); }
.geo-fn { display:inline-flex; align-items:center; justify-content:center; width: 18px; height: 18px; border-radius: 50%; background: var(--checking); color: var(--bg); font: 700 10px var(--mono); margin-top: 2px; }
.geo-finding.ok .geo-fn { background: var(--finding); } .geo-finding.rejected .geo-fn { background: var(--rejected); }
.geo-finding.place .geo-fn { border-radius: 4px; }
.geo-fk { display:block; font-size: 10px; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; color: var(--checking); }
.geo-finding.ok .geo-fk { color: var(--finding); } .geo-finding.rejected .geo-fk { color: var(--rejected); }
.geo-fc { display:block; margin-top: 4px; color: var(--faint); font-size: 11.5px; }
.geo-answer { font-size: 13.5px; line-height: 1.62; color: #cbd5e1; }
.geo-answer p { margin: 0 0 10px; } .geo-answer p:last-child { margin-bottom: 0; }
.geo-answer > p:first-child { font-size: 15px; font-weight: 500; color: #fff; }
.geo-answer h1, .geo-answer h2, .geo-answer h3 { margin: 14px 0 6px; padding-top: 10px; border-top: 1px solid var(--line); color: var(--muted); font-size: 10.5px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; }
.geo-answer ul, .geo-answer ol { margin: 0 0 10px; padding-left: 18px; } .geo-answer li { margin-bottom: 6px; }
.geo-answer strong { color: #fff; } .geo-answer code { font-family: var(--mono); font-size: 12px; background: var(--panel2); padding: 1px 4px; border-radius: 4px; }
.geo-answer table { border-collapse: collapse; font-size: 12.5px; margin: 6px 0 10px; } .geo-answer td, .geo-answer th { border: 1px solid var(--line2); padding: 3px 8px; text-align: left; }
.geo-cite { display:inline-flex; align-items:center; justify-content:center; min-width: 18px; height: 18px; padding: 0 5px; margin: 0 1px; border-radius: 999px; border: 1px solid var(--finding); background: rgba(52,211,153,.12); color: var(--finding); font: 600 10.5px var(--mono); vertical-align: 2px; }
.geo-cite.rejected { border-color: var(--rejected); background: rgba(248,113,113,.12); color: var(--rejected); }
.geo-cite.checking { border-color: var(--checking); background: rgba(251,191,36,.12); color: var(--checking); }
.geo-timing { color: var(--faint); font-size: 11.5px; }
.geo-turn-err { padding: 8px 10px; border-radius: 8px; background: rgba(248,113,113,.12); border: 1px solid rgba(248,113,113,.4); color: var(--rejected); font-size: 12.5px; white-space: pre-wrap; }
.geo-composer { display:flex; gap: 8px; align-items:center; padding: 10px 12px 12px; border-top: 1px solid var(--line); }
.geo-composer .widget-text { flex: 1; }
.geo-composer .widget-text input { height: 40px; border-radius: 10px; border: 1px solid var(--line2); background: var(--panel2); color: var(--text); font-family: inherit; font-size: 13.5px; padding: 0 12px; }
.geo-composer .widget-text input:focus { border-color: var(--accent); outline: none; box-shadow: none; }
.geo-dock button.geo-ask { height: 40px; min-width: 64px; border: 0; border-radius: 10px; background: var(--accent); color: #052436; font-weight: 600; font-size: 13.5px; }
.geo-dock button.geo-ask:disabled { opacity: .4; }
.geo-caption { max-width: 560px; padding: 8px 14px; border-radius: 12px; background: rgba(13,19,33,.9); border: 1px solid rgba(255,255,255,.16); color: #e5ecf5; font: 13px Inter, system-ui, sans-serif; line-height: 1.4; }
.geo-caption .geo-cap-r { color: #94a3b8; font-size: 11.5px; }
.geo-legend { display:flex; flex-wrap: wrap; gap: 4px 12px; margin-top: 5px; color: #94a3b8; font-size: 11px; }
.geo-legend i { display:inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
.leaflet-div-icon { background: transparent; border: 0; }
.geo-pin { width: 24px; height: 24px; border-radius: 50%; border: 2px solid #0b1120; background: #34d399; color: #0b1120; font: 700 11.5px ui-monospace, Menlo, monospace; display:flex; align-items:center; justify-content:center; box-shadow: 0 2px 8px rgba(0,0,0,.6); }
.geo-pin.rejected { background: #f87171; } .geo-pin.checking { background: #fbbf24; } .geo-pin.place { border-radius: 6px; }
.geo-pop { font: 12.5px Inter, system-ui, sans-serif; color: #0f172a; line-height: 1.45; min-width: 180px; }
.geo-pop b { display:block; margin-bottom: 2px; } .geo-pop .geo-pop-f { color: #64748b; font-size: 11.5px; }
@keyframes geo-spin { to { transform: rotate(360deg); } }
@keyframes geo-blink { 0%,80%,100% { opacity:.25 } 40% { opacity: 1 } }
</style>
"""

# ----------------------------------------------------------------------------- results

INDEX_WORDS = {"naip": "NAIP imagery", "s2": "Sentinel-2 imagery"}
KIND_LABEL = {"change": "A change", "place": "A place", "resemblance": "Looks like"}


def esc(s: Any) -> str:
    return html.escape(str(s))


def records(table: dict) -> list[dict]:
    """A tool result's table ({"columns", "rows"}) as a list of dicts."""
    return [dict(zip(table["columns"], row)) for row in table["rows"]]


def is_table(v: Any) -> bool:
    return isinstance(v, dict) and "columns" in v and "rows" in v and isinstance(v["rows"], list)


def located(result: Any):
    """Every record in a tool result that has a position."""
    if is_table(result):
        if {"lat", "lon"} <= set(result["columns"]):
            yield from records(result)
    elif isinstance(result, dict):
        if {"lat", "lon"} <= result.keys():
            yield result
        else:
            for v in result.values():
                yield from located(v)
    elif isinstance(result, list):
        for v in result:
            yield from located(v)


def noded(result: Any):
    """Every record in a tool result that names a node."""
    if is_table(result):
        if "node" in result["columns"]:
            yield from records(result)
    elif isinstance(result, dict):
        if "node" in result and isinstance(result["node"], str):
            yield result
        for v in result.values():
            if v is not result:
                yield from noded(v)
    elif isinstance(result, list):
        for v in result:
            yield from noded(v)


def first_table(result: dict) -> dict | None:
    if is_table(result):
        return result
    for key in ("matches", "places", "cells", "targets", "rows"):
        if is_table(result.get(key)):
            return result[key]
    for v in result.values():
        if is_table(v):
            return v
    return None


def phrases(v: Any) -> str:
    if isinstance(v, list):
        return " / ".join(str(x) for x in v)
    return "" if v is None else str(v)


def when(a: dict) -> str:
    s, e = a.get("start"), a.get("end")
    if s and e:
        return f" between {s} and {e}"
    if s:
        return f" since {s}"
    if e:
        return f" up to {e}"
    return ""


def where(area: Any, areas: dict[str, Any]) -> str:
    if not isinstance(area, dict):
        return ""
    for name, known in areas.items():
        if area == known:
            return f" over {name}"
    if "ring" in area:
        return f" over a {len(area['ring'])}-corner polygon"
    if "radius_km" in area:
        return f" within {area['radius_km']:g} km of {area['lat']:.3f}, {area['lon']:.3f}"
    if "south" in area:
        lat = (area["south"] + area["north"]) / 2
        h = (area["north"] - area["south"]) * 111
        w = (area["east"] - area["west"]) * 111 * math.cos(math.radians(lat))
        return f" over a {w:.0f} × {h:.0f} km box"
    return " over an area"


def describe(tool: str, a: dict, areas: dict[str, Any]) -> str:
    """A move in plain words, the way the chat and the map caption say it."""
    w = where(a.get("area"), areas)
    t = when(a)
    idx = INDEX_WORDS.get(a.get("index"), "the imagery")
    match tool:
        case "search_place_find":
            if a.get("text"):
                return f"Searching {idx} for “{phrases(a['text'])}”{w or ' anywhere'}{t}"
            return f"Searching for locations similar to {a.get('like')}{w}{t}"
        case "search_change_find":
            after, before = phrases(a.get("after")), phrases(a.get("before"))
            if after and before:
                core = f"locations that changed from “{before}” to “{after}”"
            elif after:
                core = f"locations that became “{after}”"
            else:
                core = f"locations that were “{before}”"
            return f"Searching {idx} for {core}{w or ' anywhere'}{t}"
        case "search_similar_change_find":
            return f"Searching for changes similar to {a.get('node')}{w}{t}"
        case "discover_area_view":
            lv = a.get("level")
            at = " at the image level" if lv == "observations" else f" at level {lv}" if lv else ""
            return f"Overview of {idx}{w or ' everywhere'}{at}{t}"
        case "discover_area_shift_find":
            return f"Ranking{w} by how much it changed from {a.get('from_period')} to {a.get('to_period')}"
        case "discover_space_zoom":
            verb = {"in": "Zooming in on", "out": "Zooming out from", "around": "Looking around"}
            return f"{verb.get(a.get('direction', 'in'), 'Zooming in on')} {a.get('node')}"
        case "discover_time_zoom":
            return f"Time series for {a.get('node')}{t}"
        case "discover_node_compare":
            return f"Comparing {a.get('node_a')} with {a.get('node_b')}"
        case "toolbox_classifier_build":
            labels = ", ".join((a.get("examples") or {}).keys())
            return f"Building a classifier: {labels}"
        case "toolbox_classifier_apply":
            inside = f" inside {a['inside']}" if a.get("inside") else ""
            return f"Classifying every location{w}{inside} with {a.get('classifier')}"
        case "reasoning_imagery_check":
            n = len(a.get("nodes") or [])
            return f"Checking against the imagery: “{a.get('claim', '')}” at {n} location{'s' if n != 1 else ''}"
        case "user_alias_set":
            return f"Naming {a.get('node')} “{a.get('name')}”"
        case "user_identity_get":
            return "Checking who is signed in"
    return tool.replace("_", " ")


def verdict_state(verdict: str | None) -> str:
    if verdict is None:
        return "checking"
    v = verdict.lower()
    if v.startswith("not") or v == "unchecked":
        return "rejected" if v.startswith("not") else "checking"
    return "ok"


def summarize(tool: str, c: Any) -> str:
    """One line about what a move returned."""
    if not isinstance(c, dict):
        return "done"
    if tool == "reasoning_imagery_check":
        k = len(c.get("supported_at") or [])
        return f"{c.get('verdict', 'unchecked')} · {k} location{'s' if k != 1 else ''} supported"
    if tool == "discover_time_zoom":
        span = c.get("span") or []
        line = f"{c.get('steps', '?')} images"
        if len(span) == 2:
            line += f", {span[0]} → {span[1]}"
        br = c.get("break")
        if isinstance(br, dict):
            between = br.get("between") or [br.get("from"), br.get("to")]
            verdict = br.get("verdict") or ""
            if between and all(between):
                line += f" · changed between {between[0]} and {between[1]}"
            if verdict:
                line += f" ({verdict})"
        else:
            line += " · no lasting change found"
        return line
    if tool == "discover_node_compare" and "similarity" in c:
        return f"similarity {c['similarity']}"
    if tool == "toolbox_classifier_build":
        labels = c.get("labels")
        names = ", ".join(labels.keys()) if isinstance(labels, dict) else ""
        return f"{c.get('classifier', 'classifier')} · {names} · {c.get('method', '')}".strip(" ·")
    if tool == "toolbox_classifier_apply" and isinstance(c.get("counts"), dict):
        return " · ".join(f"{k} {v}" for k, v in c["counts"].items())
    table = first_table(c)
    if table:
        rows = records(table)
        n = len(rows)
        cols = set(table["columns"])
        what = "changes" if "change" in cols else "locations" if "node" in cols else "cells"
        line = f"{n} {what}"
        sims = [r["similarity"] for r in rows if isinstance(r.get("similarity"), (int, float))]
        if sims:
            line += f" · strongest {max(sims):.2f}"
        return line
    if c.get("note"):
        return str(c["note"])
    return "done"


def record_label(r: dict) -> str:
    head = r.get("name") or r.get("node") or r.get("change") or r.get("from") or ""
    bits = []
    if r.get("before") and r.get("after"):
        bits.append(f"{r['before']} → {r['after']}")
    elif r.get("observed"):
        bits.append(str(r["observed"]))
    elif r.get("period"):
        bits.append(str(r["period"]))
    if isinstance(r.get("similarity"), (int, float)):
        bits.append(f"similarity {r['similarity']:.2f}")
    if r.get("label"):
        bits.append(str(r["label"]))
    return f"<div class='geo-pop'><b>{esc(head)}</b>{esc(' · '.join(bits))}</div>"


# ----------------------------------------------------------------------------- the conversation


@dataclass
class Finding:
    n: int
    text: str
    kind: str
    verdict: str | None = None
    supported: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    @property
    def state(self) -> str:
        return verdict_state(self.verdict)

    def check_line(self) -> str:
        if self.verdict is None:
            return "checking…"
        k, m = len(self.supported), len(self.supported) + len(self.rejected)
        line = f"{self.verdict} · {k} of {m} location{'s' if m != 1 else ''}" if m else self.verdict
        shows = [f"{node.split('/')[-1]}: {what}" for node, what in self.rejected if what]
        if shows:
            line += " · rejected: " + "; ".join(shows[:3])
        return line


@dataclass
class Beat:
    says: str = ""
    thoughts: str = ""
    tool: str | None = None
    args: dict = field(default_factory=dict)
    line: str = ""
    result: str = ""
    error: str | None = None
    finding: Finding | None = None
    done: bool = False


@dataclass
class Turn:
    question: str
    beats: list[Beat] = field(default_factory=list)
    streaming: str = ""
    answer: str = ""
    error: str | None = None
    done: bool = False
    started: float = field(default_factory=time.time)
    seconds: float = 0.0


# ----------------------------------------------------------------------------- the map


def camera(points: list[tuple[float, float]]) -> tuple[tuple[float, float], int]:
    """A center and zoom that show every (lat, lon) point: the map's traits, set from Python."""
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    center = ((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2)
    span = max(max(lats) - min(lats), max(lons) - min(lons), 0.001)
    zoom = int(math.log2(1260 / span))  # about 3.5 tiles of 256 px across a 900 px map
    return center, max(3, min(13, zoom))



class LiveMap:
    """Satellite imagery with, on top: the area each move looked at (a faint trail), what the
    latest move found, and a numbered pin for every place a finding rests on."""

    def __init__(self, outline: dict | None, height: int):
        self.map = Map(
            center=(39.0, -98.0),
            zoom=4,
            basemap=basemaps.Esri.WorldImagery,
            scroll_wheel_zoom=True,
            layout=W.Layout(height=f"{height}px", width="100%"),
        )
        self.map.add(TileLayer(url=LABELS_TILES, opacity=0.85, name="labels"))
        self.trail = LayerGroup()
        self.view = LayerGroup()
        self.pins = LayerGroup()
        for g in (self.trail, self.view, self.pins):
            self.map.add(g)
        self.caption = W.HTML()
        self.caption.add_class("geo-caption-box")
        self.map.add(WidgetControl(widget=self.caption, position="bottomleft"))
        self.outline = outline
        self._draw_outline()
        self.say("Ask a question.", "")

    def _draw_outline(self):
        a = self.outline
        style = dict(color="white", weight=1, fill=False, opacity=0.7)
        if not isinstance(a, dict):
            return
        if "ring" in a:
            self.trail.add(Polygon(locations=[(p["lat"], p["lon"]) for p in a["ring"]], **style))
        elif "radius_km" in a:
            self.trail.add(Circle(location=(a["lat"], a["lon"]), radius=int(a["radius_km"] * 1000), **style))
        elif "south" in a:
            self.trail.add(Rectangle(bounds=[(a["south"], a["west"]), (a["north"], a["east"])], **style))

    def reset(self):
        for g in (self.trail, self.view, self.pins):
            g.clear_layers()
        self._draw_outline()
        self.map.center, self.map.zoom = (39.0, -98.0), 4
        self.say("Ask a question.", "")

    def say(self, line: str, result: str):
        legend = (
            f"<div class='geo-legend'><span><i style='background:{PLACE}'></i>match</span>"
            f"<span><i style='background:{CHANGE}'></i>change</span>"
            f"<span><i style='background:{FINDING}'></i>supported by the imagery</span>"
            f"<span><i style='background:{REJECTED}'></i>rejected</span></div>"
        )
        self.caption.value = (
            f"<div class='geo-caption'><div>{esc(line)}</div>"
            f"<div class='geo-cap-r'>{esc(result)}</div>{legend}</div>"
        )

    def lens(self, area: Any):
        """The area a move looks at: drawn, kept faintly, and the camera on it."""
        if not isinstance(area, dict) or area == self.outline:
            return
        style = dict(color=ACCENT, weight=1, fill=False, opacity=0.6, dash_array="4 4")
        if "ring" in area:
            pts = [(p["lat"], p["lon"]) for p in area["ring"]]
            self.trail.add(Polygon(locations=pts, **style))
            self.fit([(p[0], p[1]) for p in pts])
        elif "radius_km" in area:
            self.trail.add(Circle(location=(area["lat"], area["lon"]), radius=int(area["radius_km"] * 1000), **style))
            d = area["radius_km"] / 111
            self.fit([(area["lat"] - d, area["lon"] - d), (area["lat"] + d, area["lon"] + d)])
        elif "south" in area:
            self.trail.add(Rectangle(bounds=[(area["south"], area["west"]), (area["north"], area["east"])], **style))
            self.fit([(area["south"], area["west"]), (area["north"], area["east"])])

    def fit(self, points: list[tuple[float, float]]):
        if points:
            self.map.center, self.map.zoom = camera(points)

    def show(self, tool: str, rows: list[dict]):
        """What a move found, as the map's one view: dots coloured by what they are."""
        self.view.clear_layers()
        if not rows:
            return
        palette: dict[str, str] = {}
        for r in rows:
            if "change" in r:
                color = CHANGE
            elif tool.startswith("search"):
                color = PLACE
            elif r.get("label"):
                color = palette.setdefault(str(r["label"]), CLASSES[len(palette) % len(CLASSES)])
            else:
                color = CELL
            self.view.add(
                CircleMarker(
                    location=(r["lat"], r["lon"]), radius=6, color=color, fill_color=color,
                    fill_opacity=0.85, weight=2, popup=W.HTML(record_label(r)),
                )
            )
        self.fit([(r["lat"], r["lon"]) for r in rows])

    def pin(self, finding: Finding, node: str, at: tuple[float, float], rejected: bool, shows: str = ""):
        state = "rejected" if rejected else finding.state
        kind = " place" if finding.kind == "place" else ""
        icon = DivIcon(html=f"<div class='geo-pin {state}{kind}'>{finding.n}</div>", icon_size=[24, 24], icon_anchor=[12, 12])
        what = f"<div class='geo-pop-f'>{esc(node)}" + (f" · shows: {esc(shows)}" if shows else "") + "</div>"
        label = "Rejected" if rejected else (KIND_LABEL.get(finding.kind, "") if finding.verdict else "Checking")
        pop = W.HTML(f"<div class='geo-pop'><b>{finding.n} · {esc(label)}</b>{esc(finding.text)}{what}</div>")
        self.pins.add(Marker(location=at, icon=icon, draggable=False, popup=pop))


# ----------------------------------------------------------------------------- the app


class GeoChat:
    """A chat dock beside a live map, driving a Pydantic AI agent that holds LGND Geo's tools.

    Args:
        agent: The agent. Its MCP toolsets are hooked so that every `reasoning_imagery_check`
            result carries a `finding` number, which the chat and the map share.
        areas: Named areas, so a move over one is described by its name ("over California").
        examples: Questions offered as buttons before the first question.
        title: Shown in the dock's header.
        height: The app's height in pixels.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        areas: dict[str, Any] | None = None,
        examples: list[str] | None = None,
        title: str = "LGND Geo",
        model_name: str = "Claude Opus 5.5",
        height: int = 720,
    ):
        self.agent = agent
        self.areas = areas or {}
        self.turns: list[Turn] = []
        self.findings: list[Finding] = []
        self.history: list = []
        self._beats_by_call: dict[str, Beat] = {}
        self._seen: dict[str, dict] = {}
        self._count = 0
        self._busy = False
        self._task: asyncio.Task | None = None
        self._last_render = 0.0
        self._hook_toolsets()

        outline = next(iter(self.areas.values()), None)
        self.map = LiveMap(outline, height)

        logo = f"<img src='{_LOGO}' alt='LGND'/>" if _LOGO else ""
        head = W.HTML(f"<div class='geo-head'>{logo}<span class='geo-title'>{esc(title)}</span><span class='geo-model'>{esc(model_name)}</span></div>")
        self._new = W.Button(description="New chat", layout=W.Layout(width="auto"))
        self._new.add_class("geo-new")
        self._new.on_click(lambda _: self.new_chat())
        header = W.HBox([head, self._new], layout=W.Layout(align_items="center", padding="0 12px 0 0", border_bottom="1px solid rgba(255,255,255,.08)"))

        buttons = []
        for q in examples or []:
            b = W.Button(description=q, layout=W.Layout(width="100%"))
            b.add_class("geo-example")
            b.on_click(lambda _, q=q: self._submit(q))
            buttons.append(b)
        self._examples = W.VBox(
            [W.HTML("<div class='geo-ex-h'>Examples</div>"), *buttons],
            layout=W.Layout(padding="12px 14px 4px", display="flex" if buttons else "none"),
        )
        self._examples.add_class("geo-examples")

        self._transcript = W.HTML(layout=W.Layout(flex="1 1 0", min_height="0", overflow_y="auto"))
        self._transcript.add_class("geo-scroll")

        self._text = W.Text(placeholder="Ask about a location", continuous_update=False, layout=W.Layout(flex="1 1 auto"))
        self._text.observe(self._on_text, "value")
        self._ask = W.Button(description="Ask", layout=W.Layout(width="auto"))
        self._ask.add_class("geo-ask")
        self._ask.on_click(lambda _: self._submit(self._text.value))
        composer = W.HBox([self._text, self._ask], layout=W.Layout(padding="10px 12px 12px", align_items="center", border_top="1px solid rgba(255,255,255,.08)"))
        composer.add_class("geo-composer")

        dock = W.VBox([header, self._examples, self._transcript, composer], layout=W.Layout(width="440px", min_width="440px", height=f"{height}px"))
        dock.add_class("geo-dock")
        map_box = W.Box([self.map.map], layout=W.Layout(flex="1 1 0", min_width="0", height=f"{height}px", border_radius="14px", overflow="hidden"))
        self.widget = W.VBox(
            [W.HTML(CSS), W.HBox([map_box, dock], layout=W.Layout(width="100%", height=f"{height}px", align_items="stretch"))],
            layout=W.Layout(width="100%"),
        )
        self.widget.add_class("geo-app")
        self._render(force=True)

    # ---- numbering the findings: the imagery check's result tells the model its number
    def _hook_toolsets(self):
        async def hook(ctx, call_tool, name, args):
            result = await call_tool(name, args)
            if name == "reasoning_imagery_check" and isinstance(result, dict):
                self._count += 1
                result = {"finding": self._count, **result}
            return result

        for ts in self.agent.toolsets:
            if isinstance(ts, MCPToolset):
                ts.process_tool_call = hook

    def _ipython_display_(self):
        display(self.widget)

    # ---- asking
    def _on_text(self, change):
        q = (change["new"] or "").strip()
        if q:
            self._text.value = ""
            self._submit(q)

    def _submit(self, question: str):
        question = (question or "").strip()
        if not question or self._busy:
            return
        self._text.value = ""
        self._task = asyncio.ensure_future(self.ask(question))

    def new_chat(self):
        if self._task and not self._task.done():
            self._task.cancel()
        self._busy = False
        self.turns, self.findings, self.history = [], [], []
        self._beats_by_call, self._seen, self._count = {}, {}, 0
        self.map.reset()
        self._examples.layout.display = "flex" if len(self._examples.children) > 1 else "none"
        self._ask.disabled = False
        self._render(force=True)

    async def ask(self, question: str) -> Turn:
        """Ask the agent one question; the dock and the map follow it as it works."""
        while self._busy:
            await asyncio.sleep(0.1)
        self._busy = True
        self._ask.disabled = True
        self._examples.layout.display = "none"
        turn = Turn(question)
        self.turns.append(turn)
        self._render(force=True)
        try:
            async with self.agent.run_stream_events(question, message_history=self.history) as events:
                async for event in events:
                    self._on_event(turn, event)
        except asyncio.CancelledError:
            turn.error = "Stopped."
            raise
        except Exception as e:  # shown in the dock, like any other failed step
            turn.error = f"{type(e).__name__}: {e}"
        finally:
            turn.done = True
            turn.seconds = time.time() - turn.started
            if turn.beats and turn.beats[-1].tool is None and not turn.beats[-1].thoughts.strip():
                turn.beats.pop()
            if not turn.answer and turn.streaming.strip():
                turn.answer = turn.streaming
            turn.streaming = ""
            self._busy = False
            self._ask.disabled = False
            self._render(force=True)
        return turn

    # ---- the event stream, into the conversation and onto the map
    def _forming(self, turn: Turn) -> Beat:
        if turn.beats and turn.beats[-1].tool is None:
            return turn.beats[-1]
        beat = Beat()
        turn.beats.append(beat)
        return beat

    def _on_event(self, turn: Turn, event):
        match event:
            case PartStartEvent(part=ThinkingPart(content=text)):
                self._forming(turn).thoughts += text or ""
            case PartDeltaEvent(delta=ThinkingPartDelta(content_delta=text)):
                self._forming(turn).thoughts += text or ""
            case PartStartEvent(part=TextPart(content=text)):
                turn.streaming += text or ""
            case PartDeltaEvent(delta=TextPartDelta(content_delta=text)):
                turn.streaming += text or ""
            case FunctionToolCallEvent(part=call):
                beat = self._forming(turn)
                beat.tool = call.tool_name
                beat.args = call.args_as_dict()
                beat.says = turn.streaming.strip()
                turn.streaming = ""
                beat.line = describe(call.tool_name, beat.args, self.areas)
                self._beats_by_call[call.tool_call_id] = beat
                self.map.lens(beat.args.get("area"))
                self.map.say(beat.line, "working…")
                if call.tool_name == "reasoning_imagery_check":
                    self._start_finding(beat)
                self._render(force=True)
                return
            case FunctionToolResultEvent(part=part):
                beat = self._beats_by_call.get(part.tool_call_id) or self._forming(turn)
                beat.done = True
                if isinstance(part, RetryPromptPart):
                    beat.error = str(part.content)
                    if beat.finding:
                        beat.finding.verdict = "not supported"
                        beat.finding.rejected = [(n, beat.error) for n in beat.args.get("nodes") or []]
                    self.map.say(beat.line, "refused")
                else:
                    self._on_result(beat, part.content)
                self._render(force=True)
                return
            case AgentRunResultEvent(result=result):
                turn.answer = result.output if isinstance(result.output, str) else str(result.output)
                turn.streaming = ""
                self.history = result.all_messages()
                self._render(force=True)
                return
            case _:
                return
        self._render()

    def _start_finding(self, beat: Beat):
        self._count_pending = getattr(self, "_count_pending", 0)
        finding = Finding(n=0, text=str(beat.args.get("claim", "")), kind=str(beat.args.get("kind", "place")))
        beat.finding = finding

    def _on_result(self, beat: Beat, content: Any):
        beat.result = summarize(beat.tool or "", content)
        self.map.say(beat.line, beat.result)
        if isinstance(content, dict):
            for r in located(content):
                for key in ("node", "change"):
                    if isinstance(r.get(key), str):
                        self._seen[r[key]] = r
        if beat.tool == "reasoning_imagery_check" and beat.finding is not None and isinstance(content, dict):
            f = beat.finding
            f.n = int(content.get("finding") or (len(self.findings) + 1))
            f.verdict = str(content.get("verdict") or "unchecked")
            f.supported = [str(x) for x in content.get("supported_at") or []]
            cited = [str(x) for x in beat.args.get("nodes") or []]
            shows = {}
            for r in noded(content):
                if r.get("node") not in f.supported:
                    shows[str(r["node"])] = str(r.get("shows") or r.get("sees") or r.get("reason") or "")
            for node in cited:
                if node not in f.supported and node not in shows:
                    shows[node] = ""
            f.rejected = list(shows.items())
            if f not in self.findings:
                self.findings.append(f)
            for node in f.supported:
                at = self._position(node)
                if at:
                    self.map.pin(f, node, at, rejected=False)
            for node, what in f.rejected:
                at = self._position(node)
                if at:
                    self.map.pin(f, node, at, rejected=True, shows=what)
            spots = [p for p in (self._position(n) for n in f.supported + [n for n, _ in f.rejected]) if p]
            if spots:
                self.map.fit(spots)
            return
        if isinstance(content, dict):
            rows = list(located(content))
            if rows:
                self.map.show(beat.tool or "", rows)

    def _position(self, node: str) -> tuple[float, float] | None:
        r = self._seen.get(node)
        if r is None and ">" in node:
            left, right = node.split(">", 1)
            prefix = left.split("/")[0]
            r = self._seen.get(left) or self._seen.get(f"{prefix}/{right}")
        if r is None:
            return None
        return (float(r["lat"]), float(r["lon"]))

    # ---- drawing the dock
    def _render(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_render < 0.08:
            return
        self._last_render = now
        self._transcript.value = f"<div class='geo-turns'>{''.join(self._turn_html(t) for t in self.turns)}</div>"

    def _turn_html(self, turn: Turn) -> str:
        out = [f"<article class='geo-turn'><div class='geo-q'>{esc(turn.question)}</div>"]
        beats = list(turn.beats)
        live_text = turn.streaming.strip() if not turn.done else ""
        if not turn.done and (not beats or beats[-1].tool is not None):
            beats.append(Beat())  # the beat forming now
        if beats:
            out.append("<ol class='geo-rail'>")
            for i, b in enumerate(beats):
                live = not turn.done and i == len(beats) - 1 and b.tool is None
                out.append(self._beat_html(b, live, live_text if live else ""))
            out.append("</ol>")
        if turn.error:
            out.append(f"<div class='geo-turn-err'>{esc(turn.error)}</div>")
        if turn.done and turn.answer:
            out.append(f"<div class='geo-answer'>{self._answer_html(turn.answer)}</div>")
        if turn.done:
            moves = sum(1 for b in turn.beats if b.tool)
            out.append(f"<div class='geo-timing'>{moves} move{'s' if moves != 1 else ''} · {turn.seconds:.0f} s</div>")
        out.append("</article>")
        return "".join(out)

    def _beat_html(self, b: Beat, live: bool, live_text: str) -> str:
        state = "live" if live else "error" if b.error else "found" if (b.finding and b.finding.state == "ok") else "done" if b.done else "running" if b.tool else "live"
        parts = []
        says = b.says or live_text
        if says:
            parts.append(f"<p class='geo-says'>{esc(says)}</p>")
        if b.thoughts.strip():
            parts.append(f"<details class='geo-think'{' open' if live else ''}><summary>Reasoning</summary><div>{esc(b.thoughts.strip())}</div></details>")
        if live and not says and not b.thoughts.strip():
            parts.append("<div class='geo-think'>Thinking <span class='geo-dots'><i></i><i></i><i></i></span></div>")
        if b.tool:
            res = f"<div class='geo-err'>{esc(b.error)}</div>" if b.error else f"<div class='geo-result'>{esc(b.result) if b.done else 'working…'}</div>"
            parts.append(f"<div class='geo-did'><div class='geo-line'>{esc(b.line)}</div>{res}</div>")
        if b.finding:
            f = b.finding
            kind = KIND_LABEL.get(f.kind, "")
            label = {"ok": kind or "Supported", "rejected": "Not supported", "checking": "Checking"}[f.state]
            n = f.n or "·"
            parts.append(
                f"<div class='geo-finding {f.state} {esc(f.kind)}'><span class='geo-fn'>{n}</span>"
                f"<span><span class='geo-fk'>{esc(label)}</span>{esc(f.text)}<span class='geo-fc'>{esc(f.check_line())}</span></span></div>"
            )
        return f"<li class='geo-beat {state}'><span class='geo-node'><i></i></span><div class='geo-card'>{''.join(parts)}</div></li>"

    def _answer_html(self, text: str) -> str:
        rendered = _MARKDOWN(text)
        states = {f.n: f.state for f in self.findings}

        import re

        def cite(m):
            n = int(m.group(1))
            if n not in states:
                return m.group(0)
            return f"<span class='geo-cite {states[n]}'>{n}</span>"

        return re.sub(r"\[(\d+)\]", cite, rendered)


_MARKDOWN = mistune.create_markdown(plugins=["table", "strikethrough"], escape=False)
