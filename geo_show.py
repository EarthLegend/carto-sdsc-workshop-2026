"""Small helpers for the workshop notebook: a tool result on a map, a tool's description as
Markdown, and an agent's run printed as it happens. The app at the end of the
notebook is `geo_demo.py`."""

from __future__ import annotations

import ipywidgets as W
import pandas as pd
from IPython.display import Markdown
from ipyleaflet import Circle, CircleMarker, Map, Polygon, Rectangle, TileLayer, basemaps
from pydantic_ai import Agent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.run import AgentRunResultEvent

from geo_demo import CHANGE, FINDING, LABELS_TILES, PLACE, camera, located, record_label, summarize

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)

DIM, RESET = "\x1b[2m", "\x1b[0m"


def _first_sentence(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    head, dot, _ = line.partition(". ")
    return head + ("." if dot else "")


def _cell(text: str) -> str:
    """Text safe inside a Markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


# ----------------------------------------------------------------------------- tables


def describe_tool(tool) -> Markdown:
    """One tool as the model sees it: its description and a table of its arguments."""
    schema = tool.input_schema
    required = set(schema.get("required") or [])
    lines = [f"### `{tool.name}`", "", tool.description or "", "", "| argument | | description |", "|---|---|---|"]
    for name, spec in (schema.get("properties") or {}).items():
        lines.append(f"| `{name}` | {'required' if name in required else ''} | {_cell(spec.get('description', ''))} |")
    return Markdown("\n".join(lines))


# ----------------------------------------------------------------------------- maps


def show_map(result: dict | None = None, *, area: dict | None = None, highlight=(), height: int = 440) -> Map:
    """A tool result on satellite imagery: every row with a position as a dot (amber for a place,
    rose for a change, green for a node in `highlight`), and `area` outlined, centered on the dots."""
    m = Map(
        center=(39.0, -98.0), zoom=4, basemap=basemaps.Esri.WorldImagery, scroll_wheel_zoom=True,
        layout=W.Layout(height=f"{height}px", width="100%"),
    )
    m.add(TileLayer(url=LABELS_TILES, opacity=0.85))
    outline = dict(color="white", weight=1, fill=False)
    if area and "ring" in area:
        m.add(Polygon(locations=[(p["lat"], p["lon"]) for p in area["ring"]], **outline))
    elif area and "radius_km" in area:
        m.add(Circle(location=(area["lat"], area["lon"]), radius=int(area["radius_km"] * 1000), **outline))
    elif area and "south" in area:
        m.add(Rectangle(bounds=[(area["south"], area["west"]), (area["north"], area["east"])], **outline))
    highlight = set(highlight)
    points = []
    for r in located(result) if result else []:
        node = r.get("node") or r.get("change")
        hit = node in highlight
        color = FINDING if hit else CHANGE if "change" in r else PLACE
        m.add(CircleMarker(location=(r["lat"], r["lon"]), radius=8 if hit else 6, color=color, fill_color=color, fill_opacity=0.85, weight=2, popup=W.HTML(record_label(r))))
        points.append((r["lat"], r["lon"]))
    if points:
        m.center, m.zoom = camera(points)
    return m


# ----------------------------------------------------------------------------- an agent's run


async def stream(agent: Agent, question: str, *, message_history=None):
    """Run `agent` on `question`, printing its thinking (dim), each tool call and what came back,
    then the answer as it is written. Returns the run's result, whose `all_messages()` is the
    history to pass to the next question."""
    result = None
    async with agent.run_stream_events(question, message_history=message_history or []) as events:
        async for event in events:
            match event:
                case FunctionToolCallEvent(part=call):
                    print(f"\n→ {call.tool_name} {call.args_as_json_str()[:200]}", flush=True)
                case FunctionToolResultEvent(part=part):
                    if isinstance(part, RetryPromptPart):
                        print(f"  ✗ {part.content}", flush=True)
                    else:
                        print(f"  ✓ {summarize(part.tool_name, part.content)}", flush=True)
                case PartStartEvent(part=ThinkingPart(content=text)):
                    print(f"\n{DIM}{text}", end="", flush=True)
                case PartDeltaEvent(delta=ThinkingPartDelta(content_delta=text)):
                    print(text or "", end="", flush=True)
                case PartEndEvent(part=ThinkingPart()):
                    print(RESET, flush=True)
                case PartStartEvent(part=TextPart(content=text)):
                    print(f"\n{text}", end="", flush=True)
                case PartDeltaEvent(delta=TextPartDelta(content_delta=text)):
                    print(text, end="", flush=True)
                case AgentRunResultEvent(result=r):
                    result = r
    print()
    return result
