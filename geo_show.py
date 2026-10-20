"""Small helpers for the workshop notebook: a tool result on a map, a tool's description as
Markdown, and an agent's run printed as it happens. The app at the end of the
notebook is `geo_demo.py`."""

from __future__ import annotations

import base64
import html
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.parse
import urllib.request
import ipywidgets as W
import pandas as pd
from IPython.display import HTML, Markdown, display
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

from geo_demo import _MARKDOWN, CHANGE, FINDING, LABELS_TILES, PLACE, camera, describe, located, prefetch, record_label, summarize, thumbnail

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

RUN_CSS = """
<style>
.geo-run { font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1e293b; max-width: 920px; }
.geo-run .think { color: #64748b; font-style: italic; margin: 6px 0; white-space: pre-wrap; }
.geo-run .say { margin: 6px 0; white-space: pre-wrap; }
.geo-run .call { margin: 8px 0; padding: 8px 12px; border: 1px solid #e2e8f0; border-left: 3px solid #94a3b8; border-radius: 6px; background: #f8fafc; }
.geo-run .call.done { border-left-color: #0ea5e9; }
.geo-run .call.ok { border-left-color: #10b981; }
.geo-run .call.error { border-left-color: #ef4444; }
.geo-run .call .line { font-weight: 600; }
.geo-run .call .tool { color: #64748b; font: 12px ui-monospace, Menlo, monospace; margin-left: 8px; }
.geo-run .call .result { color: #475569; font-size: 13px; margin-top: 2px; }
.geo-run .call .result.err { color: #b91c1c; white-space: pre-wrap; }
.geo-run .call details { margin-top: 4px; }
.geo-run .call summary { color: #64748b; font-size: 12px; cursor: pointer; }
.geo-run .call pre { margin: 4px 0 0; padding: 8px; background: #f1f5f9; border-radius: 4px; font: 12px/1.45 ui-monospace, Menlo, monospace; white-space: pre-wrap; }
.geo-run .answer { margin-top: 12px; padding: 12px 14px; border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; }
.geo-run .answer table { border-collapse: collapse; font-size: 13px; margin: 8px 0; }
.geo-run .answer td, .geo-run .answer th { border: 1px solid #e2e8f0; padding: 3px 8px; text-align: left; }
.geo-run .answer p { margin: 0 0 8px; } .geo-run .answer p:last-child { margin: 0; }
.geo-run .working { color: #94a3b8; font-size: 13px; }
</style>
"""


def _run_html(blocks: list[dict], answer: str, done: bool) -> str:
    parts = ["<div class='geo-run'>"]
    for b in blocks:
        if b["kind"] == "think":
            parts.append(f"<div class='think'>{html.escape(b['text'].strip())}</div>")
        elif b["kind"] == "say":
            parts.append(f"<div class='say'>{html.escape(b['text'].strip())}</div>")
        else:
            state = "error" if b["error"] else "ok" if b.get("supported") else "done" if b["done"] else ""
            args = html.escape(json.dumps(b["args"], indent=1, ensure_ascii=False))
            result = (
                f"<div class='result err'>✗ {html.escape(b['error'])}</div>" if b["error"]
                else f"<div class='result'>✓ {html.escape(b['result'])}</div>" if b["done"]
                else "<div class='result working'>running…</div>"
            )
            parts.append(
                f"<div class='call {state}'><span class='line'>{html.escape(b['line'])}</span>"
                f"<span class='tool'>{html.escape(b['tool'])}</span>{result}"
                f"<details><summary>arguments</summary><pre>{args}</pre></details></div>"
            )
    if answer.strip():
        parts.append(f"<div class='answer'>{_MARKDOWN(answer)}</div>")
    elif not done:
        parts.append("<div class='working'>thinking…</div>")
    parts.append("</div>")
    return RUN_CSS + "".join(parts)


async def stream(agent: Agent, question: str, *, message_history=None):
    """Run `agent` on `question`, showing the run as it happens: thinking, what the model says, each
    tool call in plain words with its arguments and result, and the answer rendered as Markdown.
    Returns the run's result, whose `all_messages()` is the history for the next question."""
    out = W.HTML()
    display(out)
    blocks: list[dict] = []
    by_call: dict[str, dict] = {}
    text, answer, result = "", "", None
    last = 0.0

    def render(force=False):
        nonlocal last
        if force or time.time() - last > 0.1:
            last = time.time()
            out.value = _run_html(blocks, answer, result is not None)

    def flush_text():
        nonlocal text
        if text.strip():
            blocks.append({"kind": "say", "text": text})
        text = ""

    async with agent.run_stream_events(question, message_history=message_history or []) as events:
        async for event in events:
            match event:
                case PartStartEvent(part=ThinkingPart(content=t)) | PartDeltaEvent(delta=ThinkingPartDelta(content_delta=t)):
                    if blocks and blocks[-1]["kind"] == "think":
                        blocks[-1]["text"] += t or ""
                    else:
                        blocks.append({"kind": "think", "text": t or ""})
                case PartStartEvent(part=TextPart(content=t)) | PartDeltaEvent(delta=TextPartDelta(content_delta=t)):
                    text += t or ""
                case FunctionToolCallEvent(part=call):
                    flush_text()
                    args = call.args_as_dict()
                    block = {"kind": "call", "tool": call.tool_name, "args": args, "line": describe(call.tool_name, args, {}),
                             "result": "", "error": None, "done": False, "supported": False}
                    blocks.append(block)
                    by_call[call.tool_call_id] = block
                case FunctionToolResultEvent(part=part):
                    block = by_call.get(part.tool_call_id)
                    if block:
                        block["done"] = True
                        if isinstance(part, RetryPromptPart):
                            block["error"] = str(part.content)
                        else:
                            block["result"] = summarize(part.tool_name, part.content)
                            block["supported"] = isinstance(part.content, dict) and bool(part.content.get("supported_at"))
                case AgentRunResultEvent(result=r):
                    result = r
                    answer = r.output if isinstance(r.output, str) else str(r.output)
                    text = ""
            render()
    render(force=True)
    return result


# ----------------------------------------------------------------------------- chip images


def _tile(node: str, caption: str, color: str | None, size: int) -> str:
    image = thumbnail(node)
    if image:
        data, media_type = image
        picture = f"<img src='data:{media_type};base64,{base64.b64encode(data).decode()}' style='width:{size}px;height:{size}px;object-fit:cover;display:block'>"
    else:
        picture = f"<div style='width:{size}px;height:{size}px;display:flex;align-items:center;justify-content:center;background:#1e293b;color:#94a3b8;font-size:12px'>no imagery</div>"
    border = f"border:3px solid {color};" if color else "border:3px solid transparent;"
    return (
        f"<figure style='margin:0;width:{size}px'>"
        f"<div style='{border}border-radius:6px;overflow:hidden'>{picture}</div>"
        f"<figcaption style='font:12px/1.4 monospace;color:#475569;margin-top:4px;word-break:break-all'>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def _row(tiles: list[str]) -> HTML:
    return HTML(f"<div style='display:flex;flex-wrap:wrap;gap:12px;margin:8px 0'>{''.join(tiles)}</div>")


def thumbnails(rows, verified=(), size: int = 180) -> HTML:
    """The images of the observations in `rows` (a DataFrame with `node` and `observed`, or a list
    of node ids), in a row: green frame for nodes in `verified`, red for the rest."""
    verified = set(verified)
    records = rows.to_dict("records") if hasattr(rows, "to_dict") else [{"node": n} for n in rows]
    prefetch([r["node"] for r in records])
    tiles = []
    for r in records:
        node = r["node"]
        caption = node.split("/")[-1] + (f" · {r['observed']}" if r.get("observed") else "")
        color = FINDING if node in verified else (REJECTED if verified else None)
        tiles.append(_tile(node, caption, color, size))
    return _row(tiles)


def before_after(change, size: int = 260) -> HTML:
    """The two images of a change: `change` is a row with `change`, `before` and `after` (as
    `search_change_find` returns them), or a change id like `naip/obs:12>obs:34`."""
    if isinstance(change, str):
        change_id, before_date, after_date = change, "before", "after"
    else:
        change_id, before_date, after_date = change["change"], change["before"], change["after"]
    first, second = change_id.split(">")
    index = first.split("/")[0]
    before_node, after_node = first, f"{index}/{second}"
    prefetch([before_node, after_node])
    return _row(
        [
            _tile(before_node, f"before · {before_date}", None, size),
            _tile(after_node, f"after · {after_date}", None, size),
        ]
    )
