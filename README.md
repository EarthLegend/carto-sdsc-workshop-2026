# Large Earth Observation Models in Action

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EarthLegend/carto-sdsc-workshop-2026/blob/main/LGND_Geo_Agent_Demo.ipynb)

Workshop notebook: Spatial Data Science Conference, New York, October 20, 2026. Noam Rosenthal, LGND.

LGND Geo is an index of geospatial embeddings over NAIP and Sentinel-2 imagery, served as a set of
tools over MCP at `https://geo.lgnd.ai/mcp`. Any MCP client can connect to it, and
[`lgnd-geo-sdk`](https://github.com/EarthLegend/lgnd-geo-sdk) handles sign-in and the connection
from Python. The notebook works one example, new housing on former farmland across the US:

1. **Connect**
2. **Scan the country** for a feature of interest, and verify the matches against the imagery
3. **Change detection**: where it happened, when, and from what
4. **Export** to CARTO, your warehouse, or a desktop GIS
5. **Hand the workflow to an LLM agent** (Pydantic AI): run it, watch it work, add instructions, continue a conversation
6. **An application**: a chat panel beside a live map

Files: `LGND_Geo_Agent_Demo.ipynb` (the notebook), `geo_show.py` (map, tool-description and
streaming helpers), `geo_demo.py` (the application in section 6: ipywidgets
and ipyleaflet, no server), `assets/` (the logo).

## What you need

- An **LGND Geo API key**. Sign up at the [developer portal](https://geo.lgnd.ai/portal), click
  **Create API key**, and export it as `LGND_GEO_API_KEY`. The first cell asks for it if it is
  not set.
- An **Anthropic API key** exported as `ANTHROPIC_API_KEY`, for sections 5 and 6. Workshop
  attendees can get one at [https://bit.ly/4ia4Zew](https://bit.ly/4ia4Zew). The first agent cell
  asks for it if it is missing.
- **[uv](https://docs.astral.sh/uv/)**. The SDK needs Python 3.13, which uv fetches.

## Run it

**In Colab:** click the badge above. The first cell installs the packages and fetches the helper files; the notebook then asks for your keys.

**Locally**, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -m ipykernel install --user --name lgnd-geo-demo --display-name "LGND Geo demo"
export LGND_GEO_API_KEY=...      # from https://geo.lgnd.ai/portal
export ANTHROPIC_API_KEY=sk-ant-...
uv run jupyter notebook LGND_Geo_Agent_Demo.ipynb
```

The notebook opens on the `LGND Geo demo` kernel. Run it top to bottom. Sections 1 to 4 need only
`LGND_GEO_API_KEY`; sections 5 and 6 also need `ANTHROPIC_API_KEY`.

## Environment variables

| Variable | Description |
|---|---|
| `LGND_GEO_API_KEY` | Your LGND Geo API key, from the [developer portal](https://geo.lgnd.ai/portal) |
| `ANTHROPIC_API_KEY` | Your Anthropic API key, for the agent sections |
| `LGND_GEO_MCP_URL` | Optional: another LGND Geo MCP server than `https://geo.lgnd.ai/mcp` |
