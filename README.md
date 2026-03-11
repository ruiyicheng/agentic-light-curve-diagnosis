# Light Curve Agent

AI-powered astronomical light curve analysis using LangChain and DeepAgents.

## Installation

```bash
pip install -e .
```

For optional CUDA-accelerated period searches:

```bash
pip install -e ".[gpu]"
```

## Usage

```bash
python run_agent.py ./data/example_lc.csv
python run_agent.py ./data/example_lc.csv --source-name "RR Lyr"
```

The agent now inspects the file head to infer headers, supports multi-band light curves when a `filter`/`band`-like column is present, queries VSX before running deeper diagnosis, and fetches Gaia astrometry when source metadata is available.

## Structure

- `src/lightcurve_agent/` - Main package
  - `core/` - Domain logic (no external service deps)
  - `interfaces/` - External service abstractions
  - `tools/` - LangChain tool definitions

## Testing

```bash
pytest tests/
```
