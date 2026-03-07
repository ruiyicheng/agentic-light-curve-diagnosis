# Light Curve Agent

AI-powered astronomical light curve analysis using LangChain and DeepAgents.

## Installation

```bash
pip install -e .
```

## Usage

```bash
python run_agent.py ./data/example_lc.csv
```

## Structure

- `src/lightcurve_agent/` - Main package
  - `core/` - Domain logic (no external service deps)
  - `interfaces/` - External service abstractions
  - `tools/` - LangChain tool definitions

## Testing

```bash
pytest tests/
```
