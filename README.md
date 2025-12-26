# Factorio Blueprint Optimizer

Optimize Factorio factory layouts using simulated annealing. Given a target production rate, this tool calculates the required machines and arranges them to minimize the overall footprint.

**Note:** Current output is for placement demonstration only - logistics (belts, inserters) are not yet included.

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run optimizer
uv run python main.py
```

## How It Works

1. Define target production (e.g., 5 iron gears/second)
2. Calculator determines required machines
3. Simulated annealing optimizes placement to minimize bounding box
4. Export as Factorio blueprint string

## Requirements

- Python 3.12+
- factorio-draftsman (for blueprint export)
