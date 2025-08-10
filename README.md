# Factorio Blueprint Optimizer ⚙️ (Work in Progress 🚧)

![Work in Progress](https://img.shields.io/badge/Status-Work%20in%20Progress-red)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Factorio](https://img.shields.io/badge/Factorio-Optimized-orange)

This project aims to develop an optimizer for Factorio factory layouts using a simulated annealing approach.

**WARNING: Current blueprints are not functional!**
At this stage, the generated blueprints are for demonstrating the core optimization loop (placement, overlap detection, footprint evaluation). They *do not* include any logistics (belts, inserters, pipes) and therefore will not function as a complete factory in Factorio. The next steps in development will focus on integrating logistic components and flow validation.

## How to Run

To run the main optimization script, use `uv run`:

```bash
uv run python main.py
```

This will:
1.  Load dummy game data (entities, recipes).
2.  Calculate the required number of machines for a target production.
3.  Randomly place these machines and then attempt to optimize their positions to minimize footprint and total production cost (currently a placeholder cost).
4.  Export the resulting layout as a Factorio blueprint string (if `factorio-draftsman` is installed).

You can inspect the `main.py` file to see the hardcoded target production (e.g., `iron-gear-wheel`).

## Tests

To run the tests, use `uv run pytest`:

```bash
uv run pytest
```
