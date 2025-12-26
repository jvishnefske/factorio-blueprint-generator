# Design Document: Factorio Blueprint Optimizer

## Overview

This project optimizes Factorio factory layouts by arranging machines to minimize the overall bounding box footprint using simulated annealing.

## MVP Functional Requirements

- [x] FR-001: Load entity type definitions (size, crafting speed) from JSON
- [x] FR-002: Load recipe definitions (ingredients, products, energy) from JSON
- [x] FR-003: Calculate required machine count for target production rate
- [x] FR-004: Detect entity overlap using bounding box collision
- [x] FR-005: Validate blueprint state has no overlapping entities
- [x] FR-006: Calculate bounding box footprint for fitness evaluation
- [x] FR-007: Generate random neighbor states for optimization
- [x] FR-008: Implement simulated annealing optimization loop
- [x] FR-009: Export optimized layout as Factorio blueprint string

## Architecture

### Data Flow

```
JSON Files -> GameDataLoader -> ProductionCalculator -> BlueprintState
                                                              |
                                                              v
                                          SimulatedAnnealingOptimizer
                                                              |
                                                              v
                                                    BlueprintExporter
```

### Core Modules

1. **GameDataLoader**: Parses entity and recipe JSON files into typed models
2. **ProductionCalculator**: Computes required machines for target production
3. **StateManager**: Manages entity bounds and collision detection
4. **FitnessEvaluator**: Scores layouts based on footprint area
5. **SimulatedAnnealingOptimizer**: Iteratively improves layout placement
6. **BlueprintExporter**: Converts state to Factorio blueprint string

### Data Models

- **Position**: 2D coordinates (x, y)
- **Size**: Dimensions (width, height)
- **EntityType**: Entity definition with size, crafting speed, etc.
- **Recipe**: Production recipe with ingredients and products
- **EntityInstance**: Placed entity with position and configuration
- **BlueprintState**: Complete layout state being optimized

## Constraints

- Entities must not overlap (collision detection required)
- Entity positions snap to 0.5 grid alignment (Factorio standard)
- All entities require valid positions before export

## Future Enhancements

- Logistics integration (belts, inserters, pipes)
- Multi-objective optimization (cost, power, throughput)
- Recursive production chain calculation
- Module and beacon placement optimization
