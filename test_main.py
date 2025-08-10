import main
import pytest
import json
from pathlib import Path

from main import (
    GameDataLoader,
    StateManager,
    ProductionCalculator,
    FitnessEvaluator,
    EntityType,
    Recipe,
    ItemID,
    EntityTypeID,
    RecipeID,
    Position,
    Size,
    EntityInstance,
    BlueprintState
)

# Dummy data for testing
@pytest.fixture
def dummy_entity_data():
    return {
        "assembling-machine-2": {
            "type_id": "assembling-machine-2",
            "size": {"width": 3, "height": 3},
            "crafting_speed": 0.75,
            "power_consumption_kw": 200,
            "module_slots": 2
        },
        "substation": {
            "type_id": "substation",
            "size": {"width": 2, "height": 2},
            "power_consumption_kw": 10,
            "supply_area": {"width": 18, "height": 18}
        },
        "iron-furnace": {
            "type_id": "iron-furnace",
            "size": {"width": 2, "height": 2},
            "crafting_speed": 1.0,
            "power_consumption_kw": 90
        }
    }

@pytest.fixture
def dummy_recipe_data():
    return {
        "iron-gear-wheel": {
            "recipe_id": "iron-gear-wheel",
            "ingredients": [{"item_id": "iron-plate", "amount": 2}],
            "products": [{"item_id": "iron-gear-wheel", "amount": 1}],
            "energy_required_seconds": 0.5
        },
        "iron-plate": {
            "recipe_id": "iron-plate",
            "ingredients": [{"item_id": "iron-ore", "amount": 1}],
            "products": [{"item_id": "iron-plate", "amount": 1}],
            "energy_required_seconds": 3.2
        }
    }

@pytest.fixture
def create_dummy_data_files(tmp_path, dummy_entity_data, dummy_recipe_data):
    entity_path = tmp_path / "entities.json"
    recipe_path = tmp_path / "recipes.json"
    
    with open(entity_path, "w") as f:
        json.dump(dummy_entity_data, f)
    with open(recipe_path, "w") as f:
        json.dump(dummy_recipe_data, f)
        
    return entity_path, recipe_path

# --- Tests for GameDataLoader ---
def test_game_data_loader_loads_correctly(create_dummy_data_files):
    entity_path, recipe_path = create_dummy_data_files
    entity_types, recipes = GameDataLoader.load_from_json(str(entity_path), str(recipe_path))

    assert "assembling-machine-2" in entity_types
    assert entity_types["assembling-machine-2"].type_id == "assembling-machine-2"
    assert entity_types["assembling-machine-2"].size.width == 3

    assert "iron-gear-wheel" in recipes
    assert recipes["iron-gear-wheel"].recipe_id == "iron-gear-wheel"
    assert recipes["iron-gear-wheel"].energy_required_seconds == 0.5
    assert recipes["iron-gear-wheel"].ingredients[0].item_id == "iron-plate"

# --- Tests for StateManager ---
@pytest.fixture
def state_manager(dummy_entity_data):
    # Convert dummy_entity_data to actual EntityType objects
    entity_definitions = {k: EntityType(**v) for k, v in dummy_entity_data.items()}
    return StateManager(entity_definitions)

def test_get_entity_bounds_single_entity(state_manager):
    entity_type_id = EntityTypeID("assembling-machine-2")
    entity_instance = EntityInstance(
        entity_number=1,
        entity_type_id=entity_type_id,
        position=Position(x=0.0, y=0.0)
    )
    # assembler-2 is 3x3, centered at (0,0)
    # x_min = 0 - 1.5 = -1.5
    # x_max = 0 + 1.5 = 1.5
    # y_min = 0 - 1.5 = -1.5
    # y_max = 0 + 1.5 = 1.5
    x_min, x_max, y_min, y_max = state_manager.get_entity_bounds(entity_instance)
    assert (x_min, x_max, y_min, y_max) == (-1.5, 1.5, -1.5, 1.5)

    entity_instance_offset = EntityInstance(
        entity_number=2,
        entity_type_id=entity_type_id,
        position=Position(x=5.0, y=5.0)
    )
    x_min, x_max, y_min, y_max = state_manager.get_entity_bounds(entity_instance_offset)
    assert (x_min, x_max, y_min, y_max) == (3.5, 6.5, 3.5, 6.5)

def test_get_entity_bounds_substation(state_manager):
    entity_type_id = EntityTypeID("substation")
    substation_instance = EntityInstance(
        entity_number=1,
        entity_type_id=entity_type_id,
        position=Position(x=0.0, y=0.0)
    )
    # substation is 2x2, centered at (0,0)
    x_min, x_max, y_min, y_max = state_manager.get_entity_bounds(substation_instance)
    assert (x_min, x_max, y_min, y_max) == (-1.0, 1.0, -1.0, 1.0)

def test_do_entities_overlap_overlap(state_manager):
    entity1 = EntityInstance(entity_number=1, entity_type_id=EntityTypeID("assembling-machine-2"), position=Position(x=0.0, y=0.0))
    entity2 = EntityInstance(entity_number=2, entity_type_id=EntityTypeID("substation"), position=Position(x=0.0, y=0.0)) # Directly overlapping
    assert state_manager.do_entities_overlap(entity1, entity2)

    entity3 = EntityInstance(entity_number=3, entity_type_id=EntityTypeID("assembling-machine-2"), position=Position(x=1.0, y=1.0)) # Partial overlap
    assert state_manager.do_entities_overlap(entity1, entity3)

def test_do_entities_overlap_no_overlap(state_manager):
    entity1 = EntityInstance(entity_number=1, entity_type_id=EntityTypeID("assembling-machine-2"), position=Position(x=0.0, y=0.0))
    entity2 = EntityInstance(entity_number=2, entity_type_id=EntityTypeID("substation"), position=Position(x=10.0, y=10.0)) # Far apart
    assert not state_manager.do_entities_overlap(entity1, entity2)

    entity3 = EntityInstance(entity_number=3, entity_type_id=EntityTypeID("assembling-machine-2"), position=Position(x=3.0, y=0.0)) # Adjacent on X-axis (3x3 entity)
    # e1 bounds: (-1.5, 1.5, -1.5, 1.5)
    # e3 bounds: (1.5, 4.5, -1.5, 1.5)
    # They touch at x=1.5, but do not strictly overlap by the logic used (e1_x_max <= e2_x_min or e1_x_min >= e2_x_max)
    assert not state_manager.do_entities_overlap(entity1, entity3) 
    
    entity4 = EntityInstance(entity_number=4, entity_type_id=EntityTypeID("assembling-machine-2"), position=Position(x=0.0, y=3.0)) # Adjacent on Y-axis
    assert not state_manager.do_entities_overlap(entity1, entity4)


def test_is_state_valid_no_entities(state_manager):
    state = BlueprintState(entities=[])
    assert state_manager.is_state_valid(state)

def test_is_state_valid_no_overlap(state_manager):
    entity1 = EntityInstance(entity_number=1, entity_type_id=EntityTypeID("assembling-machine-2"), position=Position(x=0.0, y=0.0))
    entity2 = EntityInstance(entity_number=2, entity_type_id=EntityTypeID("substation"), position=Position(x=10.0, y=10.0))
    state = BlueprintState(entities=[entity1, entity2])
    assert state_manager.is_state_valid(state)

def test_is_state_valid_with_overlap(state_manager):
    entity1 = EntityInstance(entity_number=1, entity_type_id=EntityTypeID("assembling-machine-2"), position=Position(x=0.0, y=0.0))
    entity2 = EntityInstance(entity_number=2, entity_type_id=EntityTypeID("substation"), position=Position(x=0.5, y=0.5)) # Overlapping
    state = BlueprintState(entities=[entity1, entity2])
    assert not state_manager.is_state_valid(state)

# --- Tests for ProductionCalculator ---
@pytest.fixture
def production_calculator(dummy_entity_data, dummy_recipe_data):
    entity_types = {k: EntityType(**v) for k, v in dummy_entity_data.items()}
    recipes = {k: Recipe(**v) for k, v in dummy_recipe_data.items()}
    return ProductionCalculator(entity_types, recipes)

def test_calculate_requirements_single_item(production_calculator):
    target = {ItemID("iron-gear-wheel"): 5.0} # 5 iron gears per second
    # iron-gear-wheel: products=1, energy_required=0.5
    # assembler-2: crafting_speed=0.75
    # items_per_second_per_machine = (1 / 0.5) * 0.75 = 2 * 0.75 = 1.5
    # machines_needed = ceil(5.0 / 1.5) = ceil(3.33) = 4
    
    state = production_calculator.calculate_requirements(target)
    
    # Expected: 4 assembling-machine-2 for iron-gear-wheel + 1 substation
    assert len(state.entities) == 5 
    assemblers = [e for e in state.entities if e.entity_type_id == "assembling-machine-2"]
    substations = [e for e in state.entities if e.entity_type_id == "substation"]
    
    assert len(assemblers) == 4
    assert all(e.recipe_id == "iron-gear-wheel" for e in assemblers)
    assert len(substations) == 1
    assert substations[0].entity_type_id == "substation"

def test_calculate_requirements_no_recipe_found(production_calculator):
    target = {ItemID("non-existent-item"): 1.0}
    state = production_calculator.calculate_requirements(target)
    # Only the substation should be added if no recipe is found for the target item
    assert len(state.entities) == 1
    assert state.entities[0].entity_type_id == "substation"

def test_calculate_requirements_zero_production(production_calculator):
    target = {ItemID("iron-gear-wheel"): 0.0}
    state = production_calculator.calculate_requirements(target)
    # Should still add a substation, but no assemblers for 0 production
    assert len(state.entities) == 1
    assert state.entities[0].entity_type_id == "substation"

# --- Tests for FitnessEvaluator ---
@pytest.fixture
def fitness_evaluator(state_manager):
    return FitnessEvaluator(state_manager)

def test_evaluate_footprint_no_entities(fitness_evaluator):
    state = BlueprintState(entities=[])
    assert fitness_evaluator.evaluate_footprint(state) == 0.0

def test_evaluate_footprint_single_entity(fitness_evaluator):
    entity = EntityInstance(
        entity_number=1,
        entity_type_id=EntityTypeID("assembling-machine-2"), # 3x3 size
        position=Position(x=0.0, y=0.0)
    )
    state = BlueprintState(entities=[entity])
    # Bounds: -1.5 to 1.5 for x and y. Width/Height = 3. Area = 3*3 = 9.0
    assert fitness_evaluator.evaluate_footprint(state) == 9.0

def test_evaluate_footprint_multiple_entities_no_gap(fitness_evaluator):
    entity1 = EntityInstance(entity_number=1, entity_type_id=EntityTypeID("assembling-machine-2"), position=Position(x=0.0, y=0.0))
    entity2 = EntityInstance(entity_number=2, entity_type_id=EntityTypeID("assembling-machine-2"), position=Position(x=3.0, y=0.0)) # Adjacent
    state = BlueprintState(entities=[entity1, entity2])
    # Entity1 bounds: (-1.5, 1.5, -1.5, 1.5)
    # Entity2 bounds: (1.5, 4.5, -1.5, 1.5)
    # Overall min_x = -1.5, max_x = 4.5 (width = 6)
    # Overall min_y = -1.5, max_y = 1.5 (height = 3)
    # Area = 6 * 3 = 18.0
    assert fitness_evaluator.evaluate_footprint(state) == 18.0

def test_evaluate_footprint_multiple_entities_with_gap(fitness_evaluator):
    entity1 = EntityInstance(entity_number=1, entity_type_id=EntityTypeID("assembling-machine-2"), position=Position(x=0.0, y=0.0))
    entity2 = EntityInstance(entity_number=2, entity_type_id=EntityTypeID("substation"), position=Position(x=10.0, y=10.0)) # Separated
    state = BlueprintState(entities=[entity1, entity2])
    # Entity1 bounds: (-1.5, 1.5, -1.5, 1.5)
    # Entity2 bounds: (9.0, 11.0, 9.0, 11.0)
    # Overall min_x = -1.5, max_x = 11.0 (width = 12.5)
    # Overall min_y = -1.5, max_y = 11.0 (height = 12.5)
    # Area = 12.5 * 12.5 = 156.25
    assert fitness_evaluator.evaluate_footprint(state) == 156.25

def test_evaluate_footprint_multiple_entities_overlapping(fitness_evaluator):
    entity1 = EntityInstance(entity_number=1, entity_type_id=EntityTypeID("assembling-machine-2"), position=Position(x=0.0, y=0.0))
    entity2 = EntityInstance(entity_number=2, entity_type_id=EntityTypeID("substation"), position=Position(x=0.5, y=0.5)) # Overlapping
    state = BlueprintState(entities=[entity1, entity2])
    # Entity1 bounds: (-1.5, 1.5, -1.5, 1.5)
    # Entity2 bounds: (-0.5, 1.5, -0.5, 1.5)
    # Overall min_x = -1.5, max_x = 1.5 (width = 3)
    # Overall min_y = -1.5, max_y = 1.5 (height = 3)
    # Area = 3 * 3 = 9.0 (Same as a single 3x3 entity, as they are fully contained within it)
    assert fitness_evaluator.evaluate_footprint(state) == 9.0

def test_evaluate_returns_footprint(fitness_evaluator):
    entity = EntityInstance(
        entity_number=1,
        entity_type_id=EntityTypeID("assembling-machine-2"), # 3x3 size
        position=Position(x=0.0, y=0.0)
    )
    state = BlueprintState(entities=[entity])
    assert fitness_evaluator.evaluate(state) == 9.0
