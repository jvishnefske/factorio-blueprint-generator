import json
import math
import random
import copy
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum

# Import the factorio-draftsman library for the final export step
try:
    from draftsman.blueprint import Blueprint
    from draftsman.entity import Entity
    from draftsman.classes.position import Position as DraftsmanPosition
    DRAFTSMAN_INSTALLED = True
except ImportError:
    DRAFTSMAN_INSTALLED = False
    print("Warning: factorio-draftsman is not installed. Blueprint export will be skipped.")
    print("Install it with: pip install factorio-draftsman")


# ==============================================================================
# PART 1: PYDANTIC MODELS (The Core Data Interface)
# ==============================================================================

class AppBaseModel(BaseModel):
    # Allow arbitrary types for custom string subclasses (ItemID, EntityTypeID, RecipeID)
    # This is needed for Pydantic v2 to correctly handle these types in schema generation.
    model_config = ConfigDict(arbitrary_types_allowed=True)

class Position(AppBaseModel):
    x: float
    y: float

class Size(AppBaseModel):
    width: int
    height: int

class Direction(int, Enum):
    NORTH = 0
    EAST = 2
    SOUTH = 4
    WEST = 6

class ItemID(str): pass
class EntityTypeID(str): pass
class RecipeID(str): pass

class RecipeIngredient(AppBaseModel):
    item_id: ItemID
    amount: float

class EntityType(AppBaseModel):
    type_id: EntityTypeID
    size: Size
    crafting_speed: Optional[float] = None
    module_slots: int = 0
    power_consumption_kw: float = 0
    supply_area: Optional[Size] = None

class Recipe(AppBaseModel):
    recipe_id: RecipeID
    ingredients: List[RecipeIngredient]
    products: List[RecipeIngredient]
    energy_required_seconds: float

class EntityInstance(AppBaseModel):
    entity_number: int
    entity_type_id: EntityTypeID
    position: Optional[Position] = None
    direction: Direction = Direction.NORTH
    recipe_id: Optional[RecipeID] = None

class BlueprintState(AppBaseModel):
    name: str = "Optimized Blueprint"
    entities: List[EntityInstance] = Field(default_factory=list)

class OptimizationProblem(AppBaseModel):
    target_production: Dict[ItemID, float]
    available_entity_types: Dict[EntityTypeID, EntityType]
    available_recipes: Dict[RecipeID, Recipe]
    initial_state: BlueprintState


# ==============================================================================
# PART 2: SOFTWARE MODULES
# ==============================================================================

# --- MODULE 1: Game Data Loader ---
class GameDataLoader:
    @staticmethod
    def load_from_json(entity_json_path: str, recipe_json_path: str) -> Tuple[Dict[EntityTypeID, EntityType], Dict[RecipeID, Recipe]]:
        with open(entity_json_path, 'r') as f:
            entity_data = json.load(f)
        with open(recipe_json_path, 'r') as f:
            recipe_data = json.load(f)

        entity_types = {k: EntityType(**v) for k, v in entity_data.items()}
        recipes = {k: Recipe(**v) for k, v in recipe_data.items()}
        return entity_types, recipes

# --- MODULE 2: Production Calculator ---
class ProductionCalculator:
    def __init__(self, entity_types: Dict[EntityTypeID, EntityType], recipes: Dict[RecipeID, Recipe]):
        self.entity_types = entity_types
        self.recipes = recipes

    def calculate_requirements(self, target_production: Dict[ItemID, float]) -> BlueprintState:
        """A simplified calculator. A real one would be recursive."""
        entities_to_create = []
        entity_counter = 1

        for item_id, rate in target_production.items():
            # Find a recipe that produces this item
            producing_recipe_id = next((rid for rid, r in self.recipes.items() if any(p.item_id == item_id for p in r.products)), None)
            if not producing_recipe_id:
                continue

            recipe = self.recipes[producing_recipe_id]
            # Assume we are using assembling-machine-2
            assembler_type = self.entity_types['assembling-machine-2']
            
            # Calculate how many machines are needed
            items_per_second_per_machine = len(recipe.products) / recipe.energy_required_seconds * assembler_type.crafting_speed
            machines_needed = math.ceil(rate / items_per_second_per_machine)

            print(f"To produce {rate}/s of {item_id}, we need {machines_needed} assemblers with recipe '{producing_recipe_id}'")

            for _ in range(machines_needed):
                entities_to_create.append(
                    EntityInstance(
                        entity_number=entity_counter,
                        entity_type_id=assembler_type.type_id,
                        recipe_id=producing_recipe_id
                    )
                )
                entity_counter += 1

        # Add a substation to power everything
        entities_to_create.append(EntityInstance(entity_number=entity_counter, entity_type_id=EntityTypeID("substation")))
        
        return BlueprintState(entities=entities_to_create)

# --- MODULE 3: State Manager & Constraint Validator ---
class StateManager:
    def __init__(self, entity_definitions: Dict[EntityTypeID, EntityType]):
        self.entity_definitions = entity_definitions

    def get_entity_bounds(self, entity: EntityInstance) -> Tuple[float, float, float, float]:
        """Calculates the bounding box of a single entity."""
        if not entity.position:
            raise ValueError("Entity has no position.")
        size = self.entity_definitions[entity.entity_type_id].size
        # Center of entity is at (pos.x, pos.y), calculate corners
        half_width = size.width / 2.0
        half_height = size.height / 2.0
        x_min = entity.position.x - half_width
        x_max = entity.position.x + half_width
        y_min = entity.position.y - half_height
        y_max = entity.position.y + half_height
        return x_min, x_max, y_min, y_max

    def do_entities_overlap(self, entity1: EntityInstance, entity2: EntityInstance) -> bool:
        """Checks if two entities' bounding boxes overlap."""
        e1_x_min, e1_x_max, e1_y_min, e1_y_max = self.get_entity_bounds(entity1)
        e2_x_min, e2_x_max, e2_y_min, e2_y_max = self.get_entity_bounds(entity2)
        
        # Check for non-overlap
        if e1_x_max <= e2_x_min or e1_x_min >= e2_x_max:
            return False
        if e1_y_max <= e2_y_min or e1_y_min >= e2_y_max:
            return False
        
        return True # They overlap

    def is_state_valid(self, state: BlueprintState) -> bool:
        """Checks the entire state for validity (currently just for overlaps)."""
        for i in range(len(state.entities)):
            for j in range(i + 1, len(state.entities)):
                if self.do_entities_overlap(state.entities[i], state.entities[j]):
                    return False # Found an overlap
        return True


# --- MODULE 4: Fitness Evaluator ---
class FitnessEvaluator:
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager

    def evaluate_footprint(self, state: BlueprintState) -> float:
        """Calculates the total area of the layout's bounding box."""
        if not state.entities:
            return 0.0

        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')

        for entity in state.entities:
            e_min_x, e_max_x, e_min_y, e_max_y = self.state_manager.get_entity_bounds(entity)
            min_x = min(min_x, e_min_x)
            max_x = max(max_x, e_max_x)
            min_y = min(min_y, e_min_y)
            max_y = max(max_y, e_max_y)

        return (max_x - min_x) * (max_y - min_y)

    def evaluate(self, state: BlueprintState) -> float:
        """Calculates the total fitness score (lower is better)."""
        # Primary objective: minimize footprint. More objectives could be added here.
        return self.evaluate_footprint(state)

# --- MODULE 5: Optimization Engine ---
class SimulatedAnnealingOptimizer:
    def __init__(self, state_manager: StateManager, fitness_evaluator: FitnessEvaluator, initial_temp: float, final_temp: float, alpha: float, iterations_per_temp: int):
        self.state_manager = state_manager
        self.fitness_evaluator = fitness_evaluator
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.iterations_per_temp = iterations_per_temp

    def _get_random_neighbor(self, state: BlueprintState) -> BlueprintState:
        """Generates a new state by making a small random change to the current one."""
        neighbor_state = copy.deepcopy(state)
        
        if not neighbor_state.entities:
            return neighbor_state
            
        # Choose a random entity to move
        entity_to_move = random.choice(neighbor_state.entities)
        
        # Move it by a small random amount
        move_x = random.uniform(-2, 2)
        move_y = random.uniform(-2, 2)
        entity_to_move.position.x += move_x
        entity_to_move.position.y += move_y
        
        # Round to snap to grid (Factorio entities have 0.5 grid alignment)
        entity_to_move.position.x = round(entity_to_move.position.x * 2) / 2
        entity_to_move.position.y = round(entity_to_move.position.y * 2) / 2

        return neighbor_state

    def optimize(self, initial_state: BlueprintState) -> BlueprintState:
        """Performs the simulated annealing optimization."""
        # Start with a random valid placement
        current_state = copy.deepcopy(initial_state)
        grid_size = len(current_state.entities) * 5
        for entity in current_state.entities:
            while True:
                entity.position = Position(x=random.uniform(-grid_size, grid_size), y=random.uniform(-grid_size, grid_size))
                if self.state_manager.is_state_valid(current_state):
                    break

        current_score = self.fitness_evaluator.evaluate(current_state)
        best_state = copy.deepcopy(current_state)
        best_score = current_score
        
        temp = self.initial_temp

        while temp > self.final_temp:
            for _ in range(self.iterations_per_temp):
                neighbor_state = self._get_random_neighbor(current_state)
                
                # If the new state is invalid, discard it
                if not self.state_manager.is_state_valid(neighbor_state):
                    continue

                neighbor_score = self.fitness_evaluator.evaluate(neighbor_state)
                
                delta = neighbor_score - current_score
                
                # If the new state is better, or if we accept a worse state by chance
                if delta < 0 or random.uniform(0, 1) < math.exp(-delta / temp):
                    current_state = neighbor_state
                    current_score = neighbor_score
                    
                    if current_score < best_score:
                        best_state = copy.deepcopy(current_state)
                        best_score = current_score

            print(f"Temp: {temp:.2f}, Current Score: {current_score:.2f}, Best Score: {best_score:.2f}")
            temp *= self.alpha
            
        return best_state

# --- MODULE 6: Blueprint Exporter ---
class BlueprintExporter:
    @staticmethod
    def to_draftsman_string(state: BlueprintState) -> Optional[str]:
        if not DRAFTSMAN_INSTALLED:
            print("Cannot export: factorio-draftsman library is not available.")
            return None
            
        blueprint = Blueprint()
        blueprint.label = state.name
        
        for entity_instance in state.entities:
            if entity_instance.position is None:
                continue
            
            new_entity = Entity(
                name=entity_instance.entity_type_id,
                position=DraftsmanPosition(entity_instance.position.x, entity_instance.position.y),
                direction=entity_instance.direction.value,
            )
            if entity_instance.recipe_id:
                new_entity.recipe = entity_instance.recipe_id
            
            blueprint.entities.append(new_entity)
            
        return blueprint.to_string()


# ==============================================================================
# PART 3: MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # --- Setup: Create dummy game data files ---
    entity_data = {
        "assembling-machine-2": {
            "type_id": "assembling-machine-2",
            "size": {"width": 3, "height": 3},
            "crafting_speed": 0.75
        },
        "substation": {
            "type_id": "substation",
            "size": {"width": 2, "height": 2},
            "supply_area": {"width": 18, "height": 18}
        }
    }
    recipe_data = {
        "iron-gear-wheel": {
            "recipe_id": "iron-gear-wheel",
            "ingredients": [{"item_id": "iron-plate", "amount": 2}],
            "products": [{"item_id": "iron-gear-wheel", "amount": 1}],
            "energy_required_seconds": 0.5
        }
    }
    with open("entities.json", "w") as f:
        json.dump(entity_data, f)
    with open("recipes.json", "w") as f:
        json.dump(recipe_data, f)

    # 1. Load Game Data
    print("--- 1. Loading Game Data ---")
    entity_types, recipes = GameDataLoader.load_from_json("entities.json", "recipes.json")
    print(f"Loaded {len(entity_types)} entity types and {len(recipes)} recipes.")

    # 2. Define Problem and Calculate Requirements
    print("\n--- 2. Defining Problem & Calculating Machine Requirements ---")
    target = {ItemID("iron-gear-wheel"): 5.0} # Target: 5 iron gears per second
    
    calculator = ProductionCalculator(entity_types, recipes)
    initial_blueprint_state = calculator.calculate_requirements(target)
    print(f"Calculation complete. Required entities: {[e.entity_type_id for e in initial_blueprint_state.entities]}")

    # 3. Initialize Core Modules
    print("\n--- 3. Initializing Optimization Modules ---")
    state_manager = StateManager(entity_types)
    fitness_evaluator = FitnessEvaluator(state_manager)
    optimizer = SimulatedAnnealingOptimizer(
        state_manager=state_manager,
        fitness_evaluator=fitness_evaluator,
        initial_temp=1000,
        final_temp=0.1,
        alpha=0.95,
        iterations_per_temp=50
    )
    print("Optimizer initialized.")

    # 4. Run Optimization
    print("\n--- 4. Starting Optimization ---")
    optimized_state = optimizer.optimize(initial_blueprint_state)
    print("Optimization finished.")

    # 5. Export Final Blueprint
    print("\n--- 5. Exporting to Blueprint String ---")
    final_blueprint_string = BlueprintExporter.to_draftsman_string(optimized_state)
    
    if final_blueprint_string:
        print("\nSUCCESS! You can import this string directly into Factorio:\n")
        print("="*60)
        print(final_blueprint_string)
        print("="*60)
