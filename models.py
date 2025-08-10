from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum

# --- Foundational Primitives ---

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

# --- Generic Game Data Definitions (To be injected) ---

class RecipeIngredient(AppBaseModel):
    item_id: ItemID
    amount: float

class EntityType(AppBaseModel):
    """Generic definition of an entity that can be placed."""
    type_id: EntityTypeID
    size: Size
    power_consumption_kw: float = 0
    crafting_speed: Optional[float] = None
    module_slots: int = 0
    # For beacons
    effect_distribution_area: Optional[Size] = None
    # For power poles
    supply_area: Optional[Size] = None
    wire_reach: Optional[float] = None
    
class Recipe(AppBaseModel):
    """Generic definition of a recipe."""
    recipe_id: RecipeID
    ingredients: List[RecipeIngredient]
    products: List[RecipeIngredient]
    energy_required_seconds: float
    category: str # e.g., "crafting", "smelting"


# --- State Representation of the Placed Blueprint ---

class EntityInstance(AppBaseModel):
    """Represents a single placed entity on the grid."""
    entity_number: int = Field(..., description="A unique ID for this instance.")
    entity_type_id: EntityTypeID
    position: Optional[Position] = None
    direction: Direction = Direction.NORTH
    
    # Specific state for certain entities
    recipe_id: Optional[RecipeID] = None
    # modules: Optional[Dict[ItemID, int]] = None # Could add modules here

class LogisticLink(AppBaseModel):
    """Represents a material flow between two entities, typically by an inserter."""
    source_entity: int
    destination_entity: int
    item_id: ItemID
    rate_per_second: float # Required flow rate from the calculator

class BlueprintState(AppBaseModel):
    """The complete, self-contained state of a factory layout. 
    This is the object the optimization algorithm operates on."""
    name: str = "Optimized Blueprint"
    entities: List[EntityInstance] = Field(default_factory=list)
    # The logistics graph is derived from the calculator and is a target
    required_links: List[LogisticLink] = Field(default_factory=list)
    
    # Grid dimensions can be dynamically calculated or fixed
    width: Optional[int] = None
    height: Optional[int] = None

class OptimizationProblem(AppBaseModel):
    """Top-level model defining the entire problem to be solved."""
    target_production: Dict[ItemID, float] = Field(..., description="e.g., {'processing-unit': 10}")
    
    # Injected game data definitions
    available_entity_types: Dict[EntityTypeID, EntityType]
    available_recipes: Dict[RecipeID, Recipe]
    
    # The state that will be optimized
    initial_state: BlueprintState
