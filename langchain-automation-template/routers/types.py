from typing import NamedTuple, Optional, Dict, Any
from core.types import TaskClassification # Re-export or use directly
from core.user_preferences import UserPreferences # Re-export or use directly

class RoutingResult(NamedTuple):
    selected_model: str
    task_classification: TaskClassification
    estimated_cost: float
    routing_reason: str
    original_query: str
    user_preferences: Optional[UserPreferences]
    error_message: Optional[str] = None
    latency_ms: float = 0.0
    input_tokens_estimated: Optional[int] = None
    output_tokens_estimated: Optional[int] = None

# You can add other router-specific types here if needed in the future
