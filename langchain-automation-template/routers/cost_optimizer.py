import time
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timedelta

from core.logger import get_logger
from core.types import ModelConfig # Assuming ModelConfig might be useful, though costs are separate for now
# If ModelConfig had cost info, it would be imported from core.types

logger = get_logger(__name__)

# Placeholder for actual model cost data structure if fetched from an external source
# For now, costs are expected to be in the main config YAML (router.yaml)

class CostOptimizer:
    """
    Manages and optimizes LLM usage costs based on configurations and tracked spending.
    """

    def __init__(self, cost_config: Dict[str, Any], model_configs: Dict[str, ModelConfig]):
        """
        Initializes the CostOptimizer.

        Args:
            cost_config (Dict[str, Any]): Configuration section for cost optimization
                                          (e.g., from router.yaml['cost_optimization']).
            model_configs (Dict[str, ModelConfig]): All available model configurations,
                                                 useful for finding alternatives.
        """
        self.daily_budget = cost_config.get("daily_budget", float('inf'))
        self.max_cost_per_request = cost_config.get("max_cost_per_request", float('inf'))
        self.prefer_cheaper_models = cost_config.get("prefer_cheaper_models", False)
        self.emergency_fallback_model = cost_config.get("emergency_fallback_model")
        self.model_token_costs = cost_config.get("model_costs", {}) # Cost per 1M tokens (input, output)

        self.model_configs = model_configs # All loaded ModelConfig objects from models.yaml

        self.current_daily_spend: float = 0.0
        self.last_reset_date = datetime.utcnow().date()

        # In-memory tracking of requests for more granular analysis (optional)
        self.request_log: List[Dict[str, Any]] = []

        logger.info(f"CostOptimizer initialized. Daily budget: ${self.daily_budget:.2f}, Max cost/req: ${self.max_cost_per_request:.2f}")
        logger.debug(f"Model costs loaded: {self.model_token_costs}")
        if not self.model_token_costs:
            logger.warning("No model costs found in configuration. Cost optimization will be limited.")
        if not self.emergency_fallback_model:
            logger.warning("Emergency fallback model not configured. Budget overruns may not have a fallback.")


    def _reset_if_new_day(self):
        """Resets daily spend if a new UTC day has started."""
        current_date = datetime.utcnow().date()
        if current_date > self.last_reset_date:
            logger.info(f"New day detected. Resetting daily spend from ${self.current_daily_spend:.2f} to $0.00.")
            self.current_daily_spend = 0.0
            self.last_reset_date = current_date
            self.request_log.clear() # Also clear request log daily

    def estimate_request_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimates the cost of a single LLM request.

        Args:
            model_name (str): The identifier of the model.
            input_tokens (int): Number of input tokens.
            output_tokens (int): Number of output tokens.

        Returns:
            float: Estimated cost of the request in USD (or configured currency).
                   Returns 0.0 if cost info for the model is not available.
        """
        self._reset_if_new_day() # Ensure daily budget is current

        if model_name not in self.model_token_costs:
            logger.warning(f"Cost information not found for model: {model_name}. Cannot estimate cost.")
            return 0.0

        costs = self.model_token_costs[model_name]
        input_cost_pm = costs.get("input_cost_pm", 0.0)  # Cost per million input tokens
        output_cost_pm = costs.get("output_cost_pm", 0.0) # Cost per million output tokens

        estimated_cost = ((input_tokens / 1_000_000) * input_cost_pm) +                          ((output_tokens / 1_000_000) * output_cost_pm)

        logger.debug(f"Estimated cost for {model_name} (in: {input_tokens}, out: {output_tokens}): ${estimated_cost:.6f}")
        return estimated_cost

    def record_spend(self, model_name: str, estimated_cost: float, input_tokens: int, output_tokens: int):
        """
        Records the spend for a request and updates the daily total.
        """
        self._reset_if_new_day()

        self.current_daily_spend += estimated_cost

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "estimated_cost": estimated_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "daily_spend_after": self.current_daily_spend
        }
        self.request_log.append(log_entry)

        logger.info(f"Recorded spend of ${estimated_cost:.4f} for {model_name}. Daily total: ${self.current_daily_spend:.2f}")

        if self.current_daily_spend > self.daily_budget:
            logger.warning(f"Daily budget of ${self.daily_budget:.2f} exceeded! Current spend: ${self.current_daily_spend:.2f}")
            # Future: Implement alert mechanisms (e.g., email, notification)

    def is_budget_exceeded(self, estimated_request_cost: float = 0.0) -> bool:
        """Checks if the daily budget is or would be exceeded by an upcoming request."""
        self._reset_if_new_day()
        return (self.current_daily_spend + estimated_request_cost) > self.daily_budget

    def is_request_too_expensive(self, estimated_request_cost: float) -> bool:
        """Checks if a single request's estimated cost exceeds the configured max_cost_per_request."""
        return estimated_request_cost > self.max_cost_per_request

    def get_current_spending_summary(self) -> Dict[str, Any]:
        self._reset_if_new_day()
        return {
            "daily_budget": self.daily_budget,
            "current_daily_spend": self.current_daily_spend,
            "remaining_daily_budget": max(0, self.daily_budget - self.current_daily_spend),
            "last_reset_date": self.last_reset_date.isoformat(),
            "requests_today": len(self.request_log)
        }

    def suggest_alternative_model(
        self,
        original_model_name: str,
        task_classification: Dict[str, Any], # From TaskClassifier
        user_preferences: Optional[Dict[str, Any]] = None # From UserPreferences
    ) -> Optional[str]:
        """
        Suggests a cheaper alternative model if the original is too expensive or budget is tight.
        This is a basic implementation. More sophisticated logic would consider model capabilities.

        Args:
            original_model_name (str): The initially selected model.
            task_classification (Dict[str, Any]): The classification of the current task.
            user_preferences (Optional[Dict[str, Any]]): User's preferences.

        Returns:
            Optional[str]: Name of a cheaper alternative model, or None if no suitable alternative.
        """
        self._reset_if_new_day()
        user_prefs = user_preferences or {}

        # Try emergency fallback if budget is already blown
        if self.is_budget_exceeded() and self.emergency_fallback_model:
            if self.emergency_fallback_model != original_model_name: # and is actually cheaper (assume it is)
                logger.warning(f"Daily budget exceeded. Suggesting emergency fallback: {self.emergency_fallback_model}")
                return self.emergency_fallback_model
            else: # if original is already the fallback, nothing to suggest
                return None

        if not self.prefer_cheaper_models and not user_prefs.get("cost_priority") == "low_cost":
            return None # No preference for cheaper models

        # Simplified: Find any model cheaper than original.
        # A real implementation would need to match capabilities from ModelConfig.
        # For now, we'll just look at the cost list.
        original_cost_info = self.model_token_costs.get(original_model_name)
        if not original_cost_info:
            return None # Cannot compare if original model cost is unknown

        # Estimate cost for a "typical" interaction (e.g. 1k input, 0.5k output) for comparison
        # This is very rough. A better way would be to use actual estimated tokens for the current query.
        typical_input_tokens = int(task_classification.get("details",{}).get("estimated_context_tokens", 1000) * 0.7)
        typical_output_tokens = int(task_classification.get("details",{}).get("estimated_context_tokens", 1000) * 0.3)

        original_typical_cost = self.estimate_request_cost(original_model_name, typical_input_tokens, typical_output_tokens)

        cheapest_alternative: Optional[str] = None
        cheapest_alternative_cost = original_typical_cost

        # Iterate through all models that have cost information
        for model_name, cost_info in self.model_token_costs.items():
            if model_name == original_model_name:
                continue
            if user_prefs.get("banned_models") and model_name in user_prefs["banned_models"]:
                continue

            # Check if this model is listed in model_configs (i.e., it's a known, configured model)
            if model_name not in self.model_configs:
                logger.debug(f"Skipping model '{model_name}' for cost optimization as it's not in loaded model_configs.")
                continue

            current_model_typical_cost = self.estimate_request_cost(model_name, typical_input_tokens, typical_output_tokens)

            if current_model_typical_cost < cheapest_alternative_cost:
                # Further checks: Does this model meet basic requirements?
                # For now, we assume any model with cost info is a potential candidate.
                # A future enhancement would be to check ModelConfig capabilities against task requirements.
                cheapest_alternative = model_name
                cheapest_alternative_cost = current_model_typical_cost

        if cheapest_alternative:
            logger.info(f"Suggesting cheaper alternative to {original_model_name}: {cheapest_alternative} (Est. cost diff for typical query: ${original_typical_cost - cheapest_alternative_cost:.4f})")
            return cheapest_alternative

        return None


if __name__ == '__main__':
    # Mock configurations for testing
    mock_cost_config = {
        "daily_budget": 1.00, # $1.00
        "max_cost_per_request": 0.10, # 10 cents
        "prefer_cheaper_models": True,
        "emergency_fallback_model": "model_C_free",
        "model_costs": {
            "model_A_expensive": {"input_cost_pm": 10.00, "output_cost_pm": 30.00},
            "model_B_medium": {"input_cost_pm": 1.00, "output_cost_pm": 3.00},
            "model_C_free": {"input_cost_pm": 0.00, "output_cost_pm": 0.00},
            "model_D_no_cost_info": {}
        }
    }
    # Mock model_configs (normally loaded from models.yaml via a config manager)
    mock_model_configs = {
        "model_A_expensive": {"name": "Model A", "provider": "ProviderX", "api_key_env": "X", "model_name": "A", "temperature": 0.7, "max_tokens": 100},
        "model_B_medium": {"name": "Model B", "provider": "ProviderY", "api_key_env": "Y", "model_name": "B", "temperature": 0.7, "max_tokens": 100},
        "model_C_free": {"name": "Model C", "provider": "ProviderZ", "api_key_env": "Z", "model_name": "C", "temperature": 0.7, "max_tokens": 100},
    }

    optimizer = CostOptimizer(mock_cost_config, mock_model_configs)

    print("--- Initial State ---")
    print(optimizer.get_current_spending_summary())

    print("\n--- Estimating Costs ---")
    cost_a = optimizer.estimate_request_cost("model_A_expensive", 10000, 2000) # 10k in, 2k out
    print(f"Cost for Model A: ${cost_a:.6f}")
    cost_b = optimizer.estimate_request_cost("model_B_medium", 10000, 2000)
    print(f"Cost for Model B: ${cost_b:.6f}")
    cost_c = optimizer.estimate_request_cost("model_C_free", 10000, 2000)
    print(f"Cost for Model C: ${cost_c:.6f}")
    cost_d = optimizer.estimate_request_cost("model_D_no_cost_info", 10000, 2000) # Should be 0.0 and log warning
    print(f"Cost for Model D (no info): ${cost_d:.6f}")


    print("\n--- Recording Spend & Budget Checks ---")
    optimizer.record_spend("model_B_medium", cost_b, 10000, 2000)
    print(f"Budget exceeded after spending ${cost_b:.4f}? {optimizer.is_budget_exceeded()}")
    print(f"Is ${cost_a:.4f} (Model A) too expensive for a single request? {optimizer.is_request_too_expensive(cost_a)}")

    # Simulate spending more
    optimizer.record_spend("model_B_medium", cost_b * 5, 50000, 10000) # Spend 5x cost_b
    print(optimizer.get_current_spending_summary())

    # Try to spend beyond budget
    print(f"Budget exceeded before next spend? {optimizer.is_budget_exceeded(cost_a)}") # Check if adding cost_a exceeds
    optimizer.record_spend("model_A_expensive", cost_a, 10000,2000) # This should trigger budget warning
    print(optimizer.get_current_spending_summary())

    print("\n--- Suggesting Alternatives ---")
    mock_task_classification = {
        "category": "coding",
        "confidence": 0.9,
        "details": {"complexity_level": "medium", "estimated_context_tokens": 1000}
    }
    # 1. Original model is expensive, prefer cheaper is true
    alt = optimizer.suggest_alternative_model("model_A_expensive", mock_task_classification)
    print(f"Alternative for model_A_expensive: {alt}") # Expect model_B_medium or model_C_free

    # 2. Budget is now exceeded, should suggest emergency fallback
    alt_emergency = optimizer.suggest_alternative_model("model_A_expensive", mock_task_classification)
    print(f"Alternative for model_A_expensive (budget exceeded): {alt_emergency}") # Expect model_C_free

    # 3. Test with prefer_cheaper_models = False (need to re-init or mock)
    optimizer.prefer_cheaper_models = False
    alt_no_pref = optimizer.suggest_alternative_model("model_A_expensive", mock_task_classification)
    print(f"Alternative for model_A_expensive (prefer_cheaper=False, budget exceeded): {alt_no_pref}") # Still fallback due to budget
    optimizer.prefer_cheaper_models = True # reset

    # 4. Test with a user preference for low cost
    low_cost_user_prefs = {"cost_priority": "low_cost"}
    alt_user_low_cost = optimizer.suggest_alternative_model("model_A_expensive", mock_task_classification, low_cost_user_prefs)
    print(f"Alternative for model_A_expensive (user low_cost, budget exceeded): {alt_user_low_cost}")


    print("\n--- Test Daily Reset (Manual Simulation) ---")
    optimizer.last_reset_date = datetime.utcnow().date() - timedelta(days=1) # Simulate yesterday
    print(f"Spend before reset check: {optimizer.current_daily_spend}")
    optimizer._reset_if_new_day() # Call explicitly for test
    print(f"Spend after reset check: {optimizer.current_daily_spend}")
    print(optimizer.get_current_spending_summary())
