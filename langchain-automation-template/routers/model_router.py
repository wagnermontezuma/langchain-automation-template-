import time
import yaml # For loading config
from typing import Dict, Optional, Any, NamedTuple

from core.logger import get_logger
from core.types import TaskClassification, ModelConfig # Assuming TaskClassification is from core.types
from core.user_preferences import UserPreferences, CostPriority
from core.exceptions import ConfigError, RuntimeError as CoreRuntimeError # Alias to avoid clash
from core.monitoring import get_monitoring_system, MonitoringSystem
from .task_classifier import TaskClassifier
from .cost_optimizer import CostOptimizer
from .types import RoutingResult # Import from local types.py
from .routing_cache import RoutingCache # Add this import

logger = get_logger(__name__)

# Define a structure for the routing result
# Moved to routers/types.py
# class RoutingResult(NamedTuple):
#     selected_model: str
#     task_classification: TaskClassification
#     estimated_cost: float
#     routing_reason: str
#     original_query: str
#     user_preferences: Optional[UserPreferences]
#     error_message: Optional[str] = None
#     latency_ms: float = 0.0
#     input_tokens_estimated: Optional[int] = None
#     output_tokens_estimated: Optional[int] = None


DEFAULT_ROUTER_CONFIG_PATH = "config/router.yaml"
DEFAULT_MODELS_CONFIG_PATH = "config/models.yaml"

class IntelligentModelRouter:
    """
    Orchestrates task classification and model selection based on rules,
    cost optimization, and user preferences.
    """

    def __init__(
        self,
        router_config_path: str = DEFAULT_ROUTER_CONFIG_PATH,
        models_config_path: str = DEFAULT_MODELS_CONFIG_PATH,
        # routing_cache: Optional[RoutingCache] = None # For later
    ):
        logger.info(f"Initializing IntelligentModelRouter with config: {router_config_path} and models: {models_config_path}")
        self.router_config_path = router_config_path
        self.models_config_path = models_config_path

        self._load_configs() # Loads router_config and model_configs

        self.task_classifier = TaskClassifier()
        self.cost_optimizer = CostOptimizer(
            cost_config=self.router_config.get("cost_optimization", {}),
            model_configs=self.model_configs
        )
        self.monitoring_system: MonitoringSystem = get_monitoring_system()

        # Initialize RoutingCache based on config
        cache_config = self.router_config.get("routing_cache_config", {})
        if cache_config.get("enabled", False): # Check if cache is enabled in config
            self.cache = RoutingCache(
                default_ttl_seconds=cache_config.get("ttl_seconds", 3600),
                max_size=cache_config.get("max_size", 1000)
            )
            logger.info("RoutingCache enabled and initialized from router_config.")
        else:
            self.cache = None
            logger.info("RoutingCache is disabled in router_config.")

        logger.info("IntelligentModelRouter initialized successfully.")

    def _load_configs(self):
        """Loads router and model configurations from YAML files."""
        try:
            with open(self.router_config_path, 'r') as f:
                self.router_config = yaml.safe_load(f)
            if not self.router_config:
                raise ConfigError(f"Router configuration file is empty or invalid: {self.router_config_path}")
            logger.debug(f"Router configuration loaded from {self.router_config_path}")
        except FileNotFoundError:
            logger.error(f"Router configuration file not found: {self.router_config_path}")
            raise ConfigError(f"Router configuration file not found: {self.router_config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing router configuration YAML: {e}")
            raise ConfigError(f"Error parsing router configuration YAML: {e}")

        try:
            with open(self.models_config_path, 'r') as f:
                self.model_configs: Dict[str, ModelConfig] = yaml.safe_load(f) # Type hint here
            if not self.model_configs:
                raise ConfigError(f"Models configuration file is empty or invalid: {self.models_config_path}")
            logger.debug(f"Model configurations loaded from {self.models_config_path}")
        except FileNotFoundError:
            logger.error(f"Models configuration file not found: {self.models_config_path}")
            raise ConfigError(f"Models configuration file not found: {self.models_config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing models configuration YAML: {e}")
            raise ConfigError(f"Error parsing models configuration YAML: {e}")

        # Validate that routing rules point to existing models in model_configs or model_costs
        self._validate_routing_rules()


    def _validate_routing_rules(self):
        rules = self.router_config.get("routing_rules", {})
        for task_type, complexities in rules.items():
            if isinstance(complexities, dict):
                for complexity, model_name in complexities.items():
                    if model_name not in self.model_configs and model_name not in self.cost_optimizer.model_token_costs:
                        logger.warning(f"Model '{model_name}' in routing_rules for {task_type}/{complexity} not found in model_configs or model_costs.")
            elif isinstance(complexities, str): # e.g. default model for a type
                 if complexities not in self.model_configs and complexities not in self.cost_optimizer.model_token_costs:
                        logger.warning(f"Model '{complexities}' in routing_rules for {task_type}/default not found in model_configs or model_costs.")


    def classify_task(self, query: str) -> TaskClassification:
        """
        Classifies the given query using the TaskClassifier.
        Uses cache if enabled.
        """
        if self.cache:
            # Normalize query for cache key consistency if needed, or use raw query
            cache_key = f"classify_task:{query}"
            cached_classification = self.cache.get(cache_key)
            if cached_classification:
                logger.debug(f"Task classification cache hit for query: '{query}'")
                self.monitoring_system.record_cache_hit() # Record hit with monitoring system
                return cached_classification
            else:
                logger.debug(f"Task classification cache miss for query: '{query}'")
                self.monitoring_system.record_cache_miss() # Record miss

        classification = self.task_classifier.classify(query)

        if self.cache:
            # Use cache_key defined above
            self.cache.set(f"classify_task:{query}", classification)
            logger.debug(f"Stored task classification in cache for query: '{query}'")

        return classification

    def select_best_model(
        self,
        classification: TaskClassification,
        preferences: Optional[UserPreferences] = None
    ) -> Tuple[str, str, float]: # model_name, reason, estimated_cost_for_typical_tokens
        """
        Selects the best model based on classification, routing rules, cost, and preferences.
        Returns the model name (string), the reason for selection, and an estimated cost.
        """
        prefs = preferences or {}
        task_type = classification["category"]
        complexity = classification.get("details", {}).get("complexity_level", "default")

        # Estimated tokens for cost calculation during selection
        # These are *not* the final tokens used by the LLM call, but for selection logic.
        # The actual LLM call might refine this based on prompt engineering.
        est_input_tokens = classification.get("details", {}).get("estimated_context_tokens", 1000)
        est_output_tokens = int(est_input_tokens * 0.3) # Assume output is 30% of input for rough cost estimation

        # 1. Check user preferred models first if "performance" or "balanced" (and model exists)
        if prefs.get("cost_priority", "balanced") != "low_cost" and prefs.get("preferred_models"):
            for preferred_model in prefs["preferred_models"]:
                if preferred_model not in self.model_configs:
                    logger.warning(f"User preferred model '{preferred_model}' not found in configurations. Skipping.")
                    continue
                if prefs.get("banned_models") and preferred_model in prefs["banned_models"]:
                    continue # Skip if user also banned it (unlikely but possible)

                cost = self.cost_optimizer.estimate_request_cost(preferred_model, est_input_tokens, est_output_tokens)
                if self.cost_optimizer.is_request_too_expensive(cost):
                    logger.info(f"User preferred model {preferred_model} is too expensive for this request (est. ${cost:.4f}).")
                    continue
                if self.cost_optimizer.is_budget_exceeded(cost):
                    logger.info(f"User preferred model {preferred_model} would exceed daily budget (est. ${cost:.4f}).")
                    continue
                return preferred_model, f"User preferred model and within budget/cost limits.", cost

        # 2. Apply routing rules
        rules = self.router_config.get("routing_rules", {})
        model_name: Optional[str] = None
        reason = "No specific rule matched, using fallback logic."

        if task_type in rules:
            task_rules = rules[task_type]
            if isinstance(task_rules, str): # Direct model for the type
                model_name = task_rules
                reason = f"Direct rule for task type '{task_type}'."
            elif isinstance(task_rules, dict):
                if complexity in task_rules:
                    model_name = task_rules[complexity]
                    reason = f"Rule for task type '{task_type}' and complexity '{complexity}'."
                elif "default" in task_rules:
                    model_name = task_rules["default"]
                    reason = f"Default rule for task type '{task_type}'."
                else: # No specific complexity or default for this task type
                    reason = f"No complexity or default rule for task type '{task_type}'."

        if not model_name: # Fallback to a general default if no rules applied
            model_name = self.router_config.get("default_routing_model", self.cost_optimizer.emergency_fallback_model or list(self.model_configs.keys())[0])
            reason = f"No specific rules matched, using system default model: {model_name}"
            if not model_name: # Should not happen if configs are valid
                 raise CoreRuntimeError("No models available for routing after all fallbacks.")


        # 3. Cost Optimization and Budget Check for the rule-selected model
        if model_name:
            if prefs.get("banned_models") and model_name in prefs["banned_models"]:
                logger.info(f"Rule-selected model '{model_name}' is banned by user. Attempting alternative.")
                # Attempt to find an alternative not banned, or fallback to emergency.
                # This logic can be expanded. For now, clear model_name to trigger further fallback.
                model_name = None
                reason += " User banned. "


            if model_name: # If not cleared by ban
                current_model_cost = self.cost_optimizer.estimate_request_cost(model_name, est_input_tokens, est_output_tokens)

                if self.cost_optimizer.is_request_too_expensive(current_model_cost):
                    reason += f" Model '{model_name}' (est. ${current_model_cost:.4f}) exceeds max_cost_per_request. "
                    logger.warning(reason)
                    model_name = None # Trigger alternative search

                if model_name and self.cost_optimizer.is_budget_exceeded(current_model_cost):
                    reason += f" Model '{model_name}' (est. ${current_model_cost:.4f}) would exceed daily budget. "
                    logger.warning(reason)
                    model_name = None # Trigger alternative search

                # If model is still selected, consider suggesting a cheaper one based on preferences
                if model_name and (self.cost_optimizer.prefer_cheaper_models or prefs.get("cost_priority") == "low_cost"):
                    alternative_model = self.cost_optimizer.suggest_alternative_model(model_name, classification, prefs)
                    if alternative_model:
                        alt_cost = self.cost_optimizer.estimate_request_cost(alternative_model, est_input_tokens, est_output_tokens)
                        if not self.cost_optimizer.is_request_too_expensive(alt_cost) and                            not self.cost_optimizer.is_budget_exceeded(alt_cost):
                            reason += f" Cost optimization: Switched from '{model_name}' to cheaper '{alternative_model}'."
                            model_name = alternative_model
                            current_model_cost = alt_cost
                        else:
                            reason += f" Cheaper alternative '{alternative_model}' still too expensive or over budget."

                if model_name: # If a model is still selected after all checks
                    return model_name, reason, current_model_cost


        # 4. Fallback if no model selected yet (e.g. rule model too expensive/over budget, or banned)
        logger.warning(f"Primary model selection failed or resulted in no model. Reason: {reason}. Attempting fallback.")
        if self.cost_optimizer.emergency_fallback_model:
            fallback_model = self.cost_optimizer.emergency_fallback_model
            if not (prefs.get("banned_models") and fallback_model in prefs["banned_models"]):
                fallback_cost = self.cost_optimizer.estimate_request_cost(fallback_model, est_input_tokens, est_output_tokens)
                # Emergency fallback ignores max_cost_per_request but respects overall budget if possible
                if not self.cost_optimizer.is_budget_exceeded(fallback_cost) or self.cost_optimizer.current_daily_spend == 0: # Allow if budget is $0
                    logger.info(f"Using emergency fallback model: {fallback_model}")
                    return fallback_model, reason + f" Using emergency fallback '{fallback_model}'.", fallback_cost
                else:
                    reason += f" Emergency fallback '{fallback_model}' would also exceed budget."
                    logger.error(reason)
                    raise CoreRuntimeError(f"All models including emergency fallback would exceed budget or are unsuitable. Reason: {reason}")
            else:
                reason += f" Emergency fallback '{fallback_model}' is banned by user."
                logger.error(reason)
                raise CoreRuntimeError(f"Emergency fallback model banned by user. No model available. Reason: {reason}")

        # Absolute last resort: first model from config that's not banned (if any)
        # This part might be too risky if costs are not well-defined for all models.
        # Consider removing if emergency_fallback_model is made mandatory.
        for m_name in self.model_configs.keys():
            if not (prefs.get("banned_models") and m_name in prefs["banned_models"]):
                m_cost = self.cost_optimizer.estimate_request_cost(m_name, est_input_tokens, est_output_tokens)
                if not self.cost_optimizer.is_request_too_expensive(m_cost) and                    not self.cost_optimizer.is_budget_exceeded(m_cost):
                    return m_name, reason + f" Using last resort available model '{m_name}'.", m_cost

        logger.error(f"Unable to select any model after all fallbacks. Reason: {reason}")
        raise CoreRuntimeError(f"Failed to select a model. Reason: {reason}")


    def route_query(self, query: str, preferences: Optional[UserPreferences] = None) -> RoutingResult:
        """
        Routes the query: classifies, selects model, estimates cost, and logs.
        This method does NOT call the LLM; it only determines which LLM to call.
        """
        start_time = time.monotonic()
        error_message: Optional[str] = None
        selected_model_name: str = "unknown" # Default
        estimated_cost: float = 0.0
        reason: str = "Routing failed"
        final_classification: Optional[TaskClassification] = None

        try:
            # 1. Classify Task
            final_classification = self.classify_task(query)
            logger.info(f"Query classified: {final_classification}")

            # 2. Select Best Model
            selected_model_name, reason, estimated_cost = self.select_best_model(final_classification, preferences)
            logger.info(f"Model selected: {selected_model_name}. Reason: {reason}. Est. Cost: ${estimated_cost:.6f}")

        except ConfigError as e:
            logger.error(f"Configuration error during routing: {e}", exc_info=True)
            error_message = f"Configuration error: {str(e)}"
            # Fallback to emergency if possible, otherwise this is a critical setup issue
            selected_model_name = self.cost_optimizer.emergency_fallback_model or "critical_config_error_model"
            reason = f"Routing failed due to config error: {e}. Attempting emergency model."
            if final_classification is None: # Ensure classification is not None for RoutingResult
                 final_classification = {"task_description": query, "category": "unknown", "confidence": 0.0, "details": {}}

        except CoreRuntimeError as e: # Catch runtime errors from selection (e.g. no model available)
            logger.error(f"Runtime error during model selection: {e}", exc_info=True)
            error_message = f"Routing error: {str(e)}"
            # selected_model_name will be whatever was last attempted or default 'unknown'
            reason = f"Routing failed: {e}"
            if final_classification is None:
                 final_classification = {"task_description": query, "category": "unknown", "confidence": 0.0, "details": {}}


        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"Unexpected error during routing: {e}", exc_info=True)
            error_message = f"Unexpected error: {str(e)}"
            selected_model_name = self.cost_optimizer.emergency_fallback_model or "unexpected_error_model"
            reason = f"Routing failed due to unexpected error: {e}. Attempting emergency model."
            if final_classification is None:
                 final_classification = {"task_description": query, "category": "unknown", "confidence": 0.0, "details": {}}

        end_time = time.monotonic()
        latency_ms = (end_time - start_time) * 1000

        # Ensure final_classification is not None before creating RoutingResult
        if final_classification is None:
            # This case should ideally be handled within the except blocks,
            # but as a final safeguard:
            final_classification = self.task_classifier.classify(query) if query else                                    {"task_description": "Unknown (query missing)", "category": "unknown", "confidence": 0.0, "details": {}}


        # Log to monitoring system
        # Note: input/output tokens are estimated here for routing, actual tokens come from LLM call
        self.monitoring_system.log_routing_decision(
            query=query,
            task_classification=final_classification,
            selected_model=selected_model_name,
            routing_reason=reason,
            estimated_cost=estimated_cost, # This is the cost for the *selected* model
            latency_ms=latency_ms,
            user_preferences=preferences,
            error=error_message,
            input_tokens=final_classification.get("details", {}).get("estimated_context_tokens"),
            output_tokens=int(final_classification.get("details", {}).get("estimated_context_tokens", 0) * 0.3) # Rough output est.
        )

        # Record spend with CostOptimizer (important: this assumes the routed query WILL be executed)
        # In a real system, spend might be recorded *after* successful LLM execution.
        # For now, record based on routing decision.
        if not error_message and selected_model_name not in ["unknown", "critical_config_error_model", "unexpected_error_model"]:
             self.cost_optimizer.record_spend(
                selected_model_name,
                estimated_cost,
                final_classification.get("details", {}).get("estimated_context_tokens",0),
                int(final_classification.get("details", {}).get("estimated_context_tokens", 0) * 0.3)
            )


        return RoutingResult(
            selected_model=selected_model_name,
            task_classification=final_classification,
            estimated_cost=estimated_cost,
            routing_reason=reason,
            original_query=query,
            user_preferences=preferences,
            error_message=error_message,
            latency_ms=latency_ms,
            input_tokens_estimated=final_classification.get("details", {}).get("estimated_context_tokens"),
            output_tokens_estimated=int(final_classification.get("details", {}).get("estimated_context_tokens", 0) * 0.3)
        )

    def get_routing_stats(self) -> Dict[str, Any]:
        """Retrieves current routing statistics from the monitoring system."""
        return self.monitoring_system.get_routing_metrics()

    def get_cost_summary(self) -> Dict[str, Any]:
        """Retrieves current cost summary from the cost optimizer."""
        return self.cost_optimizer.get_current_spending_summary()


if __name__ == '__main__':
    # This example assumes config files (router.yaml, models.yaml) are in ../../config relative to this script if run directly
    # For proper testing, run from project root or ensure paths are correct.
    # To make it runnable from anywhere for a quick test, let's use absolute paths if possible or adjust relative.
    # For simplicity, this example might fail if paths are not set up for direct execution from `routers` dir.

    print("Testing IntelligentModelRouter...")
    try:
        # You might need to adjust these paths if you run this __main__ block directly
        # This assumes config/ is in the parent directory of routers/
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) # Assumes routers/ is one level down from project root

        ROUTER_CONFIG = os.path.join(project_root, "config/router.yaml")
        MODELS_CONFIG = os.path.join(project_root, "config/models.yaml")

        # Check if config files exist before initializing
        if not os.path.exists(ROUTER_CONFIG):
            print(f"ERROR: Router config not found at {ROUTER_CONFIG}. Create it first from project root based on examples.")
            exit(1)
        if not os.path.exists(MODELS_CONFIG):
            print(f"ERROR: Models config not found at {MODELS_CONFIG}. Create it first from project root based on examples.")
            exit(1)

        router = IntelligentModelRouter(router_config_path=ROUTER_CONFIG, models_config_path=MODELS_CONFIG)

        queries_to_test = [
            "Write a python function to calculate fibonacci. It should be efficient for large numbers.",
            "Analyze sales data from last quarter and provide a short summary of key trends. Keep it simple.",
            "Translate 'Hello, how are you?' to French.",
            "What is quantum computing? Explain it like I'm five.",
            "Generate a complex SQL query to find users who have not logged in for 6 months but made a purchase.",
            "Create a poem about the sea."
        ]

        for q in queries_to_test:
            print(f"\n--- Routing Query: '{q}' ---")
            result = router.route_query(q)
            print(f"  Selected Model: {result.selected_model}")
            print(f"  Task Category: {result.task_classification['category']}")
            print(f"  Task Complexity: {result.task_classification.get('details', {}).get('complexity_level')}")
            print(f"  Est. Cost: ${result.estimated_cost:.6f}")
            print(f"  Reason: {result.routing_reason}")
            if result.error_message:
                print(f"  Error: {result.error_message}")
            print(f"  Routing Latency: {result.latency_ms:.2f} ms")

        print("\n--- Routing with User Preferences ---")
        low_cost_prefs: UserPreferences = {"cost_priority": "low_cost"}
        query_prefs = "Write a python function to calculate fibonacci. It should be efficient."
        result_prefs = router.route_query(query_prefs, preferences=low_cost_prefs)
        print(f"Query: '{query_prefs}' with {low_cost_prefs}")
        print(f"  Selected Model: {result_prefs.selected_model}")
        print(f"  Reason: {result_prefs.routing_reason}")

        banned_prefs: UserPreferences = {"banned_models": [result_prefs.selected_model]} # Ban the previously selected model
        print(f"Query: '{query_prefs}' with banned model: {banned_prefs['banned_models']}")
        result_banned = router.route_query(query_prefs, preferences=banned_prefs)
        print(f"  Selected Model (after banning {result_prefs.selected_model}): {result_banned.selected_model}")
        print(f"  Reason: {result_banned.routing_reason}")


        print("\n--- Stats & Costs ---")
        print("Routing Stats:")
        stats = router.get_routing_stats()
        # print(json.dumps(stats, indent=2)) # Full stats can be verbose
        print(f"  Total Requests Routed: {stats.get('total_requests_routed')}")
        print(f"  Cache Hits: {stats.get('routing_cache_hits')}, Misses: {stats.get('routing_cache_misses')}")

        print("Cost Summary:")
        costs = router.get_cost_summary()
        # print(json.dumps(costs, indent=2))
        print(f"  Daily Budget: ${costs.get('daily_budget'):.2f}")
        print(f"  Current Spend: ${costs.get('current_daily_spend'):.2f}")
        print(f"  Remaining Budget: ${costs.get('remaining_daily_budget'):.2f}")


    except ConfigError as e:
        print(f"Failed to initialize router due to ConfigError: {e}")
    except CoreRuntimeError as e:
        print(f"Runtime error during router operation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", exc_info=True)
