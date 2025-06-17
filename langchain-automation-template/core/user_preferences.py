from typing import TypedDict, List, Optional, Literal

# Define literal types for cost priority to ensure type safety
CostPriority = Literal["low_cost", "balanced", "performance", "unset"]

class UserPreferences(TypedDict, total=False): # total=False allows for optional keys
    """
    Defines user-specific preferences that can influence model routing and behavior.
    All fields are optional. If not provided, system defaults will be used.
    """
    # Cost-related preferences
    cost_priority: CostPriority                               # User's preference for cost vs. performance.
    max_budget_per_query: Optional[float]                     # Max acceptable estimated cost for a single query.
    monthly_budget_limit: Optional[float]                     # User's overall monthly budget for LLM usage.

    # Model selection preferences
    preferred_models: Optional[List[str]]                     # List of specific model names the user prefers.
    banned_models: Optional[List[str]]                        # List of specific model names the user wants to avoid.
    require_specific_capabilities: Optional[List[str]]        # e.g., ["image_input", "tool_use_strong"]

    # Performance and quality preferences
    desired_latency_ms: Optional[int]                         # Preferred maximum latency for a response.
    quality_preference: Optional[Literal["high", "medium", "low"]] # General preference for output quality.

    # Other preferences
    language_output: Optional[str]                            # Preferred output language (e.g., "en", "es", "fr").
    custom_instructions: Optional[str]                        # General custom instructions to be prepended to prompts.

# Example of creating and using UserPreferences
if __name__ == '__main__':
    # Example 1: User focused on low cost
    low_cost_prefs: UserPreferences = {
        "cost_priority": "low_cost",
        "max_budget_per_query": 0.10, # 10 cents
        "banned_models": ["anthropic/claude-3-opus-20240229"] # Too expensive
    }
    print(f"Low Cost Preferences: {low_cost_prefs}")

    # Example 2: User focused on high performance/quality
    high_performance_prefs: UserPreferences = {
        "cost_priority": "performance",
        "preferred_models": ["openai/gpt-4-turbo-preview", "anthropic/claude-3-opus-20240229"],
        "quality_preference": "high",
        "desired_latency_ms": 2000 # Willing to wait a bit longer for quality
    }
    print(f"High Performance Preferences: {high_performance_prefs}")

    # Example 3: Balanced user, with some custom instructions
    balanced_prefs: UserPreferences = {
        "cost_priority": "balanced",
        "language_output": "es",
        "custom_instructions": "Por favor, sé muy formal en tus respuestas."
    }
    print(f"Balanced Preferences (Spanish): {balanced_prefs}")

    # Example 4: Minimal preferences (will use system defaults for most things)
    minimal_prefs: UserPreferences = {}
    print(f"Minimal Preferences: {minimal_prefs}")

    # Example with a specific capability requirement
    capability_prefs: UserPreferences = {
        "require_specific_capabilities": ["tool_use_strong"]
    }
    print(f"Capability Preferences: {capability_prefs}")
