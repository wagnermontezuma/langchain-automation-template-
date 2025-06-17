import time
from collections import defaultdict
from typing import Dict, List, Any, Optional
from datetime import datetime
from core.logger import get_logger

logger = get_logger(__name__)

class MonitoringSystem:
    """
    Handles monitoring of routing decisions, model usage, performance, and errors.
    This is a basic in-memory implementation. For production, consider integrating
    with dedicated monitoring tools (e.g., Prometheus, Grafana, Datadog).
    """

    def __init__(self):
        # Routing decisions log
        self.routing_decisions: List[Dict[str, Any]] = []

        # Model usage stats
        self.model_usage_counts: Dict[str, int] = defaultdict(int)
        self.model_total_latency_ms: Dict[str, float] = defaultdict(float)
        self.model_error_counts: Dict[str, int] = defaultdict(int)
        self.model_input_tokens: Dict[str, int] = defaultdict(int)
        self.model_output_tokens: Dict[str, int] = defaultdict(int)
        self.model_estimated_cost: Dict[str, float] = defaultdict(float)


        # Task classification stats
        self.task_classification_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int)) # type -> complexity -> count
        self.task_classification_confidence_sum: Dict[str, float] = defaultdict(float)
        self.task_classification_total_count: int = 0

        # Cache stats
        self.routing_cache_hits: int = 0
        self.routing_cache_misses: int = 0

        logger.info("MonitoringSystem initialized.")

    def log_routing_decision(
        self,
        query: str,
        task_classification: Dict[str, Any],
        selected_model: str,
        routing_reason: str,
        estimated_cost: float,
        latency_ms: float,
        user_preferences: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None
    ):
        """Logs a single routing decision and associated metrics."""
        decision_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query, # Be mindful of PII if queries are sensitive
            "task_category": task_classification.get("category"),
            "task_complexity": task_classification.get("details", {}).get("complexity_level"),
            "task_confidence": task_classification.get("confidence"),
            "selected_model": selected_model,
            "routing_reason": routing_reason,
            "user_preferences": user_preferences or {},
            "estimated_cost_usd": estimated_cost,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": error
        }
        self.routing_decisions.append(decision_entry)

        # Update model usage stats
        self.model_usage_counts[selected_model] += 1
        self.model_total_latency_ms[selected_model] += latency_ms
        if input_tokens is not None:
            self.model_input_tokens[selected_model] += input_tokens
        if output_tokens is not None:
            self.model_output_tokens[selected_model] += output_tokens
        if estimated_cost is not None:
             self.model_estimated_cost[selected_model] += estimated_cost


        if error:
            self.model_error_counts[selected_model] += 1

        # Update task classification stats
        cat = task_classification.get("category", "unknown")
        comp = task_classification.get("details", {}).get("complexity_level", "unknown")
        self.task_classification_counts[cat][comp] += 1
        if task_classification.get("confidence") is not None:
            self.task_classification_confidence_sum[cat] += task_classification.get("confidence", 0.0)
        self.task_classification_total_count +=1

        logger.debug(f"Logged routing decision for model {selected_model}. Latency: {latency_ms:.2f}ms. Cost: ${estimated_cost:.6f}")

    def record_cache_hit(self):
        self.routing_cache_hits += 1

    def record_cache_miss(self):
        self.routing_cache_misses += 1

    def get_routing_metrics(self) -> Dict[str, Any]:
        """Returns a summary of current routing metrics."""

        model_performance = []
        for model, count in self.model_usage_counts.items():
            avg_latency = (self.model_total_latency_ms[model] / count) if count > 0 else 0
            error_rate = (self.model_error_counts[model] / count) if count > 0 else 0
            avg_input_tokens = (self.model_input_tokens[model] / count) if count > 0 else 0
            avg_output_tokens = (self.model_output_tokens[model] / count) if count > 0 else 0
            total_cost = self.model_estimated_cost[model]
            avg_cost_per_req = (total_cost / count) if count > 0 else 0

            model_performance.append({
                "model_name": model,
                "usage_count": count,
                "total_latency_ms": self.model_total_latency_ms[model],
                "average_latency_ms": avg_latency,
                "error_count": self.model_error_counts[model],
                "error_rate": error_rate,
                "total_input_tokens": self.model_input_tokens[model],
                "average_input_tokens": avg_input_tokens,
                "total_output_tokens": self.model_output_tokens[model],
                "average_output_tokens": avg_output_tokens,
                "total_estimated_cost_usd": total_cost,
                "average_estimated_cost_per_request_usd": avg_cost_per_req,
            })

        overall_avg_classification_confidence = 0
        if self.task_classification_total_count > 0:
            total_confidence_sum = sum(self.task_classification_confidence_sum.values())
            overall_avg_classification_confidence = total_confidence_sum / self.task_classification_total_count

        cache_total_lookups = self.routing_cache_hits + self.routing_cache_misses
        cache_hit_rate = (self.routing_cache_hits / cache_total_lookups) if cache_total_lookups > 0 else 0

        return {
            "total_requests_routed": len(self.routing_decisions),
            "model_performance_summary": model_performance,
            "task_classification_distribution": self.task_classification_counts,
            "average_classification_confidence": overall_avg_classification_confidence,
            "routing_cache_hits": self.routing_cache_hits,
            "routing_cache_misses": self.routing_cache_misses,
            "routing_cache_hit_rate": cache_hit_rate,
            "last_n_routing_decisions": self.routing_decisions[-20:] # Return last 20 decisions for quick view
        }

    def reset_metrics(self):
        """Resets all accumulated metrics."""
        self.routing_decisions.clear()
        self.model_usage_counts.clear()
        self.model_total_latency_ms.clear()
        self.model_error_counts.clear()
        self.model_input_tokens.clear()
        self.model_output_tokens.clear()
        self.model_estimated_cost.clear()
        self.task_classification_counts.clear()
        self.task_classification_confidence_sum.clear()
        self.task_classification_total_count = 0
        self.routing_cache_hits = 0
        self.routing_cache_misses = 0
        logger.info("All monitoring metrics have been reset.")

# Global instance (or use dependency injection)
# For simplicity in this template, a global instance can be used.
# In a larger application, you might inject this into classes that need it.
MONITORING_SYSTEM_INSTANCE = MonitoringSystem()

def get_monitoring_system() -> MonitoringSystem:
    """Provides access to the global monitoring system instance."""
    return MONITORING_SYSTEM_INSTANCE

# Convenience function as per prompt's example `from core.monitoring import get_routing_metrics`
def get_routing_metrics() -> Dict[str, Any]:
    return MONITORING_SYSTEM_INSTANCE.get_routing_metrics()


if __name__ == '__main__':
    monitor = get_monitoring_system()

    # Simulate some routing decisions
    mock_classification1 = {"category": "coding", "confidence": 0.9, "details": {"complexity_level": "high"}}
    monitor.log_routing_decision("query1", mock_classification1, "gpt-4", "high complexity coding", 0.05, 1500.0, input_tokens=100, output_tokens=200)

    mock_classification2 = {"category": "analysis", "confidence": 0.8, "details": {"complexity_level": "medium"}}
    monitor.log_routing_decision("query2", mock_classification2, "claude-2", "data analysis", 0.02, 1200.0, input_tokens=50, output_tokens=300)

    mock_classification3 = {"category": "coding", "confidence": 0.95, "details": {"complexity_level": "high"}}
    monitor.log_routing_decision("query3", mock_classification3, "gpt-4", "high complexity coding", 0.06, 1800.0, error="API Timeout", input_tokens=120, output_tokens=50) # Error example

    monitor.record_cache_hit()
    monitor.record_cache_miss()
    monitor.record_cache_miss()

    metrics = monitor.get_routing_metrics()
    import json
    print(json.dumps(metrics, indent=2))

    print("\n--- Resetting metrics ---")
    monitor.reset_metrics()
    print(json.dumps(monitor.get_routing_metrics(), indent=2))
