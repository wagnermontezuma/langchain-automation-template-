# langchain-automation-template/core/__init__.py
from .exceptions import *
from .types import *
from .logger import get_logger
from .openrouter_client import OpenRouterClient
from .retry_decorator import retry_with_exponential_backoff # Ensure this line is present and correct

__all__ = [
    # exceptions
    "ErrorType",
    "ErrorDetails",
    "LangchainAutomationError",
    "ConfigError",
    "APIError",
    "ValidationError",
    "RuntimeError",
    "NetworkError",
    "UnknownError",
    "ToolError",
    "InitializationError",
    # types
    "ModelConfig",
    "AgentConfig",
    "TaskClassification",
    "ToolCallRequest",
    "ToolCallResult",
    "ChatMessage",
    # logger
    "get_logger",
    # client
    "OpenRouterClient",
    # retry_decorator
    "retry_with_exponential_backoff", # Ensure this is exported
]
