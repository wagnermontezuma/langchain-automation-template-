from enum import Enum
from dataclasses import dataclass

class ErrorType(Enum):
    CONFIG_ERROR = "CONFIG_ERROR"
    API_ERROR = "API_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    INITIALIZATION_ERROR = "INITIALIZATION_ERROR"

@dataclass
class ErrorDetails:
    error_type: ErrorType
    message: str
    details: str = "" # Optional additional details

class LangchainAutomationError(Exception):
    """Base class for custom exceptions in this project."""
    def __init__(self, error_details: ErrorDetails, original_exception: Exception = None):
        super().__init__(f"[{error_details.error_type.value}] {error_details.message}")
        self.error_details = error_details
        self.original_exception = original_exception

    def __str__(self):
        base_message = f"[{self.error_details.error_type.value}] {self.error_details.message}"
        if self.error_details.details:
            base_message += f" (Details: {self.error_details.details})"
        if self.original_exception:
            base_message += f"\nOriginal Exception: {type(self.original_exception).__name__}: {str(self.original_exception)}"
        return base_message

class ConfigError(LangchainAutomationError):
    """Exception raised for configuration-related errors."""
    def __init__(self, message: str, details: str = "", original_exception: Exception = None):
        super().__init__(ErrorDetails(ErrorType.CONFIG_ERROR, message, details), original_exception)

class APIError(LangchainAutomationError):
    """Exception raised for API interaction errors."""
    def __init__(self, message: str, details: str = "", original_exception: Exception = None):
        super().__init__(ErrorDetails(ErrorType.API_ERROR, message, details), original_exception)

class ValidationError(LangchainAutomationError):
    """Exception raised for data validation errors."""
    def __init__(self, message: str, details: str = "", original_exception: Exception = None):
        super().__init__(ErrorDetails(ErrorType.VALIDATION_ERROR, message, details), original_exception)

class RuntimeError(LangchainAutomationError):
    """Exception raised for general runtime errors."""
    def __init__(self, message: str, details: str = "", original_exception: Exception = None):
        super().__init__(ErrorDetails(ErrorType.RUNTIME_ERROR, message, details), original_exception)

class NetworkError(LangchainAutomationError):
    """Exception raised for network-related errors."""
    def __init__(self, message: str, details: str = "", original_exception: Exception = None):
        super().__init__(ErrorDetails(ErrorType.NETWORK_ERROR, message, details), original_exception)

class UnknownError(LangchainAutomationError):
    """Exception raised for unknown or unexpected errors."""
    def __init__(self, message: str, details: str = "", original_exception: Exception = None):
        super().__init__(ErrorDetails(ErrorType.UNKNOWN_ERROR, message, details), original_exception)

class ToolError(LangchainAutomationError):
    """Exception raised for errors occurring within a tool's execution."""
    def __init__(self, message: str, details: str = "", original_exception: Exception = None):
        super().__init__(ErrorDetails(ErrorType.TOOL_ERROR, message, details), original_exception)

class InitializationError(LangchainAutomationError):
    """Exception raised for errors during object or system initialization."""
    def __init__(self, message: str, details: str = "", original_exception: Exception = None):
        super().__init__(ErrorDetails(ErrorType.INITIALIZATION_ERROR, message, details), original_exception)

# Example usage (optional, for testing within the file if run directly)
if __name__ == '__main__':
    try:
        raise ConfigError("Failed to load model configuration", details="File not found: models.yaml")
    except LangchainAutomationError as e:
        print(e)

    try:
        raise APIError("OpenAI API returned 401 Unauthorized", details="Invalid API key provided", original_exception=ValueError("Original error"))
    except LangchainAutomationError as e:
        print(e)
