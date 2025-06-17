from typing import TypedDict, List, Optional, Dict, Any

class ModelConfig(TypedDict):
    name: str              # User-friendly name for the model (e.g., "GPT-4 Turbo")
    provider: str          # Name of the provider (e.g., "OpenAI", "Anthropic", "Google")
    api_key_env: str       # Environment variable name for the API key
    base_url: Optional[str] # Base URL for the API (if not default)
    model_name: str        # Specific model identifier (e.g., "gpt-4-turbo-preview", "claude-3-opus-20240229")
    temperature: float
    max_tokens: int
    context_window: Optional[int] # Optional context window size
    default_params: Optional[Dict[str, Any]] # Optional default parameters for the model

class AgentConfig(TypedDict):
    name: str
    description: str
    model_config_name: str # Name of the ModelConfig to use (references key in models.yaml)
    tools: List[str]       # List of tool names available to the agent
    system_message_prompt: str # Path to a file containing the system message or the message itself
    max_iterations: int
    parallel_tool_calls: bool

class TaskClassification(TypedDict):
    task_description: str
    category: str          # e.g., "code_generation", "text_summarization", "question_answering"
    confidence: float      # Confidence score for the classification (0.0 to 1.0)
    details: Optional[Dict[str, Any]] # Any additional details about the classification

class ToolCallRequest(TypedDict):
    tool_name: str
    tool_input: Dict[str, Any]

class ToolCallResult(TypedDict):
    tool_name: str
    tool_output: Any
    error_message: Optional[str] # If the tool execution failed

class ChatMessage(TypedDict):
    role: str # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] # tool_call_id for tool messages

# Example usage (optional)
if __name__ == '__main__':
    gpt4_config: ModelConfig = {
        "name": "GPT-4 Turbo",
        "provider": "OpenAI",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "model_name": "gpt-4-turbo-preview",
        "temperature": 0.7,
        "max_tokens": 1000,
        "context_window": 128000,
        "default_params": {"top_p": 0.9}
    }
    print(f"Example ModelConfig: {gpt4_config}")

    coder_agent_config: AgentConfig = {
        "name": "CodeGenerationAgent",
        "description": "An agent specialized in generating code snippets.",
        "model_config_name": "Deepseek Coder",
        "tools": ["file_writer_tool", "code_executor_tool"],
        "system_message_prompt": "You are a helpful coding assistant.",
        "max_iterations": 5,
        "parallel_tool_calls": True,
    }
    print(f"Example AgentConfig: {coder_agent_config}")

    task_example: TaskClassification = {
        "task_description": "Write a python function to calculate factorial.",
        "category": "code_generation",
        "confidence": 0.95,
        "details": {"language": "python"}
    }
    print(f"Example TaskClassification: {task_example}")
