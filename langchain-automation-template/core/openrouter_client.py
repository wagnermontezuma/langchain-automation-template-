import os
import httpx
from typing import Any, Dict, Optional, AsyncGenerator, Generator, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

from .exceptions import APIError, NetworkError, ConfigError, InitializationError
from .types import ChatMessage # Assuming ChatMessage is defined in .types
from .logger import get_logger

# Load environment variables from .env file
load_dotenv()
logger = get_logger(__name__)

# Default OpenRouter API URL
OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"

class OpenRouterClient:
    """
    A client for interacting with the OpenRouter API, supporting synchronous and asynchronous requests.
    Handles retries for common transient errors.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_key_env_var: Optional[str] = "OPENROUTER_API_KEY",
        base_url: Optional[str] = None,
        timeout: float = 60.0, # seconds
        max_retries: int = 3,
        initial_retry_delay: float = 1.0, # seconds
        max_retry_delay: float = 10.0 # seconds
    ):
        """
        Initializes the OpenRouterClient.

        Args:
            model_name (str): The specific model to use (e.g., "openai/gpt-4-turbo").
            api_key (Optional[str]): The OpenRouter API key. If None, tries to load from `api_key_env_var`.
            api_key_env_var (Optional[str]): The environment variable to load the API key from.
            base_url (Optional[str]): The base URL for the OpenRouter API. Defaults to official URL.
            timeout (float): Default timeout for HTTP requests.
            max_retries (int): Maximum number of retries for failed requests.
            initial_retry_delay (float): Initial delay for retries in seconds.
            max_retry_delay (float): Maximum delay for retries in seconds.

        Raises:
            InitializationError: If the API key is not provided or found in environment variables.
        """
        if api_key:
            self.api_key = api_key
        elif api_key_env_var and os.getenv(api_key_env_var):
            self.api_key = os.getenv(api_key_env_var)
        else:
            raise InitializationError(
                f"OpenRouter API key not provided and not found in environment variable '{api_key_env_var}'. "
                "Please set the API key directly or via the environment variable."
            )

        self.model_name = model_name
        self.base_url = base_url or OPENROUTER_API_BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Recommended headers by OpenRouter
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:3000"), # Replace with your actual site URL
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "LangchainAutomationTemplate"), # Replace with your actual app name
        }

        # Configure retry decorator
        self.retry_decorator = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.initial_retry_delay, max=self.max_retry_delay),
            retry=retry_if_exception_type((NetworkError, APIError)), # Retry on specific custom exceptions
            reraise=True # Reraise the exception if all retries fail
        )

        # Apply decorator to methods that make HTTP requests
        self.chat_completion = self.retry_decorator(self._chat_completion)
        self.achat_completion = self.retry_decorator(self._achat_completion)
        self.chat_completion_stream = self.retry_decorator(self._chat_completion_stream)
        self.achat_completion_stream = self.retry_decorator(self._achat_completion_stream)


    def _prepare_payload(self, messages: List[ChatMessage], stream: bool = False, **kwargs) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            **kwargs # Allow overriding temperature, max_tokens, etc.
        }
        logger.debug(f"Request payload for {self.model_name}: {payload}")
        return payload

    def _handle_response_error(self, response: httpx.Response):
        """Handles HTTP errors from the API response."""
        try:
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
        except httpx.HTTPStatusError as e:
            error_content = e.response.text
            status_code = e.response.status_code
            logger.error(f"API Error ({status_code}): {error_content} for request to {e.request.url}")
            if 400 <= status_code < 500:
                raise APIError(
                    f"Client error {status_code} calling OpenRouter: {error_content}",
                    details=f"URL: {e.request.url}",
                    original_exception=e
                )
            elif 500 <= status_code < 600:
                raise NetworkError( # Or a more specific ServerError if preferred
                    f"Server error {status_code} from OpenRouter: {error_content}",
                    details=f"URL: {e.request.url}",
                    original_exception=e
                )
            else: # Should not happen with raise_for_status but as a fallback
                raise APIError(
                    f"Unhandled HTTP error {status_code} from OpenRouter: {error_content}",
                    details=f"URL: {e.request.url}",
                    original_exception=e
                )
        except httpx.RequestError as e: # Handles network errors like DNS failure, refused connection
            logger.error(f"Network Error: {e} for request to {e.request.url}")
            raise NetworkError(
                f"Network error occurred while requesting OpenRouter: {str(e)}",
                details=f"URL: {e.request.url}",
                original_exception=e
            )

    def _chat_completion(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        """
        Internal synchronous chat completion method (decorated with retry).
        """
        payload = self._prepare_payload(messages, stream=False, **kwargs)
        request_url = f"{self.base_url}/chat/completions"

        logger.info(f"Sending chat completion request to {self.model_name} via {request_url}")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(request_url, headers=self.headers, json=payload)
            self._handle_response_error(response)
            logger.debug(f"Response from {self.model_name}: {response.json()}")
            return response.json()
        except Exception as e: # Catch any other exception to wrap it if not already custom
            if not isinstance(e, (APIError, NetworkError, ConfigError, InitializationError)):
                logger.error(f"Unexpected error during chat completion: {e}", exc_info=True)
                raise APIError("Unexpected error during synchronous chat completion.", original_exception=e)
            raise # Reraise custom exceptions

    async def _achat_completion(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        """
        Internal asynchronous chat completion method (decorated with retry).
        """
        payload = self._prepare_payload(messages, stream=False, **kwargs)
        request_url = f"{self.base_url}/chat/completions"

        logger.info(f"Sending async chat completion request to {self.model_name} via {request_url}")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(request_url, headers=self.headers, json=payload)
            self._handle_response_error(response)
            logger.debug(f"Async response from {self.model_name}: {response.json()}")
            return response.json()
        except Exception as e:
            if not isinstance(e, (APIError, NetworkError, ConfigError, InitializationError)):
                logger.error(f"Unexpected error during async chat completion: {e}", exc_info=True)
                raise APIError("Unexpected error during asynchronous chat completion.", original_exception=e)
            raise

    def _chat_completion_stream(
        self, messages: List[ChatMessage], **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Internal synchronous streaming chat completion method (decorated with retry).
        Yields completion chunks.
        """
        payload = self._prepare_payload(messages, stream=True, **kwargs)
        request_url = f"{self.base_url}/chat/completions"

        logger.info(f"Sending streaming chat completion request to {self.model_name} via {request_url}")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", request_url, headers=self.headers, json=payload) as response:
                    self._handle_response_error(response) # Check initial response
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            line_data = line[len("data: "):]
                            if line_data.strip() == "[DONE]":
                                logger.debug("Stream [DONE] received.")
                                break
                            if line_data:
                                try:
                                    chunk = httpx.Response(content=line_data).json() # Using httpx.Response to parse json for consistency
                                    logger.debug(f"Stream chunk from {self.model_name}: {chunk}")
                                    yield chunk
                                except Exception as e: # Handle potential JSON decoding errors in stream
                                    logger.warning(f"Could not decode stream chunk: {line_data}, error: {e}")
                                    continue
        except Exception as e:
            if not isinstance(e, (APIError, NetworkError, ConfigError, InitializationError)):
                logger.error(f"Unexpected error during streaming chat completion: {e}", exc_info=True)
                raise APIError("Unexpected error during synchronous streaming chat completion.", original_exception=e)
            raise

    async def _achat_completion_stream(
        self, messages: List[ChatMessage], **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Internal asynchronous streaming chat completion method (decorated with retry).
        Yields completion chunks.
        """
        payload = self._prepare_payload(messages, stream=True, **kwargs)
        request_url = f"{self.base_url}/chat/completions"

        logger.info(f"Sending async streaming chat completion request to {self.model_name} via {request_url}")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", request_url, headers=self.headers, json=payload) as response:
                    self._handle_response_error(response) # Check initial response
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            line_data = line[len("data: "):]
                            if line_data.strip() == "[DONE]":
                                logger.debug("Async Stream [DONE] received.")
                                break
                            if line_data:
                                try:
                                    chunk = httpx.Response(content=line_data).json()
                                    logger.debug(f"Async stream chunk from {self.model_name}: {chunk}")
                                    yield chunk
                                except Exception as e:
                                    logger.warning(f"Could not decode async stream chunk: {line_data}, error: {e}")
                                    continue
        except Exception as e:
            if not isinstance(e, (APIError, NetworkError, ConfigError, InitializationError)):
                logger.error(f"Unexpected error during async streaming chat completion: {e}", exc_info=True)
                raise APIError("Unexpected error during asynchronous streaming chat completion.", original_exception=e)
            raise

# Example Usage (for direct testing of this file)
async def main_async():
    logger.info("Starting async OpenRouterClient test.")
    # Ensure OPENROUTER_API_KEY is set in your .env file or environment
    # And that the model you pick is one you have access to via OpenRouter
    # For example, use a free model like "mistralai/mistral-7b-instruct"

    # Check if API key is present
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not found. Skipping async test.")
        print("Skipping async test: OPENROUTER_API_KEY not found in environment.")
        return

    try:
        # Use a commonly available free model for testing
        client = OpenRouterClient(model_name="mistralai/mistral-7b-instruct")
                                 # api_key_env_var="YOUR_SPECIFIC_ENV_VAR_IF_DIFFERENT")

        messages: List[ChatMessage] = [
            {"role": "system", "content": "You are a helpful assistant.", "name": None},
            {"role": "user", "content": "Hello! What is the capital of France?", "name": None}
        ]

        # Test async chat completion
        logger.info("Testing async chat completion...")
        response = await client.achat_completion(messages=messages, temperature=0.7, max_tokens=50)
        logger.info(f"Async Chat Completion Response: {response['choices'][0]['message']['content']}")
        print(f"Async Response: {response['choices'][0]['message']['content']}")

        # Test async streaming chat completion
        logger.info("Testing async streaming chat completion...")
        print("Async Streaming Response:")
        async for chunk in client.achat_completion_stream(messages=messages, temperature=0.7, max_tokens=50):
            content_chunk = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if content_chunk:
                print(content_chunk, end="", flush=True)
        print("\nAsync stream finished.")
        logger.info("Async streaming finished.")

    except InitializationError as e:
        logger.error(f"Initialization Error: {e}")
        print(f"Initialization Error: {e}")
    except APIError as e:
        logger.error(f"API Error: {e}")
        print(f"API Error: {e}")
    except NetworkError as e:
        logger.error(f"Network Error: {e}")
        print(f"Network Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")

def main_sync():
    logger.info("Starting sync OpenRouterClient test.")
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not found. Skipping sync test.")
        print("Skipping sync test: OPENROUTER_API_KEY not found in environment.")
        return

    try:
        client = OpenRouterClient(model_name="mistralai/mistral-7b-instruct")

        messages: List[ChatMessage] = [
            {"role": "system", "content": "You are a helpful assistant.", "name": None},
            {"role": "user", "content": "Hi! What's the tallest mountain in the world?", "name": None}
        ]

        # Test sync chat completion
        logger.info("Testing sync chat completion...")
        response = client.chat_completion(messages=messages, temperature=0.7, max_tokens=60)
        logger.info(f"Sync Chat Completion Response: {response['choices'][0]['message']['content']}")
        print(f"Sync Response: {response['choices'][0]['message']['content']}")

        # Test sync streaming chat completion
        logger.info("Testing sync streaming chat completion...")
        print("Sync Streaming Response:")
        for chunk in client.chat_completion_stream(messages=messages, temperature=0.7, max_tokens=60):
            content_chunk = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if content_chunk:
                print(content_chunk, end="", flush=True)
        print("\nSync stream finished.")
        logger.info("Sync streaming finished.")

    except InitializationError as e:
        logger.error(f"Initialization Error: {e}")
        print(f"Initialization Error: {e}")
    except APIError as e:
        logger.error(f"API Error: {e}")
        print(f"API Error: {e}")
    except NetworkError as e:
        logger.error(f"Network Error: {e}")
        print(f"Network Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    import asyncio
    # Note: To run these tests, you need an OPENROUTER_API_KEY set in your environment.
    # You might also want to create a .env file in the root of the project with:
    # OPENROUTER_API_KEY="your_key_here"
    # OPENROUTER_SITE_URL="http://localhost:3000" (optional)
    # OPENROUTER_APP_NAME="MyTestApp" (optional)

    print("Running synchronous tests...")
    main_sync()

    print("\nRunning asynchronous tests...")
    asyncio.run(main_async())
