import asyncio
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    AsyncRetrying,
    RetryError
)
from .exceptions import NetworkError, APIError # Assuming these are defined in your exceptions module
from .logger import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 10.0    # seconds

def retry_with_exponential_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retry_on_exceptions: tuple = (NetworkError, APIError, asyncio.TimeoutError) # Default exceptions to retry on
):
    """
    A decorator factory that creates a retry decorator with exponential backoff.

    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_delay (float): The initial delay between retries in seconds.
        max_delay (float): The maximum delay between retries in seconds.
        retry_on_exceptions (tuple): A tuple of exception types to retry on.
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                retryer = AsyncRetrying(
                    stop=stop_after_attempt(max_retries),
                    wait=wait_exponential(multiplier=initial_delay, max=max_delay),
                    retry=retry_if_exception_type(retry_on_exceptions),
                    reraise=True,
                )
                try:
                    return await retryer.call(func, *args, **kwargs)
                except RetryError as e: # Tenacity wraps the last attempt's exception in RetryError
                    logger.error(f"All retries failed for {func.__name__}. Last exception: {e.last_attempt.exception}")
                    raise e.last_attempt.exception # Reraise the original exception
                except Exception as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                retryer = retry(
                    stop=stop_after_attempt(max_retries),
                    wait=wait_exponential(multiplier=initial_delay, max=max_delay),
                    retry=retry_if_exception_type(retry_on_exceptions),
                    reraise=True,
                )
                try:
                    return retryer(func)(*args, **kwargs)
                except RetryError as e:
                    logger.error(f"All retries failed for {func.__name__}. Last exception: {e.last_attempt.exception}")
                    raise e.last_attempt.exception
                except Exception as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            return sync_wrapper
    return decorator

if __name__ == '__main__':
    # Example Usage (requires .env for OPENROUTER_API_KEY for the client part)

    # --- Synchronous Example ---
    @retry_with_exponential_backoff(max_retries=3, initial_delay=0.1, max_delay=1)
    def might_fail_sync(fail_times):
        if might_fail_sync.attempts < fail_times:
            might_fail_sync.attempts += 1
            logger.info(f"Sync function: Attempt {might_fail_sync.attempts}, failing...")
            raise NetworkError("Simulated sync network error")
        logger.info("Sync function: Succeeded!")
        return "Sync success"
    might_fail_sync.attempts = 0

    print("Testing synchronous retry decorator...")
    try:
        result = might_fail_sync(2) # Fail 2 times, succeed on 3rd
        print(f"Sync result: {result}")
    except Exception as e:
        print(f"Sync test caught: {type(e).__name__}: {e}")

    might_fail_sync.attempts = 0
    try:
        print("\nTesting synchronous retry decorator (expected to fail all retries)...")
        might_fail_sync(4) # Fail 4 times, more than max_retries
    except Exception as e:
        print(f"Sync test (expected fail) caught: {type(e).__name__}: {e}")

    # --- Asynchronous Example ---
    @retry_with_exponential_backoff(max_retries=3, initial_delay=0.1, max_delay=1)
    async def might_fail_async(fail_times):
        if might_fail_async.attempts < fail_times:
            might_fail_async.attempts += 1
            logger.info(f"Async function: Attempt {might_fail_async.attempts}, failing...")
            await asyncio.sleep(0.01) # simulate async work
            raise APIError("Simulated async API error")
        logger.info("Async function: Succeeded!")
        await asyncio.sleep(0.01)
        return "Async success"
    might_fail_async.attempts = 0

    async def run_async_tests():
        print("\nTesting asynchronous retry decorator...")
        try:
            result = await might_fail_async(2)
            print(f"Async result: {result}")
        except Exception as e:
            print(f"Async test caught: {type(e).__name__}: {e}")

        might_fail_async.attempts = 0
        try:
            print("\nTesting asynchronous retry decorator (expected to fail all retries)...")
            await might_fail_async(4)
        except Exception as e:
            print(f"Async test (expected fail) caught: {type(e).__name__}: {e}")

    asyncio.run(run_async_tests())

    # Note: The OpenRouterClient itself uses tenacity directly in the provided snippet.
    # This retry_decorator.py is a more generic one that could be used elsewhere
    # or to refactor OpenRouterClient if desired.
    print("\nNote: OpenRouterClient in the prompt uses tenacity directly, not this decorator.")
    print("This retry_decorator.py provides a generic reusable decorator.")
