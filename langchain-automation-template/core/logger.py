import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# Define log format
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configure a stream handler for console output
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
stream_handler.setLevel(logging.INFO) # Default level for console

# Configure a rotating file handler for file output
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"),
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
file_handler.setLevel(logging.DEBUG) # Default level for file

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger instance.
    Args:
        name (str): Name of the logger, typically __name__ of the calling module.
        level (int): Logging level for the logger instance.
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Set the overall level for the logger.
    # If a handler's level is lower than this, it won't process those messages.
    logger.setLevel(min(level, file_handler.level, stream_handler.level if logger.propagate else logging.CRITICAL))

    # Add handlers if they haven't been added already to this specific logger
    # (to avoid duplicate logs if get_logger is called multiple times with the same name)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(stream_handler)

    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.addHandler(file_handler)

    # Prevent messages from being passed to the root logger if handlers are added
    logger.propagate = False

    return logger

# Example of a root logger configuration (optional, can be set up here or in main script)
# logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[stream_handler, file_handler])

if __name__ == '__main__':
    # Test the logger
    logger_test_module = get_logger(__name__, level=logging.DEBUG)

    logger_test_module.debug("This is a debug message.")
    logger_test_module.info("This is an info message.")
    logger_test_module.warning("This is a warning message.")
    logger_test_module.error("This is an error message.")
    logger_test_module.critical("This is a critical message.")

    # Test another logger to ensure they are distinct but use same handlers
    another_logger = get_logger("another_module", level=logging.INFO)
    another_logger.info("Info message from another_module.")
    another_logger.debug("This debug message from another_module should not appear on console if console is INFO.")

    print(f"Log file created at: {os.path.join(LOG_DIR, 'app.log')}")
