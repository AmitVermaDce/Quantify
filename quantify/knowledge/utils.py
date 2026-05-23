"""
Utility functions and classes for the Knowledge Base pipeline.
Provides logging, error handling, and common helpers.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def setup_logging(
    name: str = "knowledge-base",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger with console and optional file output.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file with environment variable overrides.

    Priority:
    1. Environment variables (highest priority)
    2. Config file
    3. Defaults (lowest priority)

    Args:
        config_path: Path to config.json file

    Returns:
        Merged configuration dictionary
    """
    defaults = {
        "llm": {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "qwen3.5:cloud",
            "embedding_model": "mxbai-embed-large",
            "embedding_dimension": 1024,
        },
        "database": {
            "url": "postgresql://localhost:5432/trading_knowledge",
            "pool_size": 10,
            "max_overflow": 20,
        },
        "extraction": {
            "chunk_size": 2000,
            "chunk_overlap": 200,
            "retry_attempts": 3,
            "retry_delay_seconds": 10,
        },
        "api": {
            "port": 3000,
            "host": "localhost",
            "log_level": "info",
        },
        "logging": {
            "level": "INFO",
            "file": "logs/knowledge-base.log",
        },
    }

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            file_config = json.load(f)
            defaults = _merge_configs(defaults, file_config)

    env_config = _load_env_config()
    return _merge_configs(defaults, env_config)


def _merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge configuration dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        elif value is not None:
            result[key] = value
    return result


def _load_env_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {}

    if os.getenv("OLLAMA_BASE_URL"):
        config.setdefault("llm", {})["base_url"] = os.getenv("OLLAMA_BASE_URL")
    if os.getenv("OLLAMA_MODEL"):
        config.setdefault("llm", {})["model"] = os.getenv("OLLAMA_MODEL")
    if os.getenv("EMBEDDING_MODEL"):
        config.setdefault("llm", {})["embedding_model"] = os.getenv("EMBEDDING_MODEL")
    if os.getenv("EMBEDDING_DIMENSION"):
        config.setdefault("llm", {})["embedding_dimension"] = int(os.getenv("EMBEDDING_DIMENSION"))

    if os.getenv("DATABASE_URL"):
        config.setdefault("database", {})["url"] = os.getenv("DATABASE_URL")
    if os.getenv("DATABASE_POOL_SIZE"):
        config.setdefault("database", {})["pool_size"] = int(os.getenv("DATABASE_POOL_SIZE"))

    if os.getenv("CHUNK_SIZE"):
        config.setdefault("extraction", {})["chunk_size"] = int(os.getenv("CHUNK_SIZE"))
    if os.getenv("CHUNK_OVERLAP"):
        config.setdefault("extraction", {})["chunk_overlap"] = int(os.getenv("CHUNK_OVERLAP"))
    if os.getenv("RETRY_ATTEMPTS"):
        config.setdefault("extraction", {})["retry_attempts"] = int(os.getenv("RETRY_ATTEMPTS"))
    if os.getenv("RETRY_DELAY_SECONDS"):
        config.setdefault("extraction", {})["retry_delay_seconds"] = int(os.getenv("RETRY_DELAY_SECONDS"))

    if os.getenv("API_PORT"):
        config.setdefault("api", {})["port"] = int(os.getenv("API_PORT"))
    if os.getenv("API_HOST"):
        config.setdefault("api", {})["host"] = os.getenv("API_HOST")
    if os.getenv("LOG_LEVEL"):
        config.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")
    if os.getenv("LOG_FILE"):
        config.setdefault("logging", {})["file"] = os.getenv("LOG_FILE")

    return config


class ExtractionError(Exception):
    """Custom exception for extraction pipeline errors."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


def retry_with_backoff(
    func,
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        delay_seconds: Initial delay between attempts
        backoff_multiplier: Multiplier for delay after each attempt
        exceptions: Tuple of exceptions to catch

    Returns:
        Function result or raises ExtractionError after max attempts

    Example:
        result = retry_with_backoff(
            lambda: api_call(),
            max_attempts=3,
            delay_seconds=10,
            exceptions=(RequestError,)
        )
    """
    import time

    last_exception = None
    delay = delay_seconds

    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts:
                logging.getLogger(__name__).warning(
                    f"Attempt {attempt}/{max_attempts} failed: {type(e).__name__}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= backoff_multiplier
            else:
                logging.getLogger(__name__).error(
                    f"All {max_attempts} attempts failed. Last error: {e}"
                )

    raise ExtractionError(
        f"Operation failed after {max_attempts} attempts",
        {"last_error": str(last_exception), "function": func.__name__},
    )


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def ensure_directory(path: str) -> Path:
    """Ensure directory exists, create if necessary."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
