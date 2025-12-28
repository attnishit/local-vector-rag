"""
Logging module for Local Vector RAG Database.

Author: Nishit Attrey

This module provides centralized logging configuration with both console
and rotating file handlers. All modules in the project should use the
logger provided by this module for consistent logging output.

Functions:
    setup_logging: Configure application-wide logging with handlers
    get_logger: Convenience function to get a logger for a module

Usage:
    # In main.py or application entry point:
    from src.logger import setup_logging
    from src.config import load_config

    config = load_config()
    logger = setup_logging(config)
    logger.info("Application started")

    # In other modules:
    import logging
    logger = logging.getLogger(__name__)
    logger.debug("Detailed diagnostic information")
    logger.info("Normal operation confirmation")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Configure application-wide logging with console and file handlers.

    This function sets up a comprehensive logging system with:
    - Console handler: INFO level and above, formatted for readability
    - Rotating file handler: DEBUG level and above, with automatic rotation
    - Configurable format and log levels from config

    The function is idempotent - calling it multiple times will not create
    duplicate handlers.

    Args:
        config: Configuration dictionary containing 'logging' section with:
            - level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            - format: Log message format string
            - console_output: Boolean, whether to log to console
            - file_output: Boolean, whether to log to file
            - log_file: Path to log file (created if doesn't exist)
            - max_bytes: Maximum log file size before rotation (bytes)
            - backup_count: Number of backup log files to keep

    Returns:
        Root logger configured for the application. All other loggers
        in the application will inherit these settings.

    Example:
        >>> config = load_config()
        >>> logger = setup_logging(config)
        >>> logger.info("Application initialized")
        2025-12-27 10:30:15,123 - root - INFO - Application initialized

    Note:
        - Console output uses INFO level by default for cleaner output
        - File output uses DEBUG level for comprehensive diagnostics
        - Log files rotate when they reach max_bytes size
        - Old log files are kept according to backup_count setting
    """
    log_config = config['logging']

    log_level_str = log_config.get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str)

    log_format = log_config.get(
        'format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    formatter = logging.Formatter(log_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    if log_config.get('console_output', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_config.get('file_output', True):
        log_file_path = Path(log_config.get('log_file', 'logs/rag.log'))
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        max_bytes = log_config.get('max_bytes', 10 * 1024 * 1024)
        backup_count = log_config.get('backup_count', 5)

        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )

        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    This is a convenience function that wraps logging.getLogger().
    Use this in individual modules to get a properly configured logger.

    Args:
        name: Name for the logger, typically __name__ of the calling module

    Returns:
        Logger instance configured with the application's settings

    Example:
        # In src/ingestion/chunker.py:
        from src.logger import get_logger

        logger = get_logger(__name__)
        logger.info("Processing documents...")
        # Output: 2025-12-27 10:30:15 - src.ingestion.chunker - INFO - Processing documents...

    Note:
        You must call setup_logging() before using get_logger() in other modules,
        otherwise you'll get the default Python logging configuration.
    """
    return logging.getLogger(name)


def log_config_info(config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Log configuration information at startup.

    This is a utility function to log key configuration values when
    the application starts. Useful for debugging and verification.

    Args:
        config: Configuration dictionary
        logger: Logger instance to use for output

    Example:
        >>> config = load_config()
        >>> logger = setup_logging(config)
        >>> log_config_info(config, logger)
        # Logs various configuration details
    """
    logger.debug("Configuration details:")
    logger.debug(f"  Project: {config['project']['name']} v{config['project']['version']}")
    logger.debug(f"  Log level: {config['logging']['level']}")
    logger.debug(f"  Log file: {config['logging'].get('log_file', 'N/A')}")
    logger.debug(f"  Data directory: {config['paths']['data_dir']}")
    logger.debug(f"  Raw data: {config['paths']['raw_dir']}")
    logger.debug(f"  Processed data: {config['paths']['processed_dir']}")
    logger.debug(f"  Embeddings: {config['paths']['embeddings_dir']}")
