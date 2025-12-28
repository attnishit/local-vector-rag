"""
Configuration module for Local Vector RAG Database.

Author: Nishit Attrey

This module handles loading, validation, and management of configuration
from the YAML config file. It also creates necessary directories on startup.

Functions:
    load_config: Load and validate configuration from YAML file
    validate_config: Validate configuration structure and values
    create_directories: Create required directories if they don't exist
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any


# Valid logging levels for validation
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.

    This function:
    1. Loads the YAML configuration file
    2. Validates the configuration structure and values
    3. Creates necessary directories
    4. Returns the validated configuration dictionary

    Args:
        config_path: Path to the YAML configuration file.
                     Defaults to "config.yaml" in the project root.

    Returns:
        Dictionary containing all configuration values with the structure:
        {
            'project': {'name': str, 'version': str},
            'logging': {...},
            'paths': {...}
        }

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
        ValueError: If configuration validation fails

    Example:
        >>> config = load_config()
        >>> print(config['project']['name'])
        'Local Vector RAG'
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please ensure config.yaml exists in the project root."
        )

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Failed to parse YAML configuration file: {config_path}\n"
            f"Error: {e}"
        )

    validate_config(config)
    create_directories(config)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and values.

    Checks that all required sections and keys exist, and that values
    are of the correct type and within valid ranges.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If any required section/key is missing or invalid

    Validation Rules:
        - 'project' section must exist with 'name' and 'version'
        - 'logging' section must exist with valid level and settings
        - 'paths' section must exist with all required paths
        - logging.level must be a valid Python logging level
        - logging.max_bytes and backup_count must be positive integers
    """
    # Validate top-level sections exist
    required_sections = ['project', 'logging', 'paths']
    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required configuration section: '{section}'\n"
                f"Please ensure config.yaml contains all required sections."
            )

    project = config['project']
    if 'name' not in project or not project['name']:
        raise ValueError("Missing or empty 'project.name' in configuration")
    if 'version' not in project or not project['version']:
        raise ValueError("Missing or empty 'project.version' in configuration")

    logging_config = config['logging']

    if 'level' not in logging_config:
        raise ValueError("Missing 'logging.level' in configuration")

    log_level = logging_config['level'].upper()
    if log_level not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid logging level: '{logging_config['level']}'\n"
            f"Must be one of: {', '.join(sorted(VALID_LOG_LEVELS))}"
        )

    if 'max_bytes' in logging_config:
        max_bytes = logging_config['max_bytes']
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError(
                f"logging.max_bytes must be a positive integer, got: {max_bytes}"
            )

    if 'backup_count' in logging_config:
        backup_count = logging_config['backup_count']
        if not isinstance(backup_count, int) or backup_count < 0:
            raise ValueError(
                f"logging.backup_count must be a non-negative integer, got: {backup_count}"
            )

    # Validate 'paths' section - just check all required paths are present
    paths = config['paths']
    required_paths = ['data_dir', 'raw_dir', 'processed_dir', 'embeddings_dir', 'logs_dir']

    for path_key in required_paths:
        if path_key not in paths or not paths[path_key]:
            raise ValueError(
                f"Missing or empty 'paths.{path_key}' in configuration"
            )


def create_directories(config: Dict[str, Any]) -> None:
    """
    Create required directories if they don't exist.

    Creates all directories specified in the 'paths' section of the config.
    If a directory already exists, it is left unchanged.

    Args:
        config: Configuration dictionary containing 'paths' section

    Note:
        This function is safe to call multiple times - existing directories
        are not modified or recreated.

    Example:
        >>> config = {'paths': {'data_dir': 'data', 'logs_dir': 'logs'}}
        >>> create_directories(config)
        # Creates 'data/' and 'logs/' directories
    """
    paths = config['paths']

    # Note: We create all paths from the config to ensure consistency
    directories_to_create = [
        paths['data_dir'],
        paths['raw_dir'],
        paths['processed_dir'],
        paths['embeddings_dir'],
        paths['logs_dir'],
    ]

    for dir_path in directories_to_create:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)


# Module-level convenience function for getting specific config values
def get_project_info(config: Dict[str, Any]) -> tuple[str, str]:
    """
    Extract project name and version from config.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (project_name, project_version)

    Example:
        >>> config = load_config()
        >>> name, version = get_project_info(config)
        >>> print(f"{name} v{version}")
        'Local Vector RAG v0.1.0-stage1'
    """
    return config['project']['name'], config['project']['version']
