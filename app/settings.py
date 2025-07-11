import json
import logging
import os
from typing import Any

logger = logging.getLogger("kids-story-app")


class AppConfig:
    """Application configuration management"""

    # Default configuration values
    _defaults = {
        "openai_key": None,  # Should be set via environment variable
        "google_api_key": None,  # Should be set via environment variable for Gemini
        "max_story_length": 500,
        "min_story_length": 300,
        "max_illustrations": 4,
        "default_voice": "nova",
        "s3_bucket_name": "kids-story-assets",
        "content_safety_level": "strict",
        "default_temperature": 0.7,
        "openai_ssm_param_name": "OPEN_AI_KEY",  # Default SSM parameter name
        "google_ssm_param_name": "GOOGLE_API_KEY",  # Default SSM parameter name for Gemini
    }

    # Cache for config values
    _config_cache = None

    @classmethod
    def get_value(cls, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        Checks environment variables first, then config file, then defaults.
        """
        # Check environment variables (with KIDSTORY_ prefix)
        env_key = f"KIDSTORY_{key.upper()}"
        if env_key in os.environ:
            return os.environ[env_key]

        # Load config if not already loaded
        if cls._config_cache is None:
            cls._load_config()

        # Check config cache
        if key in cls._config_cache:
            return cls._config_cache[key]

        # Check defaults
        if key in cls._defaults:
            return cls._defaults[key]

        # Return provided default or None
        return default

    @classmethod
    def get_openai_api_key(cls) -> str:
        """Get OpenAI API key from environment or config"""
        # Check direct environment variable first
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return key
        
        # Fall back to config system
        return cls.get_value("openai_key")
    
    @classmethod
    def get_google_api_key(cls) -> str:
        """Get Google API key from environment or config"""
        # Check direct environment variable first  
        key = os.environ.get("GOOGLE_API_KEY")
        if key:
            return key
        
        # Fall back to config system
        return cls.get_value("google_api_key")

    @classmethod
    def _load_config(cls) -> None:
        """Load configuration from file"""
        cls._config_cache = {}

        # Determine config file location
        config_path = os.environ.get(
            "KIDSTORY_CONFIG_PATH", "/opt/python/config/app_config.json"
        )

        # For local development, check current directory
        if not os.path.exists(config_path):
            local_config = "./config.json"
            if os.path.exists(local_config):
                config_path = local_config

        # Try to load config file
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    cls._config_cache = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.info("No configuration file found, using defaults")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Continue with empty config and defaults
