"""
SafeServe AI - Environment Loader
Utility to load environment variables and configuration
"""

import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_environment(env_file: str = ".env") -> Dict[str, Any]:
    """
    Load environment variables from .env file
    
    Args:
        env_file: Path to .env file
        
    Returns:
        Dictionary containing environment configuration
    """
    # Load .env file
    load_dotenv(env_file)
    
    config = {
        "NGROK_AUTH_TOKEN": os.getenv("NGROK_AUTH_TOKEN"),
        "LLM_API_URL": os.getenv("LLM_API_URL", "http://localhost:8000/chat"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "deepseek-ai/deepseek-coder-6.7b-instruct"),
        "API_HOST": os.getenv("API_HOST", "0.0.0.0"),
        "API_PORT": int(os.getenv("API_PORT", "8080")),
        "DEBUG": os.getenv("DEBUG", "False").lower() == "true",
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO")
    }
    
    # Log configuration (without sensitive data)
    logger.info("Environment configuration loaded:")
    for key, value in config.items():
        if "TOKEN" in key or "SECRET" in key:
            logger.info(f"  {key}: {'*' * 8}")
        else:
            logger.info(f"  {key}: {value}")
    
    return config

def validate_configuration(config: Dict[str, Any]) -> bool:
    """
    Validate the loaded configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["LLM_API_URL"]
    
    for key in required_keys:
        if not config.get(key):
            logger.error(f"Missing required configuration: {key}")
            return False
    
    # Validate LLM API URL format
    llm_url = config["LLM_API_URL"]
    if not llm_url.startswith(("http://", "https://")):
        logger.error("LLM_API_URL must be a valid HTTP/HTTPS URL")
        return False
    
    logger.info("Configuration validation passed")
    return True

def get_config() -> Dict[str, Any]:
    """
    Get validated configuration
    
    Returns:
        Configuration dictionary
    """
    config = load_environment()
    
    if not validate_configuration(config):
        raise ValueError("Invalid configuration")
    
    return config

# Example usage
if __name__ == "__main__":
    try:
        config = get_config()
        print("Configuration loaded successfully!")
        print(f"LLM API URL: {config['LLM_API_URL']}")
        print(f"API will run on: {config['API_HOST']}:{config['API_PORT']}")
    except Exception as e:
        print(f"Error loading configuration: {e}")