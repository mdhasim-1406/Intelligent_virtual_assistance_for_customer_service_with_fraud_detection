"""
Test suite for environment variable loading
Ensures .env variables load correctly and required configurations are present
"""

import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.load_env import get_config

class TestLoadEnv:
    """Test environment variable loading"""
    
    def test_config_loads_successfully(self):
        """Test that configuration loads without errors"""
        try:
            config = get_config()
            assert config is not None
            assert isinstance(config, dict)
        except Exception as e:
            pytest.fail(f"Config loading failed: {str(e)}")
    
    def test_required_env_variables_exist(self):
        """Test that required environment variables are present"""
        config = get_config()
        
        # Check for essential configuration keys
        required_keys = ['API_HOST', 'API_PORT']
        
        for key in required_keys:
            assert key in config or os.getenv(key) is not None, f"Missing required env var: {key}"
    
    def test_llm_api_url_configured(self):
        """Test that LLM API URL is configured"""
        config = get_config()
        
        # LLM_API_URL should exist even if default
        llm_url = config.get('LLM_API_URL', 'http://localhost:8000')
        assert llm_url is not None
        assert isinstance(llm_url, str)
        assert len(llm_url) > 0
    
    def test_debug_mode_handling(self):
        """Test that debug mode is handled correctly"""
        config = get_config()
        
        debug_setting = config.get('DEBUG', False)
        # Should be either boolean or string
        assert isinstance(debug_setting, (bool, str))
        
        # If string, should be valid boolean string
        if isinstance(debug_setting, str):
            assert debug_setting.lower() in ['true', 'false', '1', '0']
    
    def test_port_is_valid_integer(self):
        """Test that API port is a valid integer"""
        config = get_config()
        
        port = config.get('API_PORT', 8080)
        
        # Should be convertible to int
        try:
            port_int = int(port)
            assert 1 <= port_int <= 65535, f"Port {port_int} is not in valid range"
        except ValueError:
            pytest.fail(f"Port '{port}' is not a valid integer")
    
    def test_host_is_valid_string(self):
        """Test that API host is a valid string"""
        config = get_config()
        
        host = config.get('API_HOST', '0.0.0.0')
        assert isinstance(host, str)
        assert len(host) > 0
    
    def test_config_keys_are_strings(self):
        """Test that all config keys are strings"""
        config = get_config()
        
        for key in config.keys():
            assert isinstance(key, str), f"Config key {key} is not a string"
    
    def test_no_none_values_in_config(self):
        """Test that no config values are None"""
        config = get_config()
        
        none_keys = [key for key, value in config.items() if value is None]
        assert len(none_keys) == 0, f"Found None values for keys: {none_keys}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])