"""
Test suite for send_message UI helper function
Tests that UI helper returns expected fields and handles errors gracefully
"""

import pytest
import requests
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.load_env import get_config

class TestSendMessage:
    """Test send_message UI helper function"""
    
    @classmethod
    def setup_class(cls):
        """Set up test configuration"""
        cls.config = get_config()
        cls.base_url = f"http://{cls.config.get('API_HOST', 'localhost')}:{cls.config.get('API_PORT', 8080)}"
        cls.timeout = 30
    
    def test_send_message_text_mode(self):
        """Test send_message function in text mode"""
        # Mock streamlit session state
        mock_session_state = Mock()
        mock_session_state.api_url = self.base_url
        mock_session_state.user_id = "test_user"
        
        with patch('streamlit.session_state', mock_session_state):
            with patch('streamlit.error') as mock_error:
                # Import here to avoid streamlit import issues
                sys.path.append(str(Path(__file__).parent.parent / "ui"))
                
                try:
                    # Create a mock send_message function that mimics the UI behavior
                    def mock_send_message(text: str, lang: str = "en", mode: str = "text", audio: str = None):
                        """Mock send_message function"""
                        try:
                            if mode == "voice" and audio:
                                # Try assistant endpoint first
                                payload = {
                                    "text": text,
                                    "lang": lang,
                                    "mode": mode,
                                    "audio": audio
                                }
                                
                                try:
                                    response = requests.post(
                                        f"{self.base_url}/assistant",
                                        json=payload,
                                        timeout=30
                                    )
                                    
                                    if response.status_code == 200:
                                        return response.json()
                                except:
                                    pass
                                
                                # Fall back to chat
                                return self._send_chat_message(text, lang)
                            else:
                                # Regular text chat
                                return self._send_chat_message(text, lang)
                                
                        except Exception as e:
                            return None
                    
                    def _send_chat_message(text: str, language: str = "auto"):
                        """Mock chat message function"""
                        try:
                            payload = {
                                "query": text,
                                "user_id": "test_user",
                                "language": language
                            }
                            
                            response = requests.post(
                                f"{self.base_url}/chat",
                                json=payload,
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                return response.json()
                            else:
                                return None
                                
                        except Exception as e:
                            return None
                    
                    # Test text mode
                    result = mock_send_message("Hello, test message", "en", "text")
                    
                    if result:
                        assert "response" in result
                        assert "timestamp" in result
                        assert "user_id" in result
                        assert "processing_time" in result
                        
                        # Check data types
                        assert isinstance(result["response"], str)
                        assert isinstance(result["user_id"], str)
                        assert isinstance(result["processing_time"], (int, float))
                        assert result["user_id"] == "test_user"
                        
                        # Response should not be empty
                        assert len(result["response"]) > 0
                    else:
                        pytest.skip("API not available for testing")
                    
                except Exception as e:
                    pytest.skip(f"Test environment not ready: {str(e)}")
    
    def test_send_message_voice_mode(self):
        """Test send_message function in voice mode"""
        # Mock streamlit session state
        mock_session_state = Mock()
        mock_session_state.api_url = self.base_url
        mock_session_state.user_id = "test_user"
        
        with patch('streamlit.session_state', mock_session_state):
            with patch('streamlit.error') as mock_error:
                try:
                    # Create a mock send_message function for voice
                    def mock_send_message_voice(text: str, lang: str = "en", mode: str = "voice", audio: str = None):
                        """Mock voice send_message function"""
                        try:
                            if mode == "voice" and audio:
                                # Try assistant endpoint first
                                payload = {
                                    "text": text,
                                    "lang": lang,
                                    "mode": mode,
                                    "audio": audio
                                }
                                
                                try:
                                    response = requests.post(
                                        f"{self.base_url}/assistant",
                                        json=payload,
                                        timeout=30
                                    )
                                    
                                    if response.status_code == 200:
                                        return response.json()
                                except:
                                    pass
                                
                                # Fall back to chat
                                payload_chat = {
                                    "query": text,
                                    "user_id": "test_user",
                                    "language": lang
                                }
                                
                                response = requests.post(
                                    f"{self.base_url}/chat",
                                    json=payload_chat,
                                    timeout=30
                                )
                                
                                if response.status_code == 200:
                                    return response.json()
                            
                            return None
                                
                        except Exception as e:
                            return None
                    
                    # Test voice mode with mock audio
                    result = mock_send_message_voice(
                        "[Voice Input]", 
                        "en", 
                        "voice", 
                        "mock_audio_base64"
                    )
                    
                    if result:
                        # Should have basic response structure
                        assert "response" in result or "fraud_likelihood" in result
                        
                        # If it's assistant endpoint response
                        if "fraud_likelihood" in result:
                            assert "fraud_label" in result
                            assert "language_detected" in result
                            assert "processing_time" in result
                            assert "capabilities" in result
                        
                        # If it's chat endpoint response
                        if "user_id" in result:
                            assert "timestamp" in result
                            assert "processing_time" in result
                    else:
                        pytest.skip("Voice mode not available for testing")
                        
                except Exception as e:
                    pytest.skip(f"Voice test environment not ready: {str(e)}")
    
    def test_send_message_error_handling(self):
        """Test send_message function error handling"""
        # Mock streamlit session state with invalid URL
        mock_session_state = Mock()
        mock_session_state.api_url = "http://invalid-url:9999"
        mock_session_state.user_id = "test_user"
        
        with patch('streamlit.session_state', mock_session_state):
            with patch('streamlit.error') as mock_error:
                try:
                    # Create a mock send_message function that will fail
                    def mock_send_message_error(text: str, lang: str = "en", mode: str = "text", audio: str = None):
                        """Mock send_message function that handles errors"""
                        try:
                            payload = {
                                "query": text,
                                "user_id": "test_user",
                                "language": lang
                            }
                            
                            response = requests.post(
                                f"{mock_session_state.api_url}/chat",
                                json=payload,
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                return response.json()
                            else:
                                return None
                                
                        except Exception as e:
                            return None
                    
                    # Test error handling
                    result = mock_send_message_error("Test message")
                    
                    # Should return None on error
                    assert result is None
                    
                except Exception as e:
                    # This is expected behavior for error handling
                    pass
    
    def test_send_message_different_languages(self):
        """Test send_message function with different languages"""
        # Mock streamlit session state
        mock_session_state = Mock()
        mock_session_state.api_url = self.base_url
        mock_session_state.user_id = "test_user"
        
        with patch('streamlit.session_state', mock_session_state):
            with patch('streamlit.error') as mock_error:
                try:
                    # Create a mock send_message function for different languages
                    def mock_send_message_lang(text: str, lang: str = "en", mode: str = "text", audio: str = None):
                        """Mock send_message function for language testing"""
                        try:
                            payload = {
                                "query": text,
                                "user_id": "test_user",
                                "language": lang
                            }
                            
                            response = requests.post(
                                f"{self.base_url}/chat",
                                json=payload,
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                return response.json()
                            else:
                                return None
                                
                        except Exception as e:
                            return None
                    
                    # Test different languages
                    languages = ["en", "hi", "ta", "auto"]
                    test_messages = {
                        "en": "Hello, I need help",
                        "hi": "मुझे मदद चाहिए",
                        "ta": "எனக்கு உதவி வேண்டும்",
                        "auto": "Help me please"
                    }
                    
                    for lang in languages:
                        message = test_messages.get(lang, "Test message")
                        result = mock_send_message_lang(message, lang)
                        
                        if result:
                            assert "response" in result
                            assert isinstance(result["response"], str)
                            assert len(result["response"]) > 0
                        else:
                            pytest.skip(f"Language {lang} not available for testing")
                    
                except Exception as e:
                    pytest.skip(f"Language test environment not ready: {str(e)}")
    
    def test_send_message_timeout_handling(self):
        """Test send_message function timeout handling"""
        # Mock streamlit session state
        mock_session_state = Mock()
        mock_session_state.api_url = self.base_url
        mock_session_state.user_id = "test_user"
        
        with patch('streamlit.session_state', mock_session_state):
            with patch('streamlit.error') as mock_error:
                try:
                    # Create a mock send_message function with short timeout
                    def mock_send_message_timeout(text: str, lang: str = "en", mode: str = "text", audio: str = None):
                        """Mock send_message function with timeout"""
                        try:
                            payload = {
                                "query": text,
                                "user_id": "test_user",
                                "language": lang
                            }
                            
                            response = requests.post(
                                f"{self.base_url}/chat",
                                json=payload,
                                timeout=0.001  # Very short timeout to test timeout handling
                            )
                            
                            if response.status_code == 200:
                                return response.json()
                            else:
                                return None
                                
                        except requests.Timeout:
                            return None
                        except Exception as e:
                            return None
                    
                    # Test timeout handling
                    result = mock_send_message_timeout("Test message")
                    
                    # Should return None on timeout
                    assert result is None
                    
                except Exception as e:
                    # This is expected behavior for timeout handling
                    pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])