"""
Perception Module for Audio Processing and Sentiment Analysis

This module handles speech-to-text conversion using OpenAI Whisper and
sentiment analysis using OpenRouter's language models.
"""

import os
import logging
import requests
import json
import tempfile
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerceptionModule:
    """
    Handles audio transcription and sentiment analysis for customer interactions.
    """
    
    def __init__(self):
        """Initialize the perception module with Whisper model."""
        try:
            # Import whisper and verify it's properly installed
            import whisper
            logger.info("Loading Whisper model...")
            
            # This is the correct way to load the model
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Whisper library not installed: {e}")
            raise RuntimeError(
                "OpenAI Whisper library is not installed. "
                "Please install it using: pip install openai-whisper"
            )
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(
                "Failed to load Whisper model. This could be due to:\n"
                "1. Missing internet connection for model download\n"
                "2. Missing ffmpeg installation (required by Whisper)\n"
                "3. Insufficient disk space\n"
                f"Original error: {e}\n\n"
                "Solutions:\n"
                "- Install ffmpeg: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)\n"
                "- Ensure stable internet connection for model download\n"
                "- Free up disk space (models require ~500MB)"
            )
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, str]:
        """
        Transcribe audio file using Whisper model.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            Dict[str, str]: Dictionary containing 'text' and 'language' keys
        """
        if not audio_path or not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return {"text": "", "language": "en", "error": "Audio file not found"}
        
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            
            # Transcribe audio using Whisper
            result = self.whisper_model.transcribe(audio_path)
            
            # Extract text and detected language
            transcribed_text = result.get("text", "").strip()
            detected_language = result.get("language", "en")
            
            # Map language codes to full names for better handling
            language_mapping = {
                "en": "English",
                "ta": "Tamil", 
                "hi": "Hindi",
                "es": "Spanish",
                "fr": "French"
            }
            
            language_name = language_mapping.get(detected_language, detected_language)
            
            logger.info(f"Transcription successful. Language: {language_name}")
            
            return {
                "text": transcribed_text,
                "language": detected_language,
                "language_name": language_name
            }
            
        except FileNotFoundError:
            logger.error(f"Audio file not found during transcription: {audio_path}")
            return {"text": "", "language": "en", "error": "Audio file not found"}
        except PermissionError as e:
            logger.error(f"Permission denied accessing audio file: {e}")
            return {"text": "", "language": "en", "error": "Permission denied accessing audio file"}
        except ValueError as e:
            logger.error(f"Invalid audio file format: {e}")
            return {"text": "", "language": "en", "error": f"Invalid audio file format: {str(e)}"}
        except RuntimeError as e:
            logger.error(f"Whisper runtime error: {e}")
            return {"text": "", "language": "en", "error": f"Transcription failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected transcription error: {e}")
            return {"text": "", "language": "en", "error": f"Transcription failed: {str(e)}"}
    
    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of text using OpenRouter API.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Sentiment classification ('Positive', 'Negative', or 'Neutral')
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for sentiment analysis")
            return "Neutral"
        
        # Get API key from environment
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            logger.error("OPENROUTER_API_KEY not found in environment variables")
            return "Neutral"
        
        try:
            # Construct sentiment analysis prompt
            sentiment_prompt = f"""
            Analyze the sentiment of the following customer service text. 
            Consider the emotional tone, urgency, and overall mood expressed.
            
            Text: "{text}"
            
            Respond with only one word: Negative, Positive, or Neutral.
            
            Guidelines:
            - Negative: Frustrated, angry, disappointed, upset, complaining
            - Positive: Happy, satisfied, grateful, excited, pleased
            - Neutral: Informational, factual, calm, routine inquiries
            """
            
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistralai/mistral-7b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": sentiment_prompt
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            logger.info("Sending sentiment analysis request to OpenRouter")
            
            # Make API request with timeout
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                sentiment = response_data["choices"][0]["message"]["content"].strip()
                
                # Validate and normalize response
                valid_sentiments = ["Negative", "Positive", "Neutral"]
                
                # Check if response contains any valid sentiment
                for valid_sentiment in valid_sentiments:
                    if valid_sentiment.lower() in sentiment.lower():
                        logger.info(f"Sentiment analysis successful: {valid_sentiment}")
                        return valid_sentiment
                
                # If no valid sentiment found, default to Neutral
                logger.warning(f"Invalid sentiment response: {sentiment}, defaulting to Neutral")
                return "Neutral"
                
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return "Neutral"
                
        except requests.exceptions.Timeout:
            logger.error("OpenRouter API request timed out")
            return "Neutral"
        except requests.exceptions.ConnectionError as e:
            logger.error(f"OpenRouter API connection failed: {e}")
            return "Neutral"
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            return "Neutral"
        except KeyError as e:
            logger.error(f"Invalid API response format: {e}")
            return "Neutral"
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from API: {e}")
            return "Neutral"
    
    def save_uploaded_audio(self, uploaded_file) -> Optional[str]:
        """
        Save uploaded audio file to temporary location.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Optional[str]: Path to saved file or None if failed
        """
        if not uploaded_file:
            return None
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            logger.info(f"Audio file saved to: {tmp_path}")
            return tmp_path
            
        except OSError as e:
            logger.error(f"Failed to save audio file - OS error: {e}")
            return None
        except AttributeError as e:
            logger.error(f"Invalid uploaded file object: {e}")
            return None
        except ValueError as e:
            logger.error(f"Invalid file data: {e}")
            return None
    
    def cleanup_temp_file(self, file_path: str) -> None:
        """
        Clean up temporary audio file.
        
        Args:
            file_path (str): Path to file to delete
        """
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except OSError as e:
            logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")
        except TypeError as e:
            logger.warning(f"Invalid file path provided for cleanup: {e}")
    
    def health_check(self) -> Dict[str, any]:
        """
        Perform health check of the perception module.
        
        Returns:
            Dict: Health check results
        """
        return {
            "whisper_model_loaded": self.whisper_model is not None,
            "openrouter_api_key_present": bool(os.environ.get("OPENROUTER_API_KEY")),
            "module_initialized": True
        }