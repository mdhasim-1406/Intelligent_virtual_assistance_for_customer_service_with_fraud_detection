"""
SafeServe AI - Voice Utilities
Speech-to-text (Vosk) and text-to-speech (Coqui TTS) functionality
"""

import os
import json
import wave
import logging
import asyncio
import tempfile
import base64
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from io import BytesIO
import threading
import time

# Voice processing imports
try:
    import vosk
    import pyaudio
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logging.warning("Vosk not available. Install with: pip install vosk pyaudio")

try:
    import torch
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    logging.warning("Coqui TTS not available. Install with: pip install TTS")

try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logging.warning("Audio processing not available. Install with: pip install librosa soundfile")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VoiceResult:
    """Voice processing result structure"""
    text: str
    language: str
    confidence: float
    audio_file: Optional[str] = None
    processing_time: float = 0.0
    error: Optional[str] = None

class VoiceProcessor:
    """
    Voice processing with speech-to-text and text-to-speech capabilities
    """
    
    def __init__(self, voice_cache_dir: str = "voice_cache"):
        self.voice_cache_dir = voice_cache_dir
        self.vosk_models = {}
        self.tts_models = {}
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi', 
            'ta': 'Tamil',
            'te': 'Telugu',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'mr': 'Marathi',
            'pa': 'Punjabi',
            'ur': 'Urdu',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic'
        }
        
        self._ensure_cache_dir()
        self._initialize_models()
    
    def _ensure_cache_dir(self):
        """Ensure voice cache directory exists"""
        if not os.path.exists(self.voice_cache_dir):
            os.makedirs(self.voice_cache_dir)
            logger.info(f"Created voice cache directory: {self.voice_cache_dir}")
    
    def _initialize_models(self):
        """Initialize voice models"""
        logger.info("Initializing voice processing models...")
        
        # Initialize Vosk models for speech recognition
        if VOSK_AVAILABLE:
            self._initialize_vosk_models()
        
        # Initialize Coqui TTS models
        if COQUI_AVAILABLE:
            self._initialize_tts_models()
    
    def _initialize_vosk_models(self):
        """Initialize Vosk speech recognition models"""
        try:
            # Download and setup Vosk models for different languages
            vosk_model_urls = {
                'en': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip',
                'hi': 'https://alphacephei.com/vosk/models/vosk-model-hi-0.22.zip',
                'ta': 'https://alphacephei.com/vosk/models/vosk-model-ta-0.22.zip',
                'te': 'https://alphacephei.com/vosk/models/vosk-model-te-0.22.zip',
                'bn': 'https://alphacephei.com/vosk/models/vosk-model-bn-0.22.zip'
            }
            
            # For demo purposes, we'll use a lightweight English model
            self.vosk_models['en'] = self._load_vosk_model('en')
            logger.info("Vosk models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Vosk models: {str(e)}")
    
    def _load_vosk_model(self, language: str):
        """Load Vosk model for specific language"""
        try:
            # Try to load from cache first
            model_path = os.path.join(self.voice_cache_dir, f"vosk-model-{language}")
            
            if os.path.exists(model_path):
                return vosk.Model(model_path)
            else:
                # Use small model for demo
                return vosk.Model(lang=language) if VOSK_AVAILABLE else None
                
        except Exception as e:
            logger.error(f"Error loading Vosk model for {language}: {str(e)}")
            return None
    
    def _initialize_tts_models(self):
        """Initialize Coqui TTS models"""
        try:
            # Initialize TTS models for different languages
            self.tts_models = {
                'en': TTS("tts_models/en/ljspeech/tacotron2-DDC_ph") if COQUI_AVAILABLE else None,
                'hi': TTS("tts_models/hi/mai/tacotron2-DDC") if COQUI_AVAILABLE else None,
                'ta': TTS("tts_models/ta/mai/tacotron2-DDC") if COQUI_AVAILABLE else None,
                'te': TTS("tts_models/te/mai/tacotron2-DDC") if COQUI_AVAILABLE else None,
                'es': TTS("tts_models/es/mai/tacotron2-DDC") if COQUI_AVAILABLE else None,
                'fr': TTS("tts_models/fr/mai/tacotron2-DDC") if COQUI_AVAILABLE else None,
                'de': TTS("tts_models/de/mai/tacotron2-DDC") if COQUI_AVAILABLE else None
            }
            
            logger.info("Coqui TTS models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing TTS models: {str(e)}")
            self.tts_models = {}
    
    def speech_to_text(self, audio_data: bytes, language: str = 'en') -> VoiceResult:
        """
        Convert speech to text using Vosk
        
        Args:
            audio_data: Raw audio data
            language: Language code
            
        Returns:
            VoiceResult with transcribed text
        """
        start_time = time.time()
        
        try:
            if not VOSK_AVAILABLE:
                return VoiceResult(
                    text="Speech recognition not available",
                    language=language,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    error="Vosk not installed"
                )
            
            # Get or load model for language
            model = self.vosk_models.get(language)
            if not model:
                model = self._load_vosk_model(language)
                if model:
                    self.vosk_models[language] = model
            
            if not model:
                return VoiceResult(
                    text="Model not available for this language",
                    language=language,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    error=f"No model for language: {language}"
                )
            
            # Process audio with Vosk
            recognizer = vosk.KaldiRecognizer(model, 16000)
            
            # Convert audio data to proper format
            processed_audio = self._process_audio_for_recognition(audio_data)
            
            # Process audio in chunks
            result_text = ""
            confidence = 0.0
            
            for chunk in self._audio_chunks(processed_audio, 4000):
                if recognizer.AcceptWaveform(chunk):
                    result = json.loads(recognizer.Result())
                    if result.get('text'):
                        result_text += result['text'] + " "
                        confidence = max(confidence, result.get('confidence', 0.0))
            
            # Final result
            final_result = json.loads(recognizer.FinalResult())
            if final_result.get('text'):
                result_text += final_result['text']
                confidence = max(confidence, final_result.get('confidence', 0.0))
            
            processing_time = time.time() - start_time
            
            return VoiceResult(
                text=result_text.strip(),
                language=language,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in speech-to-text: {str(e)}")
            return VoiceResult(
                text="Error processing speech",
                language=language,
                confidence=0.0,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def text_to_speech(self, text: str, language: str = 'en', voice_speed: float = 1.0) -> VoiceResult:
        """
        Convert text to speech using Coqui TTS
        
        Args:
            text: Text to convert to speech
            language: Language code
            voice_speed: Speech speed multiplier
            
        Returns:
            VoiceResult with audio file path
        """
        start_time = time.time()
        
        try:
            if not COQUI_AVAILABLE:
                return VoiceResult(
                    text=text,
                    language=language,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    error="Coqui TTS not available"
                )
            
            # Get TTS model for language
            tts_model = self.tts_models.get(language)
            if not tts_model:
                # Fallback to English if language not available
                tts_model = self.tts_models.get('en')
                
            if not tts_model:
                return VoiceResult(
                    text=text,
                    language=language,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    error="No TTS model available"
                )
            
            # Generate unique filename
            timestamp = int(time.time())
            audio_filename = f"tts_{language}_{timestamp}.wav"
            audio_path = os.path.join(self.voice_cache_dir, audio_filename)
            
            # Generate speech
            tts_model.tts_to_file(text=text, file_path=audio_path)
            
            # Adjust speed if needed
            if voice_speed != 1.0:
                audio_path = self._adjust_audio_speed(audio_path, voice_speed)
            
            processing_time = time.time() - start_time
            
            return VoiceResult(
                text=text,
                language=language,
                confidence=0.9,  # High confidence for TTS
                audio_file=audio_path,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")
            return VoiceResult(
                text=text,
                language=language,
                confidence=0.0,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def _process_audio_for_recognition(self, audio_data: bytes) -> np.ndarray:
        """Process audio data for speech recognition"""
        try:
            if AUDIO_PROCESSING_AVAILABLE:
                # Load audio with librosa
                audio, sr = librosa.load(BytesIO(audio_data), sr=16000)
                return (audio * 32767).astype(np.int16)
            else:
                # Simple fallback processing
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                return audio_array
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return np.frombuffer(audio_data, dtype=np.int16)
    
    def _audio_chunks(self, audio_data: np.ndarray, chunk_size: int):
        """Split audio into chunks for processing"""
        for i in range(0, len(audio_data), chunk_size):
            yield audio_data[i:i + chunk_size].tobytes()
    
    def _adjust_audio_speed(self, audio_path: str, speed: float) -> str:
        """Adjust audio playback speed"""
        try:
            if not AUDIO_PROCESSING_AVAILABLE:
                return audio_path
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Adjust speed
            audio_fast = librosa.effects.time_stretch(audio, rate=speed)
            
            # Save adjusted audio
            adjusted_path = audio_path.replace('.wav', f'_speed_{speed}.wav')
            sf.write(adjusted_path, audio_fast, sr)
            
            return adjusted_path
            
        except Exception as e:
            logger.error(f"Error adjusting audio speed: {str(e)}")
            return audio_path
    
    def process_voice_input(self, audio_data: bytes, language: str = None) -> Dict[str, Any]:
        """
        Complete voice input processing pipeline
        
        Args:
            audio_data: Raw audio data
            language: Language code (auto-detect if None)
            
        Returns:
            Dictionary with processing results
        """
        # Detect language if not provided
        if language is None:
            language = self._detect_audio_language(audio_data)
        
        # Convert speech to text
        stt_result = self.speech_to_text(audio_data, language)
        
        return {
            'text': stt_result.text,
            'language': stt_result.language,
            'confidence': stt_result.confidence,
            'processing_time': stt_result.processing_time,
            'error': stt_result.error
        }
    
    def generate_voice_response(self, text: str, language: str = 'en', voice_speed: float = 1.0) -> Dict[str, Any]:
        """
        Generate voice response for text
        
        Args:
            text: Text to convert to speech
            language: Language code
            voice_speed: Speech speed multiplier
            
        Returns:
            Dictionary with voice generation results
        """
        tts_result = self.text_to_speech(text, language, voice_speed)
        
        # Convert audio file to base64 for API transmission
        audio_base64 = None
        if tts_result.audio_file and os.path.exists(tts_result.audio_file):
            audio_base64 = self._audio_to_base64(tts_result.audio_file)
        
        return {
            'text': tts_result.text,
            'language': tts_result.language,
            'audio_file': tts_result.audio_file,
            'audio_base64': audio_base64,
            'processing_time': tts_result.processing_time,
            'confidence': tts_result.confidence,
            'error': tts_result.error
        }
    
    def _detect_audio_language(self, audio_data: bytes) -> str:
        """Detect language from audio data"""
        # Simple heuristic: try recognition with different models
        # In a real implementation, you'd use a language detection model
        
        languages_to_try = ['en', 'hi', 'ta', 'te', 'bn']
        best_language = 'en'
        best_confidence = 0.0
        
        for lang in languages_to_try:
            if lang in self.vosk_models:
                try:
                    result = self.speech_to_text(audio_data, lang)
                    if result.confidence > best_confidence:
                        best_confidence = result.confidence
                        best_language = lang
                except Exception:
                    continue
        
        return best_language
    
    def _audio_to_base64(self, audio_path: str) -> str:
        """Convert audio file to base64 string"""
        try:
            with open(audio_path, 'rb') as audio_file:
                return base64.b64encode(audio_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting audio to base64: {str(e)}")
            return ""
    
    def _base64_to_audio(self, base64_data: str) -> bytes:
        """Convert base64 string to audio bytes"""
        try:
            return base64.b64decode(base64_data)
        except Exception as e:
            logger.error(f"Error converting base64 to audio: {str(e)}")
            return b""
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old audio files from cache"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for filename in os.listdir(self.voice_cache_dir):
                file_path = os.path.join(self.voice_cache_dir, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        logger.info(f"Removed old audio file: {filename}")
        except Exception as e:
            logger.error(f"Error cleaning up old files: {str(e)}")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages for voice processing"""
        return self.supported_languages.copy()
    
    def is_language_supported(self, language: str) -> bool:
        """Check if language is supported for voice processing"""
        return language in self.supported_languages
    
    def get_voice_capabilities(self) -> Dict[str, Any]:
        """Get voice processing capabilities"""
        return {
            'speech_to_text': VOSK_AVAILABLE,
            'text_to_speech': COQUI_AVAILABLE,
            'audio_processing': AUDIO_PROCESSING_AVAILABLE,
            'supported_languages': list(self.supported_languages.keys()),
            'vosk_models_loaded': list(self.vosk_models.keys()),
            'tts_models_loaded': list(self.tts_models.keys())
        }

# Example usage
if __name__ == "__main__":
    # Initialize voice processor
    voice_processor = VoiceProcessor()
    
    # Test capabilities
    print("🎙️ Testing Voice Processing Capabilities:")
    print("=" * 50)
    
    capabilities = voice_processor.get_voice_capabilities()
    print(f"Speech-to-Text Available: {capabilities['speech_to_text']}")
    print(f"Text-to-Speech Available: {capabilities['text_to_speech']}")
    print(f"Audio Processing Available: {capabilities['audio_processing']}")
    print(f"Supported Languages: {len(capabilities['supported_languages'])}")
    print(f"Vosk Models Loaded: {capabilities['vosk_models_loaded']}")
    print(f"TTS Models Loaded: {capabilities['tts_models_loaded']}")
    
    # Test text-to-speech
    print("\n🔊 Testing Text-to-Speech:")
    test_texts = [
        ("Hello, welcome to SafeServe AI customer service", "en"),
        ("नमस्ते, SafeServe AI ग्राहक सेवा में आपका स्वागत है", "hi"),
        ("வணக்கம், SafeServe AI வாடிக்கையாளர் சேவைக்கு வரவேற்கிறோம்", "ta")
    ]
    
    for text, lang in test_texts:
        print(f"\n{voice_processor.supported_languages[lang]}: {text}")
        result = voice_processor.generate_voice_response(text, lang)
        print(f"   Audio Generated: {result['audio_file'] is not None}")
        print(f"   Processing Time: {result['processing_time']:.2f}s")
        if result['error']:
            print(f"   Error: {result['error']}")
    
    # Test cleanup
    print("\n🧹 Testing File Cleanup:")
    voice_processor.cleanup_old_files(max_age_hours=0)  # Remove all files
    
    print("\nVoice processing system ready!")