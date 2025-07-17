"""
SafeServe AI - Voice Interaction Utilities
Speech-to-text (Vosk) and text-to-speech (Coqui TTS) for multilingual voice support
"""

import os
import json
import base64
import logging
import tempfile
import subprocess
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import wave
import numpy as np
from pathlib import Path

# Try to import voice libraries
try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logging.warning("Vosk not available. Voice recognition will be limited.")

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("Coqui TTS not available. Voice synthesis will be limited.")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logging.warning("soundfile not available. Using fallback audio processing.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VoiceProcessingResult:
    """Voice processing result structure"""
    success: bool
    text: str = ""
    language: str = ""
    confidence: float = 0.0
    audio_path: Optional[str] = None
    error_message: Optional[str] = None

class VoiceRecognitionEngine:
    """
    Voice recognition using Vosk for offline speech-to-text
    """
    
    def __init__(self, cache_dir: str = "./voice_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.supported_languages = {
            'en': 'vosk-model-en-us-0.22',
            'hi': 'vosk-model-hi-0.22',
            'ta': 'vosk-model-ta-0.22',  # If available
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Vosk models for supported languages"""
        if not VOSK_AVAILABLE:
            logger.warning("Vosk not available. Voice recognition disabled.")
            return
        
        for lang_code, model_name in self.supported_languages.items():
            model_path = self.cache_dir / model_name
            
            if model_path.exists():
                try:
                    logger.info(f"Loading Vosk model for {lang_code}: {model_name}")
                    model = vosk.Model(str(model_path))
                    self.models[lang_code] = model
                    logger.info(f"Successfully loaded model for {lang_code}")
                except Exception as e:
                    logger.error(f"Failed to load model for {lang_code}: {str(e)}")
            else:
                logger.warning(f"Model not found for {lang_code}: {model_path}")
                logger.info(f"To download: wget https://alphacephei.com/vosk/models/{model_name}.zip")
    
    def recognize_speech(self, audio_data: bytes, language: str = "en", sample_rate: int = 16000) -> VoiceProcessingResult:
        """
        Recognize speech from audio data
        
        Args:
            audio_data: Raw audio bytes
            language: Language code for recognition
            sample_rate: Audio sample rate
            
        Returns:
            VoiceProcessingResult with recognized text
        """
        if not VOSK_AVAILABLE:
            return VoiceProcessingResult(
                success=False,
                error_message="Vosk not available for voice recognition"
            )
        
        if language not in self.models:
            return VoiceProcessingResult(
                success=False,
                error_message=f"Model not available for language: {language}"
            )
        
        try:
            model = self.models[language]
            recognizer = vosk.KaldiRecognizer(model, sample_rate)
            
            # Process audio data
            if recognizer.AcceptWaveform(audio_data):
                result = json.loads(recognizer.Result())
                text = result.get('text', '')
                confidence = result.get('confidence', 0.0)
                
                return VoiceProcessingResult(
                    success=True,
                    text=text,
                    language=language,
                    confidence=confidence
                )
            else:
                # Partial result
                result = json.loads(recognizer.PartialResult())
                text = result.get('partial', '')
                
                return VoiceProcessingResult(
                    success=True,
                    text=text,
                    language=language,
                    confidence=0.5
                )
                
        except Exception as e:
            return VoiceProcessingResult(
                success=False,
                error_message=f"Speech recognition error: {str(e)}"
            )
    
    def recognize_from_file(self, audio_file_path: str, language: str = "en") -> VoiceProcessingResult:
        """
        Recognize speech from audio file
        
        Args:
            audio_file_path: Path to audio file
            language: Language code for recognition
            
        Returns:
            VoiceProcessingResult with recognized text
        """
        try:
            # Read audio file
            if SOUNDFILE_AVAILABLE:
                audio_data, sample_rate = sf.read(audio_file_path)
                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            else:
                # Fallback using wave
                with wave.open(audio_file_path, 'rb') as wf:
                    audio_bytes = wf.readframes(wf.getnframes())
                    sample_rate = wf.getframerate()
            
            return self.recognize_speech(audio_bytes, language, sample_rate)
            
        except Exception as e:
            return VoiceProcessingResult(
                success=False,
                error_message=f"Error reading audio file: {str(e)}"
            )
    
    def process_base64_audio(self, audio_base64: str, language: str = "en") -> VoiceProcessingResult:
        """
        Process base64 encoded audio
        
        Args:
            audio_base64: Base64 encoded audio data
            language: Language code for recognition
            
        Returns:
            VoiceProcessingResult with recognized text
        """
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            # Recognize speech
            result = self.recognize_from_file(tmp_path, language)
            
            # Clean up
            os.unlink(tmp_path)
            
            return result
            
        except Exception as e:
            return VoiceProcessingResult(
                success=False,
                error_message=f"Error processing base64 audio: {str(e)}"
            )

class VoiceSynthesisEngine:
    """
    Voice synthesis using Coqui TTS for multilingual text-to-speech
    """
    
    def __init__(self, cache_dir: str = "./voice_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.tts_models = {}
        self.supported_languages = {
            'en': 'tts_models/en/ljspeech/tacotron2-DDC',
            'hi': 'tts_models/hi/male/fairseq',
            'ta': 'tts_models/ta/male/fairseq',
        }
        
        # Initialize TTS models
        self._initialize_tts_models()
    
    def _initialize_tts_models(self):
        """Initialize TTS models for supported languages"""
        if not TTS_AVAILABLE:
            logger.warning("Coqui TTS not available. Voice synthesis disabled.")
            return
        
        try:
            # Initialize with English model first
            logger.info("Initializing TTS models...")
            self.tts_models['en'] = TTS(model_name=self.supported_languages['en'])
            logger.info("English TTS model loaded successfully")
            
            # Try to load other language models
            for lang_code, model_name in self.supported_languages.items():
                if lang_code != 'en':
                    try:
                        self.tts_models[lang_code] = TTS(model_name=model_name)
                        logger.info(f"TTS model loaded for {lang_code}")
                    except Exception as e:
                        logger.warning(f"Could not load TTS model for {lang_code}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Failed to initialize TTS models: {str(e)}")
    
    def synthesize_speech(self, text: str, language: str = "en", output_path: Optional[str] = None) -> VoiceProcessingResult:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            language: Language code for synthesis
            output_path: Output file path (optional)
            
        Returns:
            VoiceProcessingResult with audio file path
        """
        if not TTS_AVAILABLE:
            return VoiceProcessingResult(
                success=False,
                error_message="Coqui TTS not available for voice synthesis"
            )
        
        if language not in self.tts_models:
            # Fall back to English if language not available
            logger.warning(f"TTS model not available for {language}, falling back to English")
            language = 'en'
            
            if language not in self.tts_models:
                return VoiceProcessingResult(
                    success=False,
                    error_message="No TTS models available"
                )
        
        try:
            # Generate output path if not provided
            if output_path is None:
                output_path = self.cache_dir / f"tts_output_{hash(text)}_{language}.wav"
            
            # Synthesize speech
            tts_model = self.tts_models[language]
            tts_model.tts_to_file(text=text, file_path=str(output_path))
            
            return VoiceProcessingResult(
                success=True,
                text=text,
                language=language,
                audio_path=str(output_path)
            )
            
        except Exception as e:
            return VoiceProcessingResult(
                success=False,
                error_message=f"Speech synthesis error: {str(e)}"
            )
    
    def synthesize_to_base64(self, text: str, language: str = "en") -> VoiceProcessingResult:
        """
        Synthesize speech and return as base64
        
        Args:
            text: Text to synthesize
            language: Language code for synthesis
            
        Returns:
            VoiceProcessingResult with base64 audio data
        """
        # Generate audio file
        result = self.synthesize_speech(text, language)
        
        if not result.success:
            return result
        
        try:
            # Read audio file and encode as base64
            with open(result.audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Clean up temporary file
            os.unlink(result.audio_path)
            
            return VoiceProcessingResult(
                success=True,
                text=text,
                language=language,
                audio_path=audio_base64  # Store base64 in audio_path field
            )
            
        except Exception as e:
            return VoiceProcessingResult(
                success=False,
                error_message=f"Error encoding audio to base64: {str(e)}"
            )

class VoiceInteractionManager:
    """
    Complete voice interaction management
    """
    
    def __init__(self, cache_dir: str = "./voice_cache"):
        self.recognition_engine = VoiceRecognitionEngine(cache_dir)
        self.synthesis_engine = VoiceSynthesisEngine(cache_dir)
        self.cache_dir = Path(cache_dir)
    
    def process_voice_input(self, audio_base64: str, language: str = "en") -> Dict[str, Any]:
        """
        Process voice input and return structured result
        
        Args:
            audio_base64: Base64 encoded audio data
            language: Expected language
            
        Returns:
            Dictionary with recognition results
        """
        result = self.recognition_engine.process_base64_audio(audio_base64, language)
        
        return {
            "success": result.success,
            "text": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "error_message": result.error_message
        }
    
    def generate_voice_response(self, text: str, language: str = "en") -> Dict[str, Any]:
        """
        Generate voice response from text
        
        Args:
            text: Text to synthesize
            language: Target language
            
        Returns:
            Dictionary with synthesis results
        """
        result = self.synthesis_engine.synthesize_to_base64(text, language)
        
        return {
            "success": result.success,
            "text": result.text,
            "language": result.language,
            "audio_base64": result.audio_path if result.success else None,
            "error_message": result.error_message
        }
    
    def get_voice_capabilities(self) -> Dict[str, Any]:
        """Get voice processing capabilities"""
        return {
            "recognition_available": VOSK_AVAILABLE,
            "synthesis_available": TTS_AVAILABLE,
            "recognition_languages": list(self.recognition_engine.models.keys()),
            "synthesis_languages": list(self.synthesis_engine.tts_models.keys()),
            "cache_dir": str(self.cache_dir)
        }
    
    def install_voice_dependencies(self) -> Dict[str, str]:
        """
        Provide installation instructions for voice dependencies
        
        Returns:
            Dictionary with installation commands
        """
        instructions = {
            "vosk": "pip install vosk",
            "tts": "pip install TTS",
            "soundfile": "pip install soundfile",
            "numpy": "pip install numpy",
            "vosk_models": "Download from https://alphacephei.com/vosk/models/",
            "system_deps": "sudo apt-get install ffmpeg portaudio19-dev"
        }
        
        return instructions

# Simple fallback voice processing for when dependencies are not available
class FallbackVoiceProcessor:
    """
    Fallback voice processor when full voice libraries are not available
    """
    
    def __init__(self):
        self.supported_languages = ['en', 'hi', 'ta']
    
    def process_voice_input(self, audio_base64: str, language: str = "en") -> Dict[str, Any]:
        """Fallback voice input processing"""
        return {
            "success": False,
            "text": "",
            "language": language,
            "confidence": 0.0,
            "error_message": "Voice recognition not available. Please install Vosk."
        }
    
    def generate_voice_response(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Fallback voice response generation"""
        return {
            "success": False,
            "text": text,
            "language": language,
            "audio_base64": None,
            "error_message": "Voice synthesis not available. Please install Coqui TTS."
        }

# Factory function to create appropriate voice processor
def create_voice_processor(cache_dir: str = "./voice_cache"):
    """
    Create appropriate voice processor based on available dependencies
    
    Args:
        cache_dir: Directory for caching voice models
        
    Returns:
        VoiceInteractionManager or FallbackVoiceProcessor
    """
    if VOSK_AVAILABLE and TTS_AVAILABLE:
        return VoiceInteractionManager(cache_dir)
    else:
        logger.warning("Full voice processing not available. Using fallback processor.")
        return FallbackVoiceProcessor()

# Example usage
if __name__ == "__main__":
    # Create voice processor
    voice_processor = create_voice_processor()
    
    print("🎤 Testing Voice Processing:")
    print("=" * 50)
    
    # Check capabilities
    if hasattr(voice_processor, 'get_voice_capabilities'):
        capabilities = voice_processor.get_voice_capabilities()
        print(f"Recognition Available: {capabilities['recognition_available']}")
        print(f"Synthesis Available: {capabilities['synthesis_available']}")
        print(f"Recognition Languages: {capabilities['recognition_languages']}")
        print(f"Synthesis Languages: {capabilities['synthesis_languages']}")
    
    # Test voice synthesis
    test_text = "Hello, this is SafeServe AI. How can I help you today?"
    print(f"\n🗣️ Testing Voice Synthesis:")
    print(f"Text: {test_text}")
    
    response = voice_processor.generate_voice_response(test_text, "en")
    print(f"Success: {response['success']}")
    if response['success']:
        print(f"Audio generated (Base64 length: {len(response['audio_base64'])})")
    else:
        print(f"Error: {response['error_message']}")
    
    # Installation instructions
    if hasattr(voice_processor, 'install_voice_dependencies'):
        print(f"\n📦 Installation Instructions:")
        instructions = voice_processor.install_voice_dependencies()
        for component, command in instructions.items():
            print(f"  {component}: {command}")
    
    print("\n✅ Voice processing test completed")