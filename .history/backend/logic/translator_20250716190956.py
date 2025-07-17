"""
SafeServe AI - Multilingual Translation Pipeline
Auto-detect user language and translate for LLM processing
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import langdetect
from langdetect.lang_detect_exception import LangDetectException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TranslationResult:
    """Translation result structure"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float

class MultilingualTranslator:
    """
    Multilingual translation pipeline with language detection
    """
    
    def __init__(self):
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
            'or': 'Odia',
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
        
        self.indian_languages = ['hi', 'ta', 'te', 'bn', 'gu', 'kn', 'ml', 'mr', 'or', 'pa', 'ur']
        self.translation_models = {}
        self._load_translation_models()
    
    def _load_translation_models(self):
        """Load translation models for different language pairs"""
        try:
            # Load multilingual model for general translation
            logger.info("Loading multilingual translation models...")
            
            # Use Helsinki-NLP models for various language pairs
            self.translation_models['general'] = {
                'model_name': 'Helsinki-NLP/opus-mt-mul-en',
                'model': None,
                'tokenizer': None
            }
            
            # Load Indian language models
            self.translation_models['indic'] = {
                'model_name': 'ai4bharat/indictrans2-indic-en-1B',
                'model': None,
                'tokenizer': None
            }
            
            logger.info("Translation models initialized")
            
        except Exception as e:
            logger.error(f"Error loading translation models: {str(e)}")
            self.translation_models = {}
    
    def _lazy_load_model(self, model_type: str):
        """Lazy load translation model when needed"""
        if model_type not in self.translation_models:
            return None
        
        model_info = self.translation_models[model_type]
        
        if model_info['model'] is None:
            try:
                logger.info(f"Loading {model_type} translation model...")
                model_info['tokenizer'] = AutoTokenizer.from_pretrained(
                    model_info['model_name'], 
                    trust_remote_code=True
                )
                model_info['model'] = AutoModelForSeq2SeqLM.from_pretrained(
                    model_info['model_name'],
                    trust_remote_code=True
                )
                logger.info(f"{model_type} model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load {model_type} model: {str(e)}")
                return None
        
        return model_info
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of input text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            # Clean text for better detection
            cleaned_text = re.sub(r'[^\w\s]', ' ', text).strip()
            
            if len(cleaned_text) < 3:
                return 'en', 0.5  # Default to English for very short text
            
            # Use langdetect for primary detection
            detected = langdetect.detect_langs(cleaned_text)
            
            if detected:
                primary_lang = detected[0]
                return primary_lang.lang, primary_lang.prob
            else:
                return 'en', 0.5
                
        except LangDetectException:
            # Fallback to heuristic detection
            return self._heuristic_language_detection(text)
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            return 'en', 0.5
    
    def _heuristic_language_detection(self, text: str) -> Tuple[str, float]:
        """Heuristic language detection based on character patterns"""
        # Hindi/Devanagari script detection
        if re.search(r'[\u0900-\u097F]', text):
            return 'hi', 0.8
        
        # Tamil script detection
        if re.search(r'[\u0B80-\u0BFF]', text):
            return 'ta', 0.8
        
        # Telugu script detection
        if re.search(r'[\u0C00-\u0C7F]', text):
            return 'te', 0.8
        
        # Bengali script detection
        if re.search(r'[\u0980-\u09FF]', text):
            return 'bn', 0.8
        
        # Gujarati script detection
        if re.search(r'[\u0A80-\u0AFF]', text):
            return 'gu', 0.8
        
        # Kannada script detection
        if re.search(r'[\u0C80-\u0CFF]', text):
            return 'kn', 0.8
        
        # Malayalam script detection
        if re.search(r'[\u0D00-\u0D7F]', text):
            return 'ml', 0.8
        
        # Arabic script detection
        if re.search(r'[\u0600-\u06FF]', text):
            return 'ar', 0.8
        
        # Chinese script detection
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh', 0.8
        
        # Japanese script detection
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
            return 'ja', 0.8
        
        # Korean script detection
        if re.search(r'[\uAC00-\uD7AF]', text):
            return 'ko', 0.8
        
        # Default to English
        return 'en', 0.6
    
    def translate_to_english(self, text: str, source_lang: str = None) -> TranslationResult:
        """
        Translate text to English
        
        Args:
            text: Text to translate
            source_lang: Source language code (auto-detect if None)
            
        Returns:
            TranslationResult object
        """
        if source_lang is None:
            source_lang, confidence = self.detect_language(text)
        else:
            confidence = 0.9
        
        # If already English, return as is
        if source_lang == 'en':
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language='en',
                target_language='en',
                confidence=1.0
            )
        
        # Try to translate using available models
        translated_text = self._translate_with_models(text, source_lang, 'en')
        
        if translated_text is None:
            # Fallback to simple rule-based translation for common phrases
            translated_text = self._fallback_translation(text, source_lang)
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_lang,
            target_language='en',
            confidence=confidence * 0.8  # Reduce confidence for translation
        )
    
    def translate_from_english(self, text: str, target_lang: str) -> TranslationResult:
        """
        Translate text from English to target language
        
        Args:
            text: English text to translate
            target_lang: Target language code
            
        Returns:
            TranslationResult object
        """
        # If target is English, return as is
        if target_lang == 'en':
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language='en',
                target_language='en',
                confidence=1.0
            )
        
        # Try to translate using available models
        translated_text = self._translate_with_models(text, 'en', target_lang)
        
        if translated_text is None:
            # Fallback to original text if translation fails
            translated_text = text
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language='en',
            target_language=target_lang,
            confidence=0.8
        )
    
    def _translate_with_models(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Translate using loaded models"""
        try:
            # For Indian languages, use Indic model
            if source_lang in self.indian_languages or target_lang in self.indian_languages:
                model_info = self._lazy_load_model('indic')
                if model_info and model_info['model']:
                    return self._perform_translation(text, model_info, source_lang, target_lang)
            
            # For general languages, use general model
            model_info = self._lazy_load_model('general')
            if model_info and model_info['model']:
                return self._perform_translation(text, model_info, source_lang, target_lang)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in model translation: {str(e)}")
            return None
    
    def _perform_translation(self, text: str, model_info: Dict, source_lang: str, target_lang: str) -> str:
        """Perform actual translation with model"""
        try:
            # Create translation pipeline
            translator = pipeline(
                "translation",
                model=model_info['model'],
                tokenizer=model_info['tokenizer'],
                src_lang=source_lang,
                tgt_lang=target_lang
            )
            
            # Translate text
            result = translator(text, max_length=512)
            return result[0]['translation_text']
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text  # Return original if translation fails
    
    def _fallback_translation(self, text: str, source_lang: str) -> str:
        """Fallback translation for common phrases"""
        # Simple dictionary-based translation for common customer service phrases
        common_phrases = {
            'hi': {
                'नमस्ते': 'Hello',
                'धन्यवाद': 'Thank you',
                'मदद': 'Help',
                'समस्या': 'Problem',
                'खाता': 'Account',
                'पैसा': 'Money',
                'रुपये': 'Rupees',
                'तुरंत': 'Immediately',
                'जल्दी': 'Quickly',
                'रिफंड': 'Refund'
            },
            'ta': {
                'வணக்கம்': 'Hello',
                'நன்றி': 'Thank you',
                'உதவி': 'Help',
                'பிரச்சனை': 'Problem',
                'கணக்கு': 'Account',
                'பணம்': 'Money',
                'உடனடி': 'Immediately',
                'விரைவாக': 'Quickly'
            },
            'te': {
                'నమస్కారం': 'Hello',
                'ధన్యవాదాలు': 'Thank you',
                'సహాయం': 'Help',
                'సమస్య': 'Problem',
                'ఖాతా': 'Account',
                'డబ్బు': 'Money',
                'వెంటనే': 'Immediately'
            }
        }
        
        if source_lang in common_phrases:
            phrases = common_phrases[source_lang]
            translated_text = text
            
            for original, translation in phrases.items():
                translated_text = translated_text.replace(original, translation)
            
            return translated_text
        
        return text  # Return original if no translation available
    
    def get_language_name(self, lang_code: str) -> str:
        """Get human-readable language name"""
        return self.supported_languages.get(lang_code, f"Unknown ({lang_code})")
    
    def is_supported_language(self, lang_code: str) -> bool:
        """Check if language is supported"""
        return lang_code in self.supported_languages
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages"""
        return self.supported_languages.copy()
    
    def process_multilingual_text(self, text: str, target_lang: str = None) -> Dict[str, any]:
        """
        Complete multilingual processing pipeline
        
        Args:
            text: Input text in any supported language
            target_lang: Target language for response (auto-detect if None)
            
        Returns:
            Dictionary with processing results
        """
        # Detect source language
        source_lang, detection_confidence = self.detect_language(text)
        
        # Translate to English for LLM processing
        english_result = self.translate_to_english(text, source_lang)
        
        # Determine target language
        if target_lang is None:
            target_lang = source_lang  # Respond in same language as input
        
        return {
            'original_text': text,
            'source_language': source_lang,
            'source_language_name': self.get_language_name(source_lang),
            'detection_confidence': detection_confidence,
            'english_text': english_result.translated_text,
            'target_language': target_lang,
            'target_language_name': self.get_language_name(target_lang),
            'translation_confidence': english_result.confidence
        }

# Example usage
if __name__ == "__main__":
    # Initialize translator
    translator = MultilingualTranslator()
    
    # Test messages in different languages
    test_messages = [
        "Hello, I need help with my account",
        "मुझे अपने खाते में समस्या है, तुरंत मदद चाहिए",
        "எனக்கு உதவி வேண்டும், என் கணக்கில் பிரச்சனை உள்ளது",
        "আমার অ্যাকাউন্টে সমস্যা আছে, সাহায্য চাই",
        "मला माझ्या खात्यात समस्या आहे"
    ]
    
    print("🌍 Testing Multilingual Translation:")
    print("=" * 50)
    
    for i, message in enumerate(test_messages, 1):
        result = translator.process_multilingual_text(message)
        print(f"\n{i}. Original: {result['original_text']}")
        print(f"   Detected Language: {result['source_language_name']} ({result['source_language']})")
        print(f"   Detection Confidence: {result['detection_confidence']:.2f}")
        print(f"   English Translation: {result['english_text']}")
        print(f"   Translation Confidence: {result['translation_confidence']:.2f}")
        print("-" * 50)
    
    # Test translation back to original language
    print("\n🔄 Testing Translation Back to Original Language:")
    english_response = "I understand your concern. Let me help you with your account issue immediately."
    
    for target_lang in ['hi', 'ta', 'te', 'bn']:
        result = translator.translate_from_english(english_response, target_lang)
        print(f"\n{translator.get_language_name(target_lang)}: {result.translated_text}")
    
    print(f"\nSupported Languages: {len(translator.get_supported_languages())}")
    print("Indian Languages:", [translator.get_language_name(lang) for lang in translator.indian_languages])