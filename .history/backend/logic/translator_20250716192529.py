"""
SafeServe AI - Multilingual Translation Pipeline
Auto-detect language, translate for LLM processing, translate responses back
"""

import re
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

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
    Multilingual translation pipeline supporting Tamil, Hindi, English
    """
    
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi', 
            'ta': 'Tamil',
            'te': 'Telugu',
            'bn': 'Bengali',
            'mr': 'Marathi',
            'gu': 'Gujarati'
        }
        
        self.language_detector = None
        self.translator_models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize language detection and translation models"""
        try:
            # Language detection using a lightweight model
            logger.info("Loading language detection model...")
            self.language_detector = pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection",
                device=0 if self.device == "cuda" else -1
            )
            
            # Load IndicTrans model for Indian languages
            logger.info("Loading IndicTrans model...")
            self.indic_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B")
            self.indic_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B")
            
            # Load reverse translation model
            self.indic_en_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-indic-en-1B")
            self.indic_en_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-1B")
            
            if self.device == "cuda":
                self.indic_model = self.indic_model.to("cuda")
                self.indic_en_model = self.indic_en_model.to("cuda")
            
            logger.info("Translation models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing translation models: {str(e)}")
            logger.info("Falling back to rule-based language detection")
            self.language_detector = None
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of input text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not text or len(text.strip()) < 3:
            return "en", 0.5
        
        try:
            if self.language_detector:
                # Use ML model for detection
                result = self.language_detector(text)
                lang_code = result[0]['label'].lower()
                confidence = result[0]['score']
                
                # Map some common variants
                lang_mapping = {
                    'english': 'en',
                    'hindi': 'hi',
                    'tamil': 'ta',
                    'telugu': 'te',
                    'bengali': 'bn',
                    'marathi': 'mr',
                    'gujarati': 'gu'
                }
                
                lang_code = lang_mapping.get(lang_code, lang_code)
                
                if lang_code in self.supported_languages:
                    return lang_code, confidence
                else:
                    return "en", 0.5
            else:
                # Fallback to rule-based detection
                return self._rule_based_language_detection(text)
                
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            return self._rule_based_language_detection(text)
    
    def _rule_based_language_detection(self, text: str) -> Tuple[str, float]:
        """Rule-based language detection fallback"""
        # Simple heuristics for Indian languages
        
        # Tamil detection - look for Tamil script
        tamil_pattern = r'[\u0B80-\u0BFF]'
        if re.search(tamil_pattern, text):
            return "ta", 0.8
        
        # Hindi detection - look for Devanagari script
        hindi_pattern = r'[\u0900-\u097F]'
        if re.search(hindi_pattern, text):
            return "hi", 0.8
        
        # Telugu detection
        telugu_pattern = r'[\u0C00-\u0C7F]'
        if re.search(telugu_pattern, text):
            return "te", 0.8
        
        # Bengali detection
        bengali_pattern = r'[\u0980-\u09FF]'
        if re.search(bengali_pattern, text):
            return "bn", 0.8
        
        # Check for English-like patterns
        if re.search(r'[a-zA-Z]', text):
            # Count English words vs transliterated words
            english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
            total_words = len(text.split())
            
            if total_words > 0 and english_words / total_words > 0.5:
                return "en", 0.7
        
        # Default to English
        return "en", 0.5
    
    def translate_to_english(self, text: str, source_lang: str) -> TranslationResult:
        """
        Translate text from source language to English
        
        Args:
            text: Text to translate
            source_lang: Source language code
            
        Returns:
            TranslationResult object
        """
        if source_lang == "en":
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language="en",
                target_language="en",
                confidence=1.0
            )
        
        try:
            if source_lang in ["hi", "ta", "te", "bn", "mr", "gu"]:
                # Use IndicTrans for Indian languages
                translated = self._translate_with_indictrans(text, source_lang, "en")
                confidence = 0.85
            else:
                # Fallback to simple translation
                translated = self._simple_translate(text, source_lang, "en")
                confidence = 0.6
            
            return TranslationResult(
                original_text=text,
                translated_text=translated,
                source_language=source_lang,
                target_language="en",
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error translating to English: {str(e)}")
            return TranslationResult(
                original_text=text,
                translated_text=text,  # Return original if translation fails
                source_language=source_lang,
                target_language="en",
                confidence=0.3
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
        if target_lang == "en":
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language="en",
                target_language="en",
                confidence=1.0
            )
        
        try:
            if target_lang in ["hi", "ta", "te", "bn", "mr", "gu"]:
                # Use IndicTrans for Indian languages
                translated = self._translate_with_indictrans(text, "en", target_lang)
                confidence = 0.85
            else:
                # Fallback to simple translation
                translated = self._simple_translate(text, "en", target_lang)
                confidence = 0.6
            
            return TranslationResult(
                original_text=text,
                translated_text=translated,
                source_language="en",
                target_language=target_lang,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error translating from English: {str(e)}")
            return TranslationResult(
                original_text=text,
                translated_text=text,  # Return original if translation fails
                source_language="en",
                target_language=target_lang,
                confidence=0.3
            )
    
    def _translate_with_indictrans(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using IndicTrans models"""
        try:
            if source_lang == "en" and target_lang in ["hi", "ta", "te", "bn", "mr", "gu"]:
                # English to Indian language
                inputs = self.indic_tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                
                if self.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.indic_model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=4,
                        length_penalty=0.6,
                        early_stopping=True
                    )
                
                translated = self.indic_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return translated
                
            elif source_lang in ["hi", "ta", "te", "bn", "mr", "gu"] and target_lang == "en":
                # Indian language to English
                inputs = self.indic_en_tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                
                if self.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.indic_en_model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=4,
                        length_penalty=0.6,
                        early_stopping=True
                    )
                
                translated = self.indic_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return translated
            
            else:
                # Unsupported language pair
                return text
                
        except Exception as e:
            logger.error(f"Error in IndicTrans translation: {str(e)}")
            return text
    
    def _simple_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Simple translation fallback"""
        # This is a placeholder for simple translation
        # In a real implementation, you might use a simpler model or API
        
        # For now, just return the original text
        logger.warning(f"Simple translation not implemented for {source_lang} -> {target_lang}")
        return text
    
    def process_multilingual_input(self, text: str, detected_lang: Optional[str] = None) -> Tuple[str, str]:
        """
        Process multilingual input for LLM processing
        
        Args:
            text: Input text
            detected_lang: Pre-detected language (optional)
            
        Returns:
            Tuple of (english_text, detected_language)
        """
        # Detect language if not provided
        if detected_lang is None:
            detected_lang, confidence = self.detect_language(text)
            logger.info(f"Detected language: {detected_lang} (confidence: {confidence:.2f})")
        
        # Translate to English for LLM processing
        if detected_lang != "en":
            translation_result = self.translate_to_english(text, detected_lang)
            english_text = translation_result.translated_text
            logger.info(f"Translated to English: {english_text}")
        else:
            english_text = text
        
        return english_text, detected_lang
    
    def process_multilingual_response(self, english_response: str, target_lang: str) -> str:
        """
        Process LLM response for multilingual output
        
        Args:
            english_response: Response in English
            target_lang: Target language for response
            
        Returns:
            Translated response
        """
        if target_lang == "en":
            return english_response
        
        translation_result = self.translate_from_english(english_response, target_lang)
        translated_response = translation_result.translated_text
        
        logger.info(f"Translated response to {target_lang}: {translated_response}")
        return translated_response
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages"""
        return self.supported_languages.copy()
    
    def is_language_supported(self, lang_code: str) -> bool:
        """Check if language is supported"""
        return lang_code in self.supported_languages

# Example usage
if __name__ == "__main__":
    translator = MultilingualTranslator()
    
    # Test messages in different languages
    test_messages = [
        "Hello, I need help with my account",
        "मुझे अपने खाते में मदद चाहिए",  # Hindi
        "எனக்கு என் கணக்கில் உதவி தேவை",  # Tamil
        "मला माझ्या खात्याची मदत हवी आहे"  # Marathi
    ]
    
    print("🌍 Testing Multilingual Translation:")
    print("=" * 50)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. Original: {message}")
        
        # Detect language
        detected_lang, confidence = translator.detect_language(message)
        print(f"   Detected: {translator.supported_languages.get(detected_lang, detected_lang)} ({confidence:.2f})")
        
        # Process for LLM
        english_text, lang = translator.process_multilingual_input(message, detected_lang)
        print(f"   English: {english_text}")
        
        # Simulate response and translate back
        mock_response = "I can help you with your account. What specific issue are you facing?"
        translated_response = translator.process_multilingual_response(mock_response, detected_lang)
        print(f"   Response: {translated_response}")
        print("-" * 50)