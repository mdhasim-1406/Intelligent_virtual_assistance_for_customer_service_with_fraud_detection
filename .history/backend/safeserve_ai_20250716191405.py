"""
SafeServe AI - Main Orchestrator
Integrates all components for intelligent virtual assistance with fraud detection
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import datetime

# Import all SafeServe AI modules
from backend.logic.behavioral_fraud import ConversationalFraudDetector
from backend.logic.translator import MultilingualTranslator
from backend.logic.llm_chat import LLMChatInterface
from utils.voice_utils import VoiceProcessor
from utils.load_env import load_environment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SafeServeResponse:
    """Complete SafeServe AI response structure"""
    user_message: str
    ai_response: str
    language: str
    fraud_analysis: Dict[str, Any]
    conversation_summary: Dict[str, Any]
    voice_response: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    confidence: float = 0.0
    session_id: str = ""
    timestamp: str = ""

class SafeServeAI:
    """
    Main SafeServe AI orchestrator integrating all components
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_environment()
        self.session_data = {}
        self.active_sessions = {}
        
        # Initialize all components
        self._initialize_components()
        
        # Performance monitoring
        self.performance_metrics = {
            'total_requests': 0,
            'fraud_detections': 0,
            'language_translations': 0,
            'voice_interactions': 0,
            'average_response_time': 0.0,
            'error_count': 0
        }
        
        logger.info("SafeServe AI system initialized successfully")
    
    def _initialize_components(self):
        """Initialize all SafeServe AI components"""
        try:
            # Initialize fraud detector
            llm_url = self.config.get('LLM_API_URL', 'http://localhost:8000/chat')
            self.fraud_detector = ConversationalFraudDetector(llm_url)
            
            # Initialize translator
            self.translator = MultilingualTranslator()
            
            # Initialize LLM chat interface
            self.llm_interface = LLMChatInterface(llm_url)
            
            # Initialize voice processor
            self.voice_processor = VoiceProcessor()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    async def process_user_input(self, 
                                user_input: str,
                                session_id: str = "default",
                                input_type: str = "text",
                                audio_data: bytes = None,
                                enable_voice_response: bool = False) -> SafeServeResponse:
        """
        Process user input through the complete SafeServe AI pipeline
        
        Args:
            user_input: User's text input or voice transcript
            session_id: Session identifier for conversation tracking
            input_type: Type of input ('text' or 'voice')
            audio_data: Raw audio data for voice input
            enable_voice_response: Whether to generate voice response
            
        Returns:
            SafeServeResponse with complete processing results
        """
        start_time = time.time()
        
        try:
            # Update metrics
            self.performance_metrics['total_requests'] += 1
            
            # Initialize session if needed
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    'start_time': datetime.datetime.now(),
                    'message_count': 0,
                    'fraud_score_history': [],
                    'languages_used': set(),
                    'voice_enabled': enable_voice_response
                }
            
            session = self.active_sessions[session_id]
            session['message_count'] += 1
            
            # Step 1: Process voice input if needed
            if input_type == "voice" and audio_data:
                voice_result = self.voice_processor.process_voice_input(audio_data)
                user_input = voice_result['text']
                detected_language = voice_result['language']
                self.performance_metrics['voice_interactions'] += 1
            else:
                # Detect language for text input
                detected_language, detection_confidence = self.translator.detect_language(user_input)
            
            # Step 2: Translate to English for processing
            translation_result = self.translator.process_multilingual_text(user_input)
            english_text = translation_result['english_text']
            source_language = translation_result['source_language']
            
            # Update session language info
            session['languages_used'].add(source_language)
            
            if source_language != 'en':
                self.performance_metrics['language_translations'] += 1
            
            # Step 3: Fraud detection analysis
            fraud_analysis = self.fraud_detector.analyze_message(english_text, session_id)
            
            # Track fraud detection
            if fraud_analysis.fraud_likelihood > 0.5:
                self.performance_metrics['fraud_detections'] += 1
            
            session['fraud_score_history'].append(fraud_analysis.fraud_likelihood)
            
            # Step 4: Generate AI response based on fraud analysis
            conversation_type = self._determine_conversation_type(fraud_analysis)
            
            llm_response = self.llm_interface.chat(
                english_text,
                user_id=session_id,
                conversation_type=conversation_type,
                temperature=0.7 if fraud_analysis.fraud_likelihood < 0.5 else 0.3
            )
            
            # Step 5: Translate response back to user's language
            if source_language != 'en':
                response_translation = self.translator.translate_from_english(
                    llm_response.response, 
                    source_language
                )
                translated_response = response_translation.translated_text
            else:
                translated_response = llm_response.response
            
            # Step 6: Generate voice response if requested
            voice_response = None
            if enable_voice_response:
                voice_response = self.voice_processor.generate_voice_response(
                    translated_response,
                    source_language
                )
            
            # Step 7: Generate conversation summary
            conversation_summary = self.llm_interface.get_conversation_summary(session_id)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(processing_time)
            
            # Create response
            response = SafeServeResponse(
                user_message=user_input,
                ai_response=translated_response,
                language=source_language,
                fraud_analysis=asdict(fraud_analysis),
                conversation_summary=conversation_summary,
                voice_response=voice_response,
                processing_time=processing_time,
                confidence=self._calculate_overall_confidence(fraud_analysis, llm_response),
                session_id=session_id,
                timestamp=datetime.datetime.now().isoformat()
            )
            
            logger.info(f"Processed request for session {session_id}: {fraud_analysis.label} risk")
            return response
            
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            self.performance_metrics['error_count'] += 1
            
            # Return error response
            return SafeServeResponse(
                user_message=user_input,
                ai_response="I apologize, but I'm experiencing technical difficulties. Please try again.",
                language="en",
                fraud_analysis={"error": str(e)},
                conversation_summary={"error": str(e)},
                processing_time=time.time() - start_time,
                confidence=0.0,
                session_id=session_id,
                timestamp=datetime.datetime.now().isoformat()
            )
    
    def _determine_conversation_type(self, fraud_analysis) -> str:
        """Determine conversation type based on fraud analysis"""
        if fraud_analysis.fraud_likelihood > 0.7:
            return "fraud_analysis"
        elif fraud_analysis.fraud_likelihood > 0.4:
            return "customer_service"  # Cautious customer service
        else:
            return "customer_service"  # Normal customer service
    
    def _calculate_overall_confidence(self, fraud_analysis, llm_response) -> float:
        """Calculate overall confidence score"""
        fraud_confidence = fraud_analysis.confidence
        llm_confidence = llm_response.confidence
        
        # Weighted average with fraud analysis having higher importance
        return (fraud_confidence * 0.6) + (llm_confidence * 0.4)
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics"""
        # Update average response time
        current_avg = self.performance_metrics['average_response_time']
        total_requests = self.performance_metrics['total_requests']
        
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.performance_metrics['average_response_time'] = new_avg
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a specific session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        user_risk_summary = self.fraud_detector.get_user_risk_summary(session_id)
        
        return {
            "session_id": session_id,
            "start_time": session['start_time'].isoformat(),
            "message_count": session['message_count'],
            "languages_used": list(session['languages_used']),
            "voice_enabled": session['voice_enabled'],
            "fraud_score_history": session['fraud_score_history'],
            "average_fraud_score": sum(session['fraud_score_history']) / len(session['fraud_score_history']) if session['fraud_score_history'] else 0,
            "user_risk_summary": user_risk_summary,
            "session_duration": str(datetime.datetime.now() - session['start_time']).split('.')[0]
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health and performance metrics"""
        # Check component health
        llm_health = self.llm_interface.health_check()
        voice_capabilities = self.voice_processor.get_voice_capabilities()
        
        return {
            "system_status": "healthy" if llm_health['status'] == "healthy" else "degraded",
            "components": {
                "fraud_detector": "operational",
                "translator": "operational", 
                "llm_interface": llm_health['status'],
                "voice_processor": "operational"
            },
            "performance_metrics": self.performance_metrics,
            "voice_capabilities": voice_capabilities,
            "active_sessions": len(self.active_sessions),
            "supported_languages": len(self.translator.get_supported_languages()),
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def get_fraud_statistics(self) -> Dict[str, Any]:
        """Get fraud detection statistics"""
        total_requests = self.performance_metrics['total_requests']
        fraud_detections = self.performance_metrics['fraud_detections']
        
        # Calculate fraud detection rate
        fraud_rate = (fraud_detections / total_requests * 100) if total_requests > 0 else 0
        
        # Get user risk profiles
        user_profiles = {}
        for session_id in self.active_sessions:
            user_profiles[session_id] = self.fraud_detector.get_user_risk_summary(session_id)
        
        return {
            "total_requests": total_requests,
            "fraud_detections": fraud_detections,
            "fraud_detection_rate": fraud_rate,
            "high_risk_users": len([p for p in user_profiles.values() if p.get('risk_level') == 'high_risk']),
            "user_risk_profiles": user_profiles,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            
        # Clear from individual components
        self.llm_interface.clear_conversation_history(session_id)
        self.fraud_detector.clear_conversation_history()
        
        logger.info(f"Cleared session: {session_id}")
    
    def clear_all_sessions(self):
        """Clear all active sessions"""
        self.active_sessions = {}
        self.llm_interface.clear_conversation_history()
        self.fraud_detector.clear_conversation_history()
        
        logger.info("Cleared all sessions")
    
    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """Export session data for analysis"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session_analytics = self.get_session_analytics(session_id)
        conversation_summary = self.llm_interface.get_conversation_summary(session_id)
        
        return {
            "session_analytics": session_analytics,
            "conversation_summary": conversation_summary,
            "export_timestamp": datetime.datetime.now().isoformat()
        }
    
    async def batch_process_messages(self, messages: List[Dict[str, Any]]) -> List[SafeServeResponse]:
        """Process multiple messages in batch"""
        tasks = []
        
        for msg in messages:
            task = self.process_user_input(
                user_input=msg['text'],
                session_id=msg.get('session_id', 'batch'),
                input_type=msg.get('input_type', 'text'),
                audio_data=msg.get('audio_data'),
                enable_voice_response=msg.get('enable_voice_response', False)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    def get_supported_features(self) -> Dict[str, Any]:
        """Get list of supported features"""
        return {
            "multilingual_support": True,
            "fraud_detection": True,
            "voice_processing": {
                "speech_to_text": self.voice_processor.get_voice_capabilities()['speech_to_text'],
                "text_to_speech": self.voice_processor.get_voice_capabilities()['text_to_speech']
            },
            "conversation_tracking": True,
            "sentiment_analysis": True,
            "session_analytics": True,
            "batch_processing": True,
            "supported_languages": self.translator.get_supported_languages(),
            "indian_languages": len([lang for lang in self.translator.indian_languages if lang in self.translator.supported_languages])
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_safeserve_ai():
        # Initialize SafeServe AI
        safeserve = SafeServeAI()
        
        print("🚀 SafeServe AI System Test")
        print("=" * 60)
        
        # Test system health
        health = safeserve.get_system_health()
        print(f"System Status: {health['system_status']}")
        print(f"Active Components: {list(health['components'].keys())}")
        
        # Test supported features
        features = safeserve.get_supported_features()
        print(f"Supported Languages: {len(features['supported_languages'])}")
        print(f"Indian Languages: {features['indian_languages']}")
        print(f"Voice Processing: STT={features['voice_processing']['speech_to_text']}, TTS={features['voice_processing']['text_to_speech']}")
        
        # Test messages in different languages
        test_messages = [
            "Hello, I need help with my account",
            "मुझे अपने खाते में समस्या है",
            "I DEMAND a refund RIGHT NOW! This is urgent!",
            "எனக்கு உதவி வேண்டும்",
            "Thank you for your help"
        ]
        
        print("\n💬 Testing Message Processing:")
        print("-" * 40)
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n{i}. User: {message}")
            
            response = await safeserve.process_user_input(
                user_input=message,
                session_id="test_session"
            )
            
            print(f"   Language: {safeserve.translator.get_language_name(response.language)}")
            print(f"   Fraud Risk: {response.fraud_analysis['label']} ({response.fraud_analysis['fraud_likelihood']:.2f})")
            print(f"   AI Response: {response.ai_response[:100]}...")
            print(f"   Processing Time: {response.processing_time:.2f}s")
            print(f"   Confidence: {response.confidence:.2f}")
        
        # Show session analytics
        print("\n📊 Session Analytics:")
        analytics = safeserve.get_session_analytics("test_session")
        print(f"   Messages: {analytics['message_count']}")
        print(f"   Languages: {analytics['languages_used']}")
        print(f"   Average Fraud Score: {analytics['average_fraud_score']:.2f}")
        print(f"   Session Duration: {analytics['session_duration']}")
        
        # Show fraud statistics
        print("\n🔍 Fraud Detection Statistics:")
        fraud_stats = safeserve.get_fraud_statistics()
        print(f"   Total Requests: {fraud_stats['total_requests']}")
        print(f"   Fraud Detections: {fraud_stats['fraud_detections']}")
        print(f"   Detection Rate: {fraud_stats['fraud_detection_rate']:.1f}%")
        
        # Performance metrics
        print("\n⚡ Performance Metrics:")
        metrics = safeserve.performance_metrics
        print(f"   Average Response Time: {metrics['average_response_time']:.3f}s")
        print(f"   Voice Interactions: {metrics['voice_interactions']}")
        print(f"   Language Translations: {metrics['language_translations']}")
        print(f"   Error Count: {metrics['error_count']}")
        
        print("\n✅ SafeServe AI system test completed successfully!")
    
    # Run test
    asyncio.run(test_safeserve_ai())