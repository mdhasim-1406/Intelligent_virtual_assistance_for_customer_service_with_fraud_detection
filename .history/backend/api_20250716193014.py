"""
SafeServe AI - Unified API Server
Production-ready API with fraud detection, multilingual support, and voice interaction
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our custom modules
from utils.load_env import load_env
from utils.voice_utils import create_voice_processor
from backend.logic.behavioral_fraud import ConversationalFraudDetector
from backend.logic.translator import MultilingualTranslator
from backend.logic.llm_chat import LLMChatInterface, ConversationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
env_vars = load_env()

# Global components (initialized on startup)
fraud_detector = None
translator = None
llm_interface = None
conversation_manager = None
voice_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application components"""
    global fraud_detector, translator, llm_interface, conversation_manager, voice_processor
    
    # Initialize components
    logger.info("🚀 Initializing SafeServe AI components...")
    
    try:
        # Initialize LLM interface
        llm_api_url = env_vars.get('LLM_API_URL', 'http://localhost:8000/chat')
        llm_interface = LLMChatInterface(llm_api_url)
        logger.info("✅ LLM interface initialized")
        
        # Initialize fraud detector
        fraud_detector = ConversationalFraudDetector(llm_api_url)
        logger.info("✅ Fraud detector initialized")
        
        # Initialize translator
        translator = MultilingualTranslator()
        logger.info("✅ Multilingual translator initialized")
        
        # Initialize conversation manager
        conversation_manager = ConversationManager(max_context_length=5)
        logger.info("✅ Conversation manager initialized")
        
        # Initialize voice processor
        voice_cache_dir = env_vars.get('VOICE_CACHE_DIR', './voice_cache')
        voice_processor = create_voice_processor(voice_cache_dir)
        logger.info("✅ Voice processor initialized")
        
        logger.info("🎯 SafeServe AI initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {str(e)}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🧹 Cleaning up SafeServe AI components...")
    # Add cleanup code here if needed

# Create FastAPI app
app = FastAPI(
    title="SafeServe AI - Enhanced Customer Assistant",
    description="Production-ready AI assistant with fraud detection, multilingual support, and voice interaction",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AssistantRequest(BaseModel):
    """Main assistant request model"""
    text: str = Field(..., description="User input text")
    lang: str = Field("auto", description="Language code (auto-detect if 'auto')")
    mode: str = Field("text", description="Interaction mode: 'text' or 'voice'")
    audio: Optional[str] = Field(None, description="Base64 encoded audio (for voice mode)")
    user_id: str = Field("anonymous", description="User identifier")
    context: Optional[str] = Field(None, description="Additional context")

class AssistantResponse(BaseModel):
    """Main assistant response model"""
    success: bool
    response: str
    translated_response: str
    fraud_likelihood: float
    fraud_label: str
    language_detected: str
    processing_time: float
    voice_url: Optional[str] = None
    audio_base64: Optional[str] = None
    error_message: Optional[str] = None
    capabilities: Dict[str, Any] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    components: Dict[str, bool]
    version: str

# API Endpoints
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "SafeServe AI",
        "version": "2.0.0",
        "description": "Enhanced customer assistant with fraud detection and multilingual support",
        "endpoints": {
            "assistant": "/assistant",
            "health": "/health",
            "capabilities": "/capabilities"
        },
        "features": [
            "Conversational fraud detection",
            "Multilingual support (English, Hindi, Tamil)",
            "Voice interaction (speech-to-text, text-to-speech)",
            "Intelligent customer service responses",
            "Real-time conversation analysis"
        ]
    }

@app.post("/assistant", response_model=AssistantResponse)
async def assistant_endpoint(request: AssistantRequest, background_tasks: BackgroundTasks):
    """
    Main assistant endpoint - handles all user interactions
    
    This endpoint processes text or voice input, detects fraud, provides multilingual support,
    and returns intelligent responses with optional voice synthesis.
    """
    start_time = datetime.now()
    
    try:
        # Step 1: Process input (text or voice)
        if request.mode == "voice" and request.audio:
            # Process voice input
            voice_result = voice_processor.process_voice_input(
                request.audio, 
                request.lang if request.lang != "auto" else "en"
            )
            
            if not voice_result["success"]:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Voice processing failed: {voice_result['error_message']}"
                )
            
            user_text = voice_result["text"]
            detected_language = voice_result["language"]
            
        else:
            # Process text input
            user_text = request.text
            detected_language = request.lang if request.lang != "auto" else None
        
        # Step 2: Language detection and translation
        if detected_language is None or detected_language == "auto":
            detected_language, confidence = translator.detect_language(user_text)
            logger.info(f"Detected language: {detected_language} (confidence: {confidence:.2f})")
        
        # Translate to English for LLM processing
        english_text, original_language = translator.process_multilingual_input(
            user_text, detected_language
        )
        
        # Step 3: Get conversation context
        conversation_context = conversation_manager.get_context(request.user_id)
        
        # Step 4: Fraud detection analysis
        fraud_analysis = fraud_detector.analyze_message(english_text, request.user_id)
        
        # Step 5: Generate LLM response
        llm_response = llm_interface.chat(
            english_text, 
            context=conversation_context,
            temperature=0.7,
            max_length=512
        )
        
        if not llm_response.success:
            logger.warning(f"LLM response failed: {llm_response.error_message}")
            english_response = "I apologize, but I'm experiencing technical difficulties. Please try again shortly."
        else:
            english_response = llm_response.response
        
        # Step 6: Translate response back to original language
        translated_response = translator.process_multilingual_response(
            english_response, original_language
        )
        
        # Step 7: Generate voice response if requested
        audio_base64 = None
        if request.mode == "voice":
            voice_response = voice_processor.generate_voice_response(
                translated_response, original_language
            )
            
            if voice_response["success"]:
                audio_base64 = voice_response["audio_base64"]
            else:
                logger.warning(f"Voice synthesis failed: {voice_response['error_message']}")
        
        # Step 8: Update conversation history
        conversation_manager.add_message(request.user_id, user_text, is_user=True)
        conversation_manager.add_message(request.user_id, translated_response, is_user=False)
        
        # Step 9: Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Step 10: Create response
        response = AssistantResponse(
            success=True,
            response=english_response,
            translated_response=translated_response,
            fraud_likelihood=fraud_analysis.fraud_likelihood,
            fraud_label=fraud_analysis.label,
            language_detected=original_language,
            processing_time=processing_time,
            audio_base64=audio_base64,
            capabilities={
                "voice_enabled": request.mode == "voice",
                "language_detected": original_language,
                "fraud_factors": fraud_analysis.risk_factors,
                "model_used": llm_response.model_used if llm_response.success else "fallback"
            }
        )
        
        # Log interaction
        logger.info(
            f"User {request.user_id}: {fraud_analysis.label} "
            f"({fraud_analysis.fraud_likelihood:.2f}) | "
            f"Lang: {original_language} | "
            f"Time: {processing_time:.2f}s"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Assistant endpoint error: {str(e)}")
        
        return AssistantResponse(
            success=False,
            response="I apologize, but I encountered an error processing your request.",
            translated_response="I apologize, but I encountered an error processing your request.",
            fraud_likelihood=0.0,
            fraud_label="error",
            language_detected=request.lang,
            processing_time=processing_time,
            error_message=str(e)
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    timestamp = datetime.now().isoformat()
    
    # Check component health
    components = {
        "llm_interface": llm_interface.health_check() if llm_interface else False,
        "fraud_detector": fraud_detector is not None,
        "translator": translator is not None,
        "conversation_manager": conversation_manager is not None,
        "voice_processor": voice_processor is not None
    }
    
    # Overall health status
    all_healthy = all(components.values())
    status = "healthy" if all_healthy else "degraded"
    
    return HealthResponse(
        status=status,
        timestamp=timestamp,
        components=components,
        version="2.0.0"
    )

@app.get("/capabilities", response_model=Dict[str, Any])
async def get_capabilities():
    """Get system capabilities"""
    capabilities = {
        "languages": {
            "supported": translator.get_supported_languages() if translator else {},
            "detection": "automatic",
            "translation": "bidirectional"
        },
        "voice": {
            "recognition": False,
            "synthesis": False,
            "languages": []
        },
        "fraud_detection": {
            "enabled": fraud_detector is not None,
            "features": [
                "conversational_analysis",
                "behavioral_patterns",
                "risk_scoring",
                "user_profiling"
            ]
        },
        "llm": {
            "available": llm_interface.health_check() if llm_interface else False,
            "model_info": llm_interface.get_service_info() if llm_interface else {}
        }
    }
    
    # Add voice capabilities if available
    if voice_processor and hasattr(voice_processor, 'get_voice_capabilities'):
        voice_caps = voice_processor.get_voice_capabilities()
        capabilities["voice"] = {
            "recognition": voice_caps.get("recognition_available", False),
            "synthesis": voice_caps.get("synthesis_available", False),
            "recognition_languages": voice_caps.get("recognition_languages", []),
            "synthesis_languages": voice_caps.get("synthesis_languages", [])
        }
    
    return capabilities

@app.get("/user/{user_id}/summary", response_model=Dict[str, Any])
async def get_user_summary(user_id: str):
    """Get user interaction summary"""
    try:
        # Get fraud risk summary
        fraud_summary = fraud_detector.get_user_risk_summary(user_id) if fraud_detector else {}
        
        # Get conversation summary
        conversation_summary = conversation_manager.get_conversation_summary(user_id) if conversation_manager else {}
        
        return {
            "user_id": user_id,
            "fraud_analysis": fraud_summary,
            "conversation_stats": conversation_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/user/{user_id}/conversation")
async def clear_user_conversation(user_id: str):
    """Clear user conversation history"""
    try:
        if conversation_manager:
            conversation_manager.clear_conversation(user_id)
        
        if fraud_detector:
            fraud_detector.clear_conversation_history()
        
        return {"message": f"Conversation cleared for user {user_id}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "Please check the API documentation"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return {"error": "Internal server error", "message": "Please try again later"}

# Run the application
if __name__ == "__main__":
    # Get configuration from environment
    host = env_vars.get('API_HOST', '0.0.0.0')
    port = int(env_vars.get('API_PORT', 8080))
    debug = env_vars.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Starting SafeServe AI API on {host}:{port}")
    logger.info(f"🔧 Debug mode: {debug}")
    
    uvicorn.run(
        "backend.api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )