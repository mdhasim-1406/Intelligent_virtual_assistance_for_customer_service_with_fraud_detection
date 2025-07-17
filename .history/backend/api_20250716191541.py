"""
SafeServe AI - Enhanced API with Fraud Detection and Voice Processing
FastAPI backend integrating all SafeServe AI components
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import logging
import os
import base64
import tempfile
from datetime import datetime

# Import SafeServe AI components
from backend.safeserve_ai import SafeServeAI, SafeServeResponse
from utils.load_env import load_environment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SafeServe AI API",
    description="Intelligent Virtual Assistant with Fraud Detection and Multilingual Support",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SafeServe AI system
safeserve_ai = SafeServeAI()

# Pydantic models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: str = Field(default="default", description="Session identifier")
    enable_voice_response: bool = Field(default=False, description="Generate voice response")
    language: Optional[str] = Field(default=None, description="Override language detection")

class VoiceChatRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    session_id: str = Field(default="default", description="Session identifier")
    enable_voice_response: bool = Field(default=True, description="Generate voice response")

class BatchChatRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="List of messages to process")

class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language code")
    source_language: Optional[str] = Field(default=None, description="Source language (auto-detect if None)")

class FraudAnalysisRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for fraud")
    session_id: str = Field(default="default", description="Session identifier")

# API Routes

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "service": "SafeServe AI API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Multilingual Support",
            "Conversational Fraud Detection",
            "Voice Processing",
            "Session Analytics",
            "Real-time Translation"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health = safeserve_ai.get_system_health()
        return health
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with fraud detection"""
    try:
        response = await safeserve_ai.process_user_input(
            user_input=request.message,
            session_id=request.session_id,
            input_type="text",
            enable_voice_response=request.enable_voice_response
        )
        
        return {
            "response": response.ai_response,
            "language": response.language,
            "fraud_analysis": response.fraud_analysis,
            "conversation_summary": response.conversation_summary,
            "voice_response": response.voice_response,
            "processing_time": response.processing_time,
            "confidence": response.confidence,
            "session_id": response.session_id,
            "timestamp": response.timestamp
        }
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice-chat")
async def voice_chat_endpoint(request: VoiceChatRequest):
    """Voice chat endpoint with speech-to-text and text-to-speech"""
    try:
        # Decode audio data
        audio_data = base64.b64decode(request.audio_base64)
        
        response = await safeserve_ai.process_user_input(
            user_input="",  # Will be filled by voice processing
            session_id=request.session_id,
            input_type="voice",
            audio_data=audio_data,
            enable_voice_response=request.enable_voice_response
        )
        
        return {
            "transcribed_text": response.user_message,
            "response": response.ai_response,
            "language": response.language,
            "fraud_analysis": response.fraud_analysis,
            "voice_response": response.voice_response,
            "processing_time": response.processing_time,
            "confidence": response.confidence,
            "session_id": response.session_id,
            "timestamp": response.timestamp
        }
        
    except Exception as e:
        logger.error(f"Voice chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-voice")
async def upload_voice_file(
    file: UploadFile = File(...),
    session_id: str = Form(default="default"),
    enable_voice_response: bool = Form(default=True)
):
    """Upload voice file for processing"""
    try:
        # Read audio file
        audio_data = await file.read()
        
        response = await safeserve_ai.process_user_input(
            user_input="",
            session_id=session_id,
            input_type="voice",
            audio_data=audio_data,
            enable_voice_response=enable_voice_response
        )
        
        return {
            "transcribed_text": response.user_message,
            "response": response.ai_response,
            "language": response.language,
            "fraud_analysis": response.fraud_analysis,
            "voice_response": response.voice_response,
            "processing_time": response.processing_time,
            "confidence": response.confidence,
            "session_id": response.session_id,
            "timestamp": response.timestamp
        }
        
    except Exception as e:
        logger.error(f"Voice upload endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-chat")
async def batch_chat_endpoint(request: BatchChatRequest):
    """Process multiple messages in batch"""
    try:
        responses = await safeserve_ai.batch_process_messages(request.messages)
        
        return {
            "results": [
                {
                    "user_message": r.user_message,
                    "ai_response": r.ai_response,
                    "language": r.language,
                    "fraud_analysis": r.fraud_analysis,
                    "processing_time": r.processing_time,
                    "confidence": r.confidence,
                    "session_id": r.session_id,
                    "timestamp": r.timestamp
                } for r in responses
            ],
            "total_processed": len(responses),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate_endpoint(request: TranslationRequest):
    """Text translation endpoint"""
    try:
        translator = safeserve_ai.translator
        
        # Detect source language if not provided
        if request.source_language is None:
            source_language, confidence = translator.detect_language(request.text)
        else:
            source_language = request.source_language
            confidence = 1.0
        
        # Translate text
        translation_result = translator.translate_text(
            request.text,
            source_language,
            request.target_language
        )
        
        return {
            "original_text": request.text,
            "translated_text": translation_result.translated_text,
            "source_language": source_language,
            "target_language": request.target_language,
            "confidence": translation_result.confidence,
            "processing_time": translation_result.processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Translation endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-fraud")
async def fraud_analysis_endpoint(request: FraudAnalysisRequest):
    """Fraud analysis endpoint"""
    try:
        fraud_analysis = safeserve_ai.fraud_detector.analyze_message(
            request.text, 
            request.session_id
        )
        
        return {
            "text": request.text,
            "fraud_likelihood": fraud_analysis.fraud_likelihood,
            "label": fraud_analysis.label,
            "confidence": fraud_analysis.confidence,
            "risk_factors": fraud_analysis.risk_factors,
            "behavioral_indicators": fraud_analysis.behavioral_indicators,
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Fraud analysis endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/analytics")
async def get_session_analytics(session_id: str):
    """Get analytics for a specific session"""
    try:
        analytics = safeserve_ai.get_session_analytics(session_id)
        return analytics
        
    except Exception as e:
        logger.error(f"Session analytics endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/export")
async def export_session_data(session_id: str):
    """Export session data for analysis"""
    try:
        export_data = safeserve_ai.export_session_data(session_id)
        return export_data
        
    except Exception as e:
        logger.error(f"Session export endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    try:
        safeserve_ai.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
        
    except Exception as e:
        logger.error(f"Clear session endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions")
async def clear_all_sessions():
    """Clear all active sessions"""
    try:
        safeserve_ai.clear_all_sessions()
        return {"message": "All sessions cleared successfully"}
        
    except Exception as e:
        logger.error(f"Clear all sessions endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fraud-statistics")
async def get_fraud_statistics():
    """Get fraud detection statistics"""
    try:
        stats = safeserve_ai.get_fraud_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Fraud statistics endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported-languages")
async def get_supported_languages():
    """Get supported languages"""
    try:
        languages = safeserve_ai.translator.get_supported_languages()
        return {
            "supported_languages": languages,
            "total_languages": len(languages),
            "indian_languages": len(safeserve_ai.translator.indian_languages),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Supported languages endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voice-capabilities")
async def get_voice_capabilities():
    """Get voice processing capabilities"""
    try:
        capabilities = safeserve_ai.voice_processor.get_voice_capabilities()
        return capabilities
        
    except Exception as e:
        logger.error(f"Voice capabilities endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features")
async def get_supported_features():
    """Get supported features"""
    try:
        features = safeserve_ai.get_supported_features()
        return features
        
    except Exception as e:
        logger.error(f"Features endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve audio files"""
    try:
        audio_path = os.path.join("voice_cache", filename)
        if os.path.exists(audio_path):
            return FileResponse(audio_path, media_type="audio/wav")
        else:
            raise HTTPException(status_code=404, detail="Audio file not found")
            
    except Exception as e:
        logger.error(f"Audio file endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cleanup")
async def cleanup_resources(background_tasks: BackgroundTasks):
    """Clean up old resources"""
    try:
        def cleanup_task():
            # Clean up old audio files
            safeserve_ai.voice_processor.cleanup_old_files(max_age_hours=24)
            logger.info("Resource cleanup completed")
        
        background_tasks.add_task(cleanup_task)
        return {"message": "Resource cleanup initiated"}
        
    except Exception as e:
        logger.error(f"Cleanup endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_performance_metrics():
    """Get performance metrics"""
    try:
        return {
            "performance_metrics": safeserve_ai.performance_metrics,
            "system_health": safeserve_ai.get_system_health(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Metrics endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("SafeServe AI API starting up...")
    
    # Perform any startup tasks
    try:
        # Test system components
        health = safeserve_ai.get_system_health()
        logger.info(f"System health: {health['system_status']}")
        
        # Log supported features
        features = safeserve_ai.get_supported_features()
        logger.info(f"Supported languages: {len(features['supported_languages'])}")
        logger.info(f"Voice processing enabled: STT={features['voice_processing']['speech_to_text']}, TTS={features['voice_processing']['text_to_speech']}")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("SafeServe AI API shutting down...")
    
    # Perform cleanup
    try:
        # Clear all sessions
        safeserve_ai.clear_all_sessions()
        
        # Clean up resources
        safeserve_ai.voice_processor.cleanup_old_files(max_age_hours=0)
        
        logger.info("Shutdown cleanup completed")
        
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Development mode
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting SafeServe AI API in development mode...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )