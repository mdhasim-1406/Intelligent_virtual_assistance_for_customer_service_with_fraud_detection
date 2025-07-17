"""
SafeServe AI - Main API Server
FastAPI-based REST API for customer service with fraud detection
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import json
from datetime import datetime
import time

# Import application modules
from backend.fraud_detection import FraudDetectionEngine
from backend.logic.llm_chat import LLMChatInterface
from backend.logic.translator import TranslationEngine
from backend.logic.behavioral_fraud import BehavioralFraudAnalyzer
from utils.voice_utils import create_voice_processor
from utils.load_env import load_environment_variables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safeserve_ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
env_vars = load_environment_variables()

# Global variables for services
fraud_engine = None
llm_interface = None
translation_engine = None
behavioral_analyzer = None
voice_processor = None

# Request/Response Models
class PredictRequest(BaseModel):
    """Request model for fraud prediction"""
    amount: float = Field(..., gt=0, description="Transaction amount")
    location: str = Field(..., min_length=1, description="Transaction location")
    merchant: str = Field(..., min_length=1, description="Merchant name")
    device_id: str = Field(..., min_length=1, description="Device identifier")
    velocity_score: float = Field(..., ge=0, le=1, description="Velocity score (0-1)")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

class AssistantRequest(BaseModel):
    """Request model for assistant endpoint"""
    text: str = Field(..., min_length=1, description="User message")
    lang: str = Field(default="en", description="Language code")
    mode: str = Field(default="text", description="Response mode: text or voice")
    
    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['text', 'voice']:
            raise ValueError('Mode must be either "text" or "voice"')
        return v

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, description="Chat message")
    context: Optional[str] = Field(default="", description="Previous conversation context")
    language: str = Field(default="en", description="Language code")

class AnalyzeTransactionRequest(BaseModel):
    """Request model for transaction analysis"""
    amount: float = Field(..., gt=0, description="Transaction amount")
    location: str = Field(..., min_length=1, description="Transaction location")
    merchant: str = Field(..., min_length=1, description="Merchant name")
    device_id: str = Field(..., min_length=1, description="Device identifier")
    velocity_score: float = Field(..., ge=0, le=1, description="Velocity score (0-1)")
    user_question: Optional[str] = Field(default="", description="User question about transaction")
    language: str = Field(default="en", description="Language code")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting SafeServe AI application...")
    await initialize_services()
    logger.info("SafeServe AI application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SafeServe AI application...")
    await cleanup_services()
    logger.info("SafeServe AI application shut down complete")

# Initialize FastAPI app
app = FastAPI(
    title="SafeServe AI",
    description="Intelligent Virtual Assistant for Customer Service with Fraud Detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_services():
    """Initialize all application services"""
    global fraud_engine, llm_interface, translation_engine, behavioral_analyzer, voice_processor
    
    try:
        # Initialize fraud detection engine
        fraud_engine = FraudDetectionEngine()
        logger.info("Fraud detection engine initialized")
        
        # Initialize LLM interface
        llm_api_url = env_vars.get('LLM_API_URL', 'http://localhost:8000/chat')
        llm_interface = LLMChatInterface(llm_api_url)
        logger.info("LLM interface initialized")
        
        # Initialize translation engine
        translation_engine = TranslationEngine()
        logger.info("Translation engine initialized")
        
        # Initialize behavioral analyzer
        behavioral_analyzer = BehavioralFraudAnalyzer()
        logger.info("Behavioral fraud analyzer initialized")
        
        # Initialize voice processor
        voice_processor = create_voice_processor()
        logger.info("Voice processor initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

async def cleanup_services():
    """Cleanup application services"""
    logger.info("Cleaning up services...")
    # Add any cleanup logic here if needed

@app.get("/")
async def root():
    """Root endpoint - service information"""
    return {
        "service": "SafeServe AI",
        "version": "1.0.0",
        "description": "Intelligent Virtual Assistant for Customer Service with Fraud Detection",
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "predict": "/predict",
            "chat": "/chat",
            "assistant": "/assistant",
            "analyze_transaction": "/analyze-transaction-with-chat"
        },
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "fraud_detection": fraud_engine is not None,
            "llm_interface": llm_interface is not None,
            "translation": translation_engine is not None,
            "behavioral_analysis": behavioral_analyzer is not None,
            "voice_processing": voice_processor is not None
        }
    }
    
    # Check LLM service health
    if llm_interface:
        health_status["services"]["llm_remote"] = llm_interface.health_check()
        health_status["fallback_info"] = llm_interface.get_fallback_info()
    
    return health_status

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    stats = {
        "uptime": time.time(),
        "version": "1.0.0",
        "endpoints_available": 6,
        "services_status": {
            "fraud_detection": "active" if fraud_engine else "inactive",
            "llm_interface": "active" if llm_interface else "inactive",
            "translation": "active" if translation_engine else "inactive",
            "behavioral_analysis": "active" if behavioral_analyzer else "inactive",
            "voice_processing": "active" if voice_processor else "inactive"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return stats

@app.post("/predict")
async def predict_fraud(request: PredictRequest):
    """Predict fraud for a transaction"""
    try:
        if not fraud_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Fraud detection service not available"
            )
        
        # Prepare transaction data
        transaction_data = {
            "amount": request.amount,
            "location": request.location,
            "merchant": request.merchant,
            "device_id": request.device_id,
            "velocity_score": request.velocity_score
        }
        
        # Run fraud prediction
        prediction = fraud_engine.predict_fraud(transaction_data)
        
        return {
            "fraud_likelihood": prediction["fraud_likelihood"],
            "risk_score": prediction["risk_score"],
            "risk_factors": prediction["risk_factors"],
            "recommendation": prediction["recommendation"],
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Fraud prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during fraud prediction"
        )

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for LLM interaction"""
    try:
        if not llm_interface:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM interface not available"
            )
        
        # Get chat response
        chat_response = llm_interface.chat(
            query=request.message,
            context=request.context
        )
        
        return {
            "response": chat_response.response,
            "model_used": chat_response.model_used,
            "processing_time": chat_response.processing_time,
            "success": chat_response.success,
            "fallback": chat_response.fallback,
            "language": request.language,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during chat processing"
        )

@app.post("/assistant")
async def assistant_endpoint(request: AssistantRequest):
    """Assistant endpoint for comprehensive customer service"""
    try:
        if not llm_interface:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Assistant service not available"
            )
        
        # Get assistant response
        chat_response = llm_interface.chat(
            query=request.text,
            context=""
        )
        
        # Analyze for fraud indicators
        fraud_likelihood = 0.0
        if fraud_engine:
            try:
                # Simple fraud analysis based on text content
                text_lower = request.text.lower()
                fraud_keywords = ['fraud', 'suspicious', 'unauthorized', 'scam', 'stolen']
                fraud_likelihood = min(1.0, sum(0.2 for keyword in fraud_keywords if keyword in text_lower))
            except:
                pass
        
        response_data = {
            "response": chat_response.response,
            "fraud_likelihood": fraud_likelihood,
            "model_used": chat_response.model_used,
            "processing_time": chat_response.processing_time,
            "success": chat_response.success,
            "fallback": chat_response.fallback,
            "language": request.lang,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add voice response if requested
        if request.mode == "voice" and voice_processor:
            try:
                voice_response = voice_processor.generate_voice_response(
                    chat_response.response, 
                    request.lang
                )
                if voice_response["success"]:
                    response_data["voice_url"] = voice_response["audio_base64"]
            except Exception as e:
                logger.warning(f"Voice generation failed: {e}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Assistant endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during assistant processing"
        )

@app.post("/analyze-transaction-with-chat")
async def analyze_transaction_with_chat(request: AnalyzeTransactionRequest):
    """Combined transaction analysis with chat support"""
    try:
        result = {
            "predict": None,
            "chat": None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Run fraud prediction
        if fraud_engine:
            try:
                transaction_data = {
                    "amount": request.amount,
                    "location": request.location,
                    "merchant": request.merchant,
                    "device_id": request.device_id,
                    "velocity_score": request.velocity_score
                }
                
                prediction = fraud_engine.predict_fraud(transaction_data)
                result["predict"] = {
                    "fraud_likelihood": prediction["fraud_likelihood"],
                    "risk_score": prediction["risk_score"],
                    "risk_factors": prediction["risk_factors"],
                    "recommendation": prediction["recommendation"]
                }
            except Exception as e:
                logger.error(f"Fraud prediction failed: {e}")
                result["predict"] = {"error": "Fraud prediction failed"}
        
        # Generate chat response if user has a question
        if request.user_question and llm_interface:
            try:
                # Create context with transaction info
                context = f"Transaction: ${request.amount} at {request.merchant} in {request.location}"
                if result["predict"]:
                    context += f" (Fraud likelihood: {result['predict'].get('fraud_likelihood', 0):.2%})"
                
                chat_response = llm_interface.chat(
                    query=request.user_question,
                    context=context
                )
                
                result["chat"] = {
                    "response": chat_response.response,
                    "model_used": chat_response.model_used,
                    "processing_time": chat_response.processing_time,
                    "success": chat_response.success,
                    "fallback": chat_response.fallback
                }
            except Exception as e:
                logger.error(f"Chat response failed: {e}")
                result["chat"] = {"error": "Chat response failed"}
        
        return result
        
    except Exception as e:
        logger.error(f"Transaction analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during transaction analysis"
        )

# Custom exception handler for 404 errors
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

# Development server
if __name__ == "__main__":
    port = int(env_vars.get('PORT', 8000))
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )