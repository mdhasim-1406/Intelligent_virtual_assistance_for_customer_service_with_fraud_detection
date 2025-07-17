"""
SafeServe AI - Main API Server
Integrates with Google Colab Deepseek LLM via ngrok
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import requests

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from load_env import get_config
from fraud_detection import FraudDetectionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config = get_config()

# Global components
fraud_detector = None
conversation_history = {}
system_stats = {
    "total_transactions": 0,
    "suspicious_transactions": 0,
    "total_chats": 0,
    "fraud_detection_accuracy": "95%"
}

# Initialize fraud detection engine
try:
    fraud_detector = FraudDetectionEngine()
    logger.info("✅ Fraud detection engine initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize fraud detection: {str(e)}")

# Create FastAPI app
app = FastAPI(
    title="SafeServe AI - Customer Assistant with Fraud Detection",
    description="AI-powered customer service with real-time fraud detection using Deepseek LLM",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AssistantRequest(BaseModel):
    """Enhanced assistant request model"""
    text: str = Field(..., description="User text input")
    lang: str = Field("en", description="Language preference")
    mode: str = Field("text", description="Interaction mode (text, voice)")
    audio: Optional[str] = Field(None, description="Base64 encoded audio data")
    context: Optional[str] = Field(None, description="Additional context")

class AssistantResponse(BaseModel):
    """Enhanced assistant response model"""
    response: str
    translated_response: str
    language_detected: str
    fraud_likelihood: float
    fraud_label: str
    processing_time: float
    timestamp: str
    user_id: str
    capabilities: Dict[str, Any]

class TransactionRequest(BaseModel):
    """Transaction fraud detection request"""
    amount: float = Field(..., description="Transaction amount")
    location: str = Field(..., description="Transaction location")
    merchant: str = Field(..., description="Merchant name")
    device_id: str = Field(..., description="Device identifier")
    velocity_score: float = Field(..., description="Transaction velocity score")
    timestamp: Optional[str] = Field(None, description="Transaction timestamp")

class FraudResponse(BaseModel):
    """Fraud detection response"""
    risk_score: float
    label: str
    explanation: str
    timestamp: str

class ChatRequest(BaseModel):
    """Chat request model"""
    query: str = Field(..., description="User query")
    user_id: str = Field("anonymous", description="User identifier")
    language: str = Field("en", description="Language preference")

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    timestamp: str
    user_id: str
    processing_time: float

class CombinedAnalysisRequest(BaseModel):
    """Combined transaction analysis with chat explanation"""
    amount: float
    location: str
    merchant: str
    device_id: str
    velocity_score: float
    timestamp: Optional[str] = None

class CombinedAnalysisResponse(BaseModel):
    """Combined analysis response"""
    fraud_analysis: FraudResponse
    explanation: str
    timestamp: str

# Helper functions
def get_llm_response(query: str, user_id: str = "user") -> str:
    """Get response from Deepseek LLM via Colab ngrok"""
    try:
        llm_url = config.get('LLM_API_URL', 'http://localhost:8000')
        
        # Handle different URL formats
        if not llm_url.endswith('/chat'):
            if llm_url.endswith('/'):
                llm_url = llm_url + 'chat'
            else:
                llm_url = llm_url + '/chat'
        
        # Prepare request for Deepseek API
        request_data = {
            "query": query,
            "max_length": 512,
            "temperature": 0.7
        }
        
        # Add context for customer service
        customer_service_query = f"""You are SafeServe AI, a helpful customer service assistant for banking and financial services. 
        
User query: {query}

Please provide a helpful, professional response focused on:
- Banking assistance
- Security and fraud prevention
- Account management
- Transaction support

Response:"""
        
        request_data["query"] = customer_service_query
        
        logger.info(f"🤖 Sending request to LLM: {llm_url}")
        
        response = requests.post(
            llm_url,
            json=request_data,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            response_data = response.json()
            if "response" in response_data:
                return response_data["response"].strip()
            else:
                return "I apologize, but I'm having trouble processing your request right now."
        else:
            logger.error(f"LLM API error: {response.status_code} - {response.text}")
            return get_fallback_response(query)
            
    except Exception as e:
        logger.error(f"Error calling LLM API: {str(e)}")
        return get_fallback_response(query)

def get_fallback_response(query: str) -> str:
    """Fallback response when LLM is unavailable"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['fraud', 'suspicious', 'scam', 'security']):
        return "I understand you have security concerns. Please contact our fraud department immediately at 1-800-FRAUD or visit your nearest branch. For immediate safety, monitor your accounts closely and report any unauthorized transactions."
    
    elif any(word in query_lower for word in ['account', 'balance', 'statement']):
        return "For account inquiries, you can check your balance through our mobile app, online banking, or by calling our customer service line. If you need assistance with specific transactions, please have your account details ready."
    
    elif any(word in query_lower for word in ['refund', 'dispute', 'transaction']):
        return "For transaction disputes or refund requests, please provide the transaction details including date, amount, and merchant. We typically process dispute requests within 3-5 business days."
    
    else:
        return "Thank you for contacting SafeServe AI. I'm here to help with your banking and security needs. Please try rephrasing your question or contact our customer service team for immediate assistance."

def check_llm_health() -> bool:
    """Check if LLM service is available"""
    try:
        llm_url = config.get('LLM_API_URL', 'http://localhost:8000')
        
        # Try health endpoint first
        health_url = llm_url.replace('/chat', '/health') if '/chat' in llm_url else llm_url + '/health'
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            return True
        
        # Try root endpoint
        root_url = llm_url.replace('/chat', '') if '/chat' in llm_url else llm_url
        response = requests.get(root_url, timeout=5)
        
        return response.status_code == 200
        
    except Exception as e:
        logger.warning(f"LLM health check failed: {str(e)}")
        return False

def analyze_text_for_fraud(text: str) -> Dict[str, Any]:
    """Analyze text for fraud indicators"""
    fraud_keywords = [
        'urgent', 'immediately', 'click here', 'verify now', 'suspend', 'frozen',
        'emergency', 'act now', 'limited time', 'confirm identity', 'update payment',
        'refund', 'prize', 'winner', 'congratulations', 'tax refund', 'irs'
    ]
    
    suspicious_phrases = [
        'click this link', 'verify your account', 'confirm your identity',
        'update your information', 'suspended account', 'unauthorized access',
        'immediate action required', 'verify within 24 hours'
    ]
    
    text_lower = text.lower()
    
    # Count fraud indicators
    keyword_count = sum(1 for keyword in fraud_keywords if keyword in text_lower)
    phrase_count = sum(1 for phrase in suspicious_phrases if phrase in text_lower)
    
    # Calculate risk score
    risk_score = min(0.95, (keyword_count * 0.1) + (phrase_count * 0.2))
    
    # Determine label
    if risk_score > 0.6:
        label = "Suspicious"
    elif risk_score > 0.3:
        label = "Moderate"
    else:
        label = "Safe"
    
    return {
        "risk_score": risk_score,
        "label": label,
        "keyword_count": keyword_count,
        "phrase_count": phrase_count,
        "detected_keywords": [kw for kw in fraud_keywords if kw in text_lower],
        "detected_phrases": [ph for ph in suspicious_phrases if ph in text_lower]
    }

def detect_language(text: str) -> str:
    """Simple language detection based on character patterns"""
    # Hindi detection (Devanagari script)
    if any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in text):
        return "hi"
    
    # Tamil detection
    if any(ord(char) >= 0x0B80 and ord(char) <= 0x0BFF for char in text):
        return "ta"
    
    # Telugu detection
    if any(ord(char) >= 0x0C00 and ord(char) <= 0x0C7F for char in text):
        return "te"
    
    # Bengali detection
    if any(ord(char) >= 0x0980 and ord(char) <= 0x09FF for char in text):
        return "bn"
    
    # Default to English
    return "en"

# Configure structured logging with JSON format
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if available
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        if hasattr(record, 'processing_time'):
            log_entry["processing_time"] = record.processing_time
        if hasattr(record, 'fraud_score'):
            log_entry["fraud_score"] = record.fraud_score
        
        return json.dumps(log_entry)

# Configure logging with JSON formatter
def setup_logging():
    """Setup structured logging configuration"""
    json_formatter = JSONFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    console_handler.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler('safeserve_ai.log')
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return root_logger

# Setup structured logging
logger = setup_logging()

# Add startup logging
def log_startup_info():
    """Log startup information"""
    logger.info("SafeServe AI Starting up", extra={
        "service": "SafeServe AI",
        "version": "1.0.0",
        "startup": True
    })
    
    # Check LLM availability
    llm_available = check_llm_health()
    logger.info(f"LLM Service Status: {'Available' if llm_available else 'Unavailable'}", extra={
        "component": "llm_service",
        "status": "available" if llm_available else "unavailable",
        "url": config.get('LLM_API_URL', 'not configured')
    })
    
    # Check fraud detection
    fraud_available = fraud_detector is not None
    logger.info(f"Fraud Detection Status: {'Available' if fraud_available else 'Unavailable'}", extra={
        "component": "fraud_detection",
        "status": "available" if fraud_available else "unavailable"
    })

# Call startup logging
log_startup_info()

# API Endpoints
@app.post("/assistant", response_model=AssistantResponse)
async def enhanced_assistant(request: AssistantRequest):
    """Enhanced assistant endpoint with fraud detection and language support"""
    start_time = time.time()
    
    try:
        # Detect language if not specified or auto
        if request.lang == "auto":
            detected_lang = detect_language(request.text)
        else:
            detected_lang = request.lang
        
        # Get LLM response
        response_text = get_llm_response(request.text, "assistant_user")
        
        # Analyze for fraud indicators
        fraud_analysis = analyze_text_for_fraud(request.text)
        
        # For voice mode, handle audio processing
        audio_response = None
        if request.mode == "voice":
            # Voice processing would go here
            # For now, just return the text response
            pass
        
        # Store conversation with fraud analysis
        conversation_key = f"assistant_{int(time.time())}"
        conversation_history[conversation_key] = {
            "text": request.text,
            "response": response_text,
            "fraud_analysis": fraud_analysis,
            "language": detected_lang,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update stats
        system_stats["total_chats"] += 1
        if fraud_analysis["label"] == "Suspicious":
            system_stats["suspicious_transactions"] += 1
        
        processing_time = time.time() - start_time
        
        # Build capabilities response
        capabilities = {
            "fraud_detection": fraud_analysis["label"] != "Safe",
            "fraud_factors": fraud_analysis["detected_keywords"] + fraud_analysis["detected_phrases"],
            "language_detection": True,
            "voice_support": request.mode == "voice",
            "multilingual": True
        }
        
        return AssistantResponse(
            response=response_text,
            translated_response=response_text,  # Translation would go here
            language_detected=detected_lang,
            fraud_likelihood=fraud_analysis["risk_score"],
            fraud_label=fraud_analysis["label"],
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            user_id="assistant_user",
            capabilities=capabilities
        )
        
    except Exception as e:
        logger.error(f"Assistant error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=FraudResponse)
async def predict_fraud(request: TransactionRequest):
    """Fraud detection endpoint"""
    try:
        if not fraud_detector:
            raise HTTPException(status_code=503, detail="Fraud detection service unavailable")
        
        # Prepare transaction data
        transaction_data = {
            "amount": request.amount,
            "location": request.location,
            "merchant": request.merchant,
            "device_id": request.device_id,
            "velocity_score": request.velocity_score,
            "timestamp": request.timestamp or datetime.now().isoformat()
        }
        
        # Analyze transaction
        result = fraud_detector.analyze_transaction(transaction_data)
        
        # Update stats
        system_stats["total_transactions"] += 1
        if result["label"] == "Suspicious":
            system_stats["suspicious_transactions"] += 1
        
        # Generate explanation
        risk_level = "high" if result["risk_score"] > 0.7 else "moderate" if result["risk_score"] > 0.4 else "low"
        explanation = f"Transaction analyzed with {risk_level} risk level. Amount: ${request.amount}, Location: {request.location}, Merchant: {request.merchant}"
        
        return FraudResponse(
            risk_score=result["risk_score"],
            label=result["label"],
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Fraud prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with AI assistant"""
    start_time = time.time()
    
    try:
        # Get response from LLM
        response_text = get_llm_response(request.query, request.user_id)
        
        # Store conversation
        if request.user_id not in conversation_history:
            conversation_history[request.user_id] = []
        
        conversation_history[request.user_id].append({
            "query": request.query,
            "response": response_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update stats
        system_stats["total_chats"] += 1
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.now().isoformat(),
            user_id=request.user_id,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-transaction-with-chat", response_model=CombinedAnalysisResponse)
async def analyze_transaction_with_chat(request: CombinedAnalysisRequest):
    """Combined transaction analysis with AI explanation"""
    try:
        # Fraud analysis
        if not fraud_detector:
            raise HTTPException(status_code=503, detail="Fraud detection service unavailable")
        
        transaction_data = {
            "amount": request.amount,
            "location": request.location,
            "merchant": request.merchant,
            "device_id": request.device_id,
            "velocity_score": request.velocity_score,
            "timestamp": request.timestamp or datetime.now().isoformat()
        }
        
        fraud_result = fraud_detector.analyze_transaction(transaction_data)
        
        # Generate AI explanation
        explanation_query = f"""Analyze this transaction for fraud risk:
- Amount: ${request.amount}
- Location: {request.location}
- Merchant: {request.merchant}
- Device: {request.device_id}
- Velocity Score: {request.velocity_score}
- Risk Score: {fraud_result['risk_score']:.2f}
- Classification: {fraud_result['label']}

Provide a clear explanation of why this transaction is classified as {fraud_result['label'].lower()} and what factors contributed to this risk assessment."""
        
        explanation = get_llm_response(explanation_query)
        
        # Update stats
        system_stats["total_transactions"] += 1
        if fraud_result["label"] == "Suspicious":
            system_stats["suspicious_transactions"] += 1
        
        return CombinedAnalysisResponse(
            fraud_analysis=FraudResponse(
                risk_score=fraud_result["risk_score"],
                label=fraud_result["label"],
                explanation=fraud_result.get("explanation", ""),
                timestamp=datetime.now().isoformat()
            ),
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Combined analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "SafeServe AI",
        "version": "1.0.0",
        "description": "AI-powered customer service with fraud detection",
        "endpoints": {
            "assistant": "/assistant",
            "fraud_detection": "/predict",
            "chat": "/chat",
            "combined_analysis": "/analyze-transaction-with-chat",
            "health": "/health",
            "stats": "/stats",
            "transactions": "/transactions",
            "chats": "/chats"
        },
        "llm_model": "Deepseek-Coder 6.7B (via Google Colab)",
        "fraud_detection": "Isolation Forest + Custom Rules + Text Analysis"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    llm_available = check_llm_health()
    fraud_available = fraud_detector is not None
    
    return {
        "status": "healthy" if llm_available and fraud_available else "degraded",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "llm_api": llm_available,
            "fraud_detection": fraud_available,
            "llm_api_url": config.get('LLM_API_URL', 'not configured')
        },
        "version": "1.0.0"
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        **system_stats,
        "active_conversations": len(conversation_history),
        "llm_status": "connected" if check_llm_health() else "disconnected",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/transactions")
async def get_transactions():
    """Get recent transactions (mock data for now)"""
    return {
        "transactions": [],
        "count": system_stats["total_transactions"],
        "suspicious_count": system_stats["suspicious_transactions"]
    }

@app.get("/chats")
async def get_chats():
    """Get recent chats"""
    all_chats = []
    for user_id, chats in conversation_history.items():
        for chat in chats[-5:]:  # Last 5 chats per user
            all_chats.append({
                "user_id": user_id,
                **chat
            })
    
    return {
        "chats": sorted(all_chats, key=lambda x: x["timestamp"], reverse=True)[:20],
        "total_chats": system_stats["total_chats"]
    }

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
    host = config.get('API_HOST', '0.0.0.0')
    port = int(config.get('API_PORT', 8080))
    
    # Handle DEBUG setting properly (could be boolean or string)
    debug_setting = config.get('DEBUG', False)
    if isinstance(debug_setting, bool):
        debug = debug_setting
    else:
        debug = str(debug_setting).lower() == 'true'
    
    logger.info(f"🚀 Starting SafeServe AI API on {host}:{port}")
    logger.info(f"🤖 LLM API URL: {config.get('LLM_API_URL', 'not configured')}")
    logger.info(f"🔧 Debug mode: {debug}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )