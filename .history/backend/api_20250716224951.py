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

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "SafeServe AI",
        "version": "1.0.0",
        "description": "AI-powered customer service with fraud detection",
        "endpoints": {
            "fraud_detection": "/predict",
            "chat": "/chat",
            "combined_analysis": "/analyze-transaction-with-chat",
            "health": "/health",
            "stats": "/stats"
        },
        "llm_model": "Deepseek-Coder 6.7B (via Google Colab)",
        "fraud_detection": "Isolation Forest + Custom Rules"
    }

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