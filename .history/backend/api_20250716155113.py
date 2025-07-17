"""
SafeServe AI - FastAPI Backend
Main API server for fraud detection and chatbot proxy
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import asyncio
from typing import Dict, List, Optional, Any
import datetime
import logging
import os
from contextlib import asynccontextmanager

# Import fraud detection engine
from fraud_detection import FraudDetectionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global fraud detector instance
fraud_detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    global fraud_detector
    
    # Startup
    logger.info("Initializing SafeServe AI backend...")
    fraud_detector = FraudDetectionEngine(model_type="isolation_forest")
    fraud_detector.train()  # Train with synthetic data
    logger.info("Fraud detection engine initialized and trained")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SafeServe AI backend...")

# Create FastAPI app
app = FastAPI(
    title="SafeServe AI Backend",
    description="AI-powered fraud detection and customer service assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TransactionData(BaseModel):
    """Transaction data model for fraud detection"""
    amount: float = Field(..., description="Transaction amount")
    timestamp: Optional[str] = Field(None, description="Transaction timestamp (ISO format)")
    location: str = Field("unknown", description="Transaction location")
    device_id: str = Field("unknown", description="Device ID")
    velocity_score: float = Field(0.0, description="Transaction velocity score")
    merchant: str = Field("unknown", description="Merchant name")
    user_id: Optional[str] = Field(None, description="User ID")

class FraudPredictionResponse(BaseModel):
    """Response model for fraud prediction"""
    risk_score: float
    label: str
    confidence: float
    timestamp: str
    transaction_id: str

class ChatRequest(BaseModel):
    """Chat request model"""
    query: str = Field(..., description="User query")
    user_id: Optional[str] = Field(None, description="User ID")
    max_length: int = Field(512, description="Maximum response length")
    temperature: float = Field(0.7, description="Response temperature")

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    timestamp: str
    user_id: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    fraud_detector_ready: bool
    llm_api_available: bool
    timestamp: str

# Load environment variables
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8000/chat")

# In-memory storage for demo purposes
transaction_history: List[Dict] = []
chat_history: List[Dict] = []

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint"""
    return {
        "message": "SafeServe AI Backend is running!",
        "version": "1.0.0",
        "endpoints": {
            "fraud_detection": "/predict",
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(transaction: TransactionData):
    """
    Predict fraud probability for a transaction
    """
    try:
        # Convert to dictionary for fraud detector
        transaction_dict = {
            "amount": transaction.amount,
            "timestamp": transaction.timestamp or datetime.datetime.now().isoformat(),
            "location": transaction.location,
            "device_id": transaction.device_id,
            "velocity_score": transaction.velocity_score,
            "merchant": transaction.merchant
        }
        
        # Make prediction
        result = fraud_detector.predict(transaction_dict)
        
        # Generate transaction ID
        transaction_id = f"txn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store in history
        transaction_record = {
            "transaction_id": transaction_id,
            "user_id": transaction.user_id,
            "amount": transaction.amount,
            "location": transaction.location,
            "merchant": transaction.merchant,
            "risk_score": result["risk_score"],
            "label": result["label"],
            "timestamp": datetime.datetime.now().isoformat()
        }
        transaction_history.append(transaction_record)
        
        # Keep only last 100 transactions
        if len(transaction_history) > 100:
            transaction_history.pop(0)
        
        return FraudPredictionResponse(
            risk_score=result["risk_score"],
            label=result["label"],
            confidence=result["confidence"],
            timestamp=datetime.datetime.now().isoformat(),
            transaction_id=transaction_id
        )
        
    except Exception as e:
        logger.error(f"Error in fraud prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fraud prediction failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_llm(request: ChatRequest):
    """
    Chat with the remote LLM API
    """
    try:
        # Prepare request for external LLM API
        llm_request = {
            "query": request.query,
            "max_length": request.max_length,
            "temperature": request.temperature
        }
        
        # Make request to external LLM API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                LLM_API_URL,
                json=llm_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"LLM API error: {response.text}"
                )
            
            llm_response = response.json()
            
        # Store chat history
        chat_record = {
            "user_id": request.user_id,
            "query": request.query,
            "response": llm_response["response"],
            "timestamp": datetime.datetime.now().isoformat()
        }
        chat_history.append(chat_record)
        
        # Keep only last 100 chat messages
        if len(chat_history) > 100:
            chat_history.pop(0)
        
        return ChatResponse(
            response=llm_response["response"],
            timestamp=datetime.datetime.now().isoformat(),
            user_id=request.user_id
        )
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM API timeout")
    except httpx.RequestError as e:
        logger.error(f"LLM API request error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"LLM API connection failed: {str(e)}")
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Check fraud detector
        fraud_detector_ready = fraud_detector is not None and fraud_detector.is_trained
        
        # Check LLM API availability
        llm_api_available = False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(LLM_API_URL.replace("/chat", "/health"))
                llm_api_available = response.status_code == 200
        except:
            pass
        
        return HealthResponse(
            status="healthy" if fraud_detector_ready else "degraded",
            fraud_detector_ready=fraud_detector_ready,
            llm_api_available=llm_api_available,
            timestamp=datetime.datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/transactions", response_model=List[Dict])
async def get_transaction_history(limit: int = 50):
    """
    Get transaction history
    """
    return transaction_history[-limit:]

@app.get("/chats", response_model=List[Dict])
async def get_chat_history(limit: int = 50):
    """
    Get chat history
    """
    return chat_history[-limit:]

@app.post("/analyze-transaction-with-chat")
async def analyze_transaction_with_chat(transaction: TransactionData):
    """
    Combined endpoint: analyze transaction and provide chat-based explanation
    """
    try:
        # First, get fraud prediction
        fraud_result = fraud_detector.predict({
            "amount": transaction.amount,
            "timestamp": transaction.timestamp or datetime.datetime.now().isoformat(),
            "location": transaction.location,
            "device_id": transaction.device_id,
            "velocity_score": transaction.velocity_score,
            "merchant": transaction.merchant
        })
        
        # Create contextual query for LLM
        query = f"""A transaction of ₹{transaction.amount} was made at {transaction.merchant} from {transaction.location}. 
        The fraud detection system scored it as {fraud_result['risk_score']:.2f} ({fraud_result['label']}). 
        Please explain this result to the customer in a friendly, reassuring way and suggest next steps if needed."""
        
        # Get LLM response
        llm_request = {
            "query": query,
            "max_length": 256,
            "temperature": 0.7
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                LLM_API_URL,
                json=llm_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                llm_response = response.json()
                explanation = llm_response["response"]
            else:
                explanation = f"Transaction analysis complete. Risk score: {fraud_result['risk_score']:.2f} ({fraud_result['label']})"
        
        return {
            "fraud_analysis": fraud_result,
            "explanation": explanation,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Combined analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """
    Get system statistics
    """
    try:
        total_transactions = len(transaction_history)
        suspicious_transactions = sum(1 for t in transaction_history if t["label"] == "Suspicious")
        total_chats = len(chat_history)
        
        return {
            "total_transactions": total_transactions,
            "suspicious_transactions": suspicious_transactions,
            "safe_transactions": total_transactions - suspicious_transactions,
            "total_chats": total_chats,
            "fraud_detection_accuracy": "95.2%",  # Mock stat
            "average_response_time": "0.3s",  # Mock stat
            "system_uptime": "99.9%"  # Mock stat
        }
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")