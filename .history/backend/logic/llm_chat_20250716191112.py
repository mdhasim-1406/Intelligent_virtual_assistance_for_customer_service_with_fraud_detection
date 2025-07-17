"""
SafeServe AI - LLM Chat Interface
Interface to communicate with Deepseek/Mistral LLM models
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import datetime
import asyncio
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    """Chat response structure"""
    response: str
    model_used: str
    timestamp: str
    processing_time: float
    confidence: float
    error: Optional[str] = None

class LLMChatInterface:
    """
    Interface for communicating with LLM models
    """
    
    def __init__(self, llm_api_url: str, model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct"):
        self.llm_api_url = llm_api_url
        self.model_name = model_name
        self.conversation_history = {}
        self.system_prompts = self._load_system_prompts()
        self.max_retries = 3
        self.timeout = 30
        
    def _load_system_prompts(self) -> Dict[str, str]:
        """Load system prompts for different conversation types"""
        return {
            "customer_service": """You are SafeServe AI, a helpful and professional customer service assistant for a financial services company. You specialize in:

- Account inquiries and support
- Transaction dispute resolution  
- Security and fraud concerns
- General banking assistance
- Payment and transfer issues

Guidelines:
- Be empathetic and understanding
- Provide clear, actionable solutions
- Maintain professional tone
- Prioritize customer security
- Ask clarifying questions when needed
- Escalate complex issues appropriately
- Never share sensitive information
- Always verify customer identity for sensitive requests

Respond in a helpful, concise manner while ensuring customer satisfaction and security.""",

            "fraud_analysis": """You are an expert fraud detection analyst. Your role is to:

- Analyze customer messages for potential fraud indicators
- Identify suspicious patterns in communication
- Assess risk levels based on conversation content
- Provide detailed explanations of fraud likelihood
- Suggest appropriate security measures

Focus on detecting:
- Urgency tactics and pressure
- Financial manipulation attempts
- Identity theft attempts
- Phishing and social engineering
- Inconsistent stories or information
- Emotional manipulation tactics

Provide objective, evidence-based analysis while maintaining customer service quality.""",

            "multilingual_support": """You are SafeServe AI, a multilingual customer service assistant. You can:

- Communicate effectively across languages
- Understand cultural context in customer interactions
- Provide culturally appropriate responses
- Handle language-specific financial terminology
- Adapt communication style to local customs

Remember to:
- Use simple, clear language
- Avoid complex financial jargon
- Be patient with language barriers
- Provide examples when helpful
- Respect cultural differences in communication styles"""
        }
    
    def _create_conversation_prompt(self, message: str, user_id: str, conversation_type: str = "customer_service") -> str:
        """Create a conversation prompt with context"""
        system_prompt = self.system_prompts.get(conversation_type, self.system_prompts["customer_service"])
        
        # Get conversation history for context
        history = self.conversation_history.get(user_id, [])
        
        # Build conversation context
        context = ""
        if history:
            context = "\n\nRecent conversation history:\n"
            for i, msg in enumerate(history[-5:], 1):  # Last 5 messages
                context += f"{i}. {msg['role']}: {msg['content']}\n"
        
        prompt = f"""{system_prompt}

{context}

Current customer message: "{message}"

Please provide a helpful, professional response:"""
        
        return prompt
    
    def _update_conversation_history(self, user_id: str, user_message: str, ai_response: str):
        """Update conversation history for user"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        history = self.conversation_history[user_id]
        
        # Add user message
        history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Add AI response
        history.append({
            "role": "assistant", 
            "content": ai_response,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Keep only last 20 messages
        if len(history) > 20:
            self.conversation_history[user_id] = history[-20:]
    
    def chat(self, message: str, user_id: str = "anonymous", conversation_type: str = "customer_service", 
             temperature: float = 0.7, max_length: int = 512) -> ChatResponse:
        """
        Send message to LLM and get response
        
        Args:
            message: User message
            user_id: User identifier for conversation tracking
            conversation_type: Type of conversation (customer_service, fraud_analysis, multilingual_support)
            temperature: Response creativity (0.0-1.0)
            max_length: Maximum response length
            
        Returns:
            ChatResponse object
        """
        start_time = datetime.datetime.now()
        
        try:
            # Create conversation prompt
            prompt = self._create_conversation_prompt(message, user_id, conversation_type)
            
            # Make API request
            response = self._make_api_request(prompt, temperature, max_length)
            
            if response:
                # Update conversation history
                self._update_conversation_history(user_id, message, response)
                
                # Calculate processing time
                processing_time = (datetime.datetime.now() - start_time).total_seconds()
                
                return ChatResponse(
                    response=response,
                    model_used=self.model_name,
                    timestamp=datetime.datetime.now().isoformat(),
                    processing_time=processing_time,
                    confidence=0.85  # Base confidence
                )
            else:
                return self._create_error_response("Failed to get response from LLM", start_time)
                
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return self._create_error_response(str(e), start_time)
    
    def _make_api_request(self, prompt: str, temperature: float, max_length: int) -> Optional[str]:
        """Make API request to LLM with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.llm_api_url,
                    json={
                        "query": prompt,
                        "temperature": temperature,
                        "max_length": max_length
                    },
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    logger.warning(f"API request failed with status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    # Wait before retry
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _create_error_response(self, error_message: str, start_time: datetime.datetime) -> ChatResponse:
        """Create error response"""
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            response="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
            model_used=self.model_name,
            timestamp=datetime.datetime.now().isoformat(),
            processing_time=processing_time,
            confidence=0.0,
            error=error_message
        )
    
    async def chat_async(self, message: str, user_id: str = "anonymous", 
                        conversation_type: str = "customer_service", 
                        temperature: float = 0.7, max_length: int = 512) -> ChatResponse:
        """
        Async version of chat method
        """
        start_time = datetime.datetime.now()
        
        try:
            prompt = self._create_conversation_prompt(message, user_id, conversation_type)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.llm_api_url,
                    json={
                        "query": prompt,
                        "temperature": temperature,
                        "max_length": max_length
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result.get("response", "").strip()
                        
                        self._update_conversation_history(user_id, message, ai_response)
                        
                        processing_time = (datetime.datetime.now() - start_time).total_seconds()
                        
                        return ChatResponse(
                            response=ai_response,
                            model_used=self.model_name,
                            timestamp=datetime.datetime.now().isoformat(),
                            processing_time=processing_time,
                            confidence=0.85
                        )
                    else:
                        return self._create_error_response(f"API error: {response.status}", start_time)
                        
        except Exception as e:
            logger.error(f"Error in async chat: {str(e)}")
            return self._create_error_response(str(e), start_time)
    
    def analyze_sentiment(self, message: str) -> Dict[str, Any]:
        """
        Analyze sentiment of user message
        
        Args:
            message: User message to analyze
            
        Returns:
            Dictionary with sentiment analysis
        """
        sentiment_prompt = f"""Analyze the sentiment and emotional tone of this customer message:

"{message}"

Provide analysis in this format:
- Sentiment: positive/negative/neutral
- Emotion: (primary emotion detected)
- Urgency: low/medium/high
- Satisfaction: satisfied/neutral/dissatisfied
- Tone: professional/casual/aggressive/polite

Response:"""
        
        try:
            response = self._make_api_request(sentiment_prompt, 0.3, 200)
            
            if response:
                return {
                    "sentiment_analysis": response,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            else:
                return {
                    "sentiment_analysis": "Unable to analyze sentiment",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                "sentiment_analysis": f"Error: {str(e)}",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def get_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get summary of conversation history for user
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with conversation summary
        """
        history = self.conversation_history.get(user_id, [])
        
        if not history:
            return {
                "summary": "No conversation history found",
                "message_count": 0,
                "last_interaction": None
            }
        
        user_messages = [msg for msg in history if msg["role"] == "user"]
        ai_messages = [msg for msg in history if msg["role"] == "assistant"]
        
        # Create summary prompt
        recent_messages = history[-10:]  # Last 10 messages
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
        
        summary_prompt = f"""Summarize this customer service conversation:

{conversation_text}

Provide a brief summary including:
- Main topics discussed
- Customer's primary concerns
- Resolution status
- Any pending issues

Summary:"""
        
        try:
            summary_response = self._make_api_request(summary_prompt, 0.3, 300)
            
            return {
                "summary": summary_response or "Unable to generate summary",
                "message_count": len(history),
                "user_messages": len(user_messages),
                "ai_messages": len(ai_messages),
                "last_interaction": history[-1]["timestamp"] if history else None,
                "conversation_duration": self._calculate_conversation_duration(history)
            }
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "message_count": len(history),
                "last_interaction": history[-1]["timestamp"] if history else None
            }
    
    def _calculate_conversation_duration(self, history: List[Dict]) -> Optional[str]:
        """Calculate total conversation duration"""
        if len(history) < 2:
            return None
        
        try:
            start_time = datetime.datetime.fromisoformat(history[0]["timestamp"])
            end_time = datetime.datetime.fromisoformat(history[-1]["timestamp"])
            duration = end_time - start_time
            
            return str(duration).split('.')[0]  # Remove microseconds
            
        except Exception:
            return None
    
    def clear_conversation_history(self, user_id: str = None):
        """Clear conversation history for user or all users"""
        if user_id:
            if user_id in self.conversation_history:
                del self.conversation_history[user_id]
                logger.info(f"Cleared conversation history for user {user_id}")
        else:
            self.conversation_history = {}
            logger.info("Cleared all conversation history")
    
    def get_active_users(self) -> List[str]:
        """Get list of users with active conversations"""
        return list(self.conversation_history.keys())
    
    def health_check(self) -> Dict[str, Any]:
        """Check LLM API health"""
        try:
            test_response = self._make_api_request("Hello, this is a health check.", 0.5, 50)
            
            return {
                "status": "healthy" if test_response else "unhealthy",
                "api_url": self.llm_api_url,
                "model": self.model_name,
                "response_received": test_response is not None,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "api_url": self.llm_api_url,
                "model": self.model_name,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

# Example usage
if __name__ == "__main__":
    # Initialize LLM interface
    llm_interface = LLMChatInterface("http://localhost:8000/chat")
    
    # Test messages
    test_messages = [
        "Hello, I have a problem with my account",
        "I made a transaction but it's not showing up",
        "Can you help me with a refund?",
        "I think someone is trying to scam me",
        "Thank you for your help"
    ]
    
    print("💬 Testing LLM Chat Interface:")
    print("=" * 50)
    
    user_id = "test_user"
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. User: {message}")
        
        # Get chat response
        response = llm_interface.chat(message, user_id)
        
        print(f"   AI: {response.response}")
        print(f"   Processing Time: {response.processing_time:.2f}s")
        print(f"   Confidence: {response.confidence:.2f}")
        
        if response.error:
            print(f"   Error: {response.error}")
        
        print("-" * 50)
    
    # Test sentiment analysis
    print("\n🎭 Testing Sentiment Analysis:")
    sentiment_result = llm_interface.analyze_sentiment("I'm really frustrated with this service!")
    print(f"Sentiment Analysis: {sentiment_result['sentiment_analysis']}")
    
    # Test conversation summary
    print("\n📊 Testing Conversation Summary:")
    summary = llm_interface.get_conversation_summary(user_id)
    print(f"Summary: {summary['summary']}")
    print(f"Message Count: {summary['message_count']}")
    print(f"Duration: {summary.get('conversation_duration', 'N/A')}")
    
    # Test health check
    print("\n🏥 Testing Health Check:")
    health = llm_interface.health_check()
    print(f"Status: {health['status']}")
    print(f"Response Received: {health['response_received']}")
    
    print(f"\nActive Users: {llm_interface.get_active_users()}")