"""
SafeServe AI - LLM Chat Interface
Clean interface to communicate with external LLM (Deepseek/Mistral via ngrok)
"""

import requests
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    """Chat response structure"""
    response: str
    model_used: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class LLMChatInterface:
    """
    Interface to communicate with external LLM services
    """
    
    def __init__(self, llm_api_url: str, timeout: int = 30):
        self.llm_api_url = llm_api_url
        self.timeout = timeout
        self.session = requests.Session()
        
        # Customer service context
        self.system_context = """You are SafeServe AI, an intelligent customer service assistant specializing in banking and financial security. You help customers with:

- Account inquiries and support
- Transaction disputes and fraud concerns
- Security questions and verification
- General banking assistance
- Fraud prevention education

Guidelines:
- Be helpful, professional, and empathetic
- Provide clear, actionable advice
- If fraud is suspected, guide the customer to appropriate security measures
- Never ask for sensitive information like passwords or PINs
- Keep responses concise but informative
- Use a reassuring tone for security concerns"""
    
    def create_customer_service_prompt(self, user_query: str, context: str = "") -> str:
        """
        Create a specialized prompt for customer service
        
        Args:
            user_query: Customer's question or concern
            context: Additional context (previous conversation, user info)
            
        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""{self.system_context}

{f"Previous context: {context}" if context else ""}

Customer: {user_query}

SafeServe AI:"""
        
        return prompt
    
    def chat(self, query: str, context: str = "", temperature: float = 0.7, max_length: int = 512) -> ChatResponse:
        """
        Send chat request to LLM
        
        Args:
            query: User query
            context: Previous conversation context
            temperature: Response randomness (0.0-1.0)
            max_length: Maximum response length
            
        Returns:
            ChatResponse object
        """
        start_time = time.time()
        
        try:
            # Create customer service prompt
            prompt = self.create_customer_service_prompt(query, context)
            
            # Prepare request
            request_data = {
                "query": prompt,
                "temperature": temperature,
                "max_length": max_length
            }
            
            # Make API request
            response = self.session.post(
                self.llm_api_url,
                json=request_data,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract response text
                if "response" in response_data:
                    response_text = response_data["response"]
                    model_used = response_data.get("model_used", "unknown")
                    
                    # Clean up response
                    response_text = self._clean_response(response_text)
                    
                    return ChatResponse(
                        response=response_text,
                        model_used=model_used,
                        processing_time=processing_time,
                        success=True
                    )
                else:
                    return ChatResponse(
                        response="I apologize, but I received an invalid response format.",
                        model_used="unknown",
                        processing_time=processing_time,
                        success=False,
                        error_message="Invalid response format"
                    )
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                
                return ChatResponse(
                    response=self._get_fallback_response(query),
                    model_used="fallback",
                    processing_time=processing_time,
                    success=False,
                    error_message=error_msg
                )
                
        except requests.exceptions.Timeout:
            processing_time = time.time() - start_time
            return ChatResponse(
                response="I apologize for the delay. Please try again in a moment.",
                model_used="fallback",
                processing_time=processing_time,
                success=False,
                error_message="Request timeout"
            )
            
        except requests.exceptions.ConnectionError:
            processing_time = time.time() - start_time
            return ChatResponse(
                response="I'm currently experiencing connection issues. Please try again shortly.",
                model_used="fallback",
                processing_time=processing_time,
                success=False,
                error_message="Connection error"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            
            return ChatResponse(
                response=self._get_fallback_response(query),
                model_used="fallback",
                processing_time=processing_time,
                success=False,
                error_message=error_msg
            )
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the LLM response"""
        # Remove any system artifacts
        response = response.replace("SafeServe AI:", "").strip()
        response = response.replace("Assistant:", "").strip()
        
        # Remove extra whitespace
        response = " ".join(response.split())
        
        # Ensure proper sentence ending
        if response and not response.endswith(('.', '!', '?')):
            response += "."
        
        return response
    
    def _get_fallback_response(self, query: str) -> str:
        """Generate fallback response when LLM is unavailable"""
        query_lower = query.lower()
        
        # Simple keyword-based responses
        if any(word in query_lower for word in ['fraud', 'suspicious', 'scam', 'security']):
            return "I understand you have security concerns. Please contact our fraud department immediately at 1-800-FRAUD or visit your nearest branch. For immediate safety, monitor your accounts closely and report any unauthorized transactions."
        
        elif any(word in query_lower for word in ['account', 'balance', 'statement']):
            return "For account inquiries, you can check your balance through our mobile app, online banking, or by calling our customer service line. If you need assistance with specific transactions, please have your account details ready."
        
        elif any(word in query_lower for word in ['refund', 'dispute', 'transaction']):
            return "For transaction disputes or refund requests, please provide the transaction details including date, amount, and merchant. We typically process dispute requests within 3-5 business days."
        
        elif any(word in query_lower for word in ['help', 'support', 'assistance']):
            return "I'm here to help! You can ask me about account inquiries, transaction issues, security concerns, or general banking questions. How can I assist you today?"
        
        else:
            return "Thank you for contacting SafeServe AI. I'm currently experiencing technical difficulties, but I'm here to help with your banking and security needs. Please try rephrasing your question or contact our customer service team for immediate assistance."
    
    def health_check(self) -> bool:
        """Check if LLM service is available"""
        try:
            # Try to get health endpoint
            health_url = self.llm_api_url.replace('/chat', '/health')
            response = self.session.get(health_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the LLM service"""
        try:
            # Try to get root endpoint
            root_url = self.llm_api_url.replace('/chat', '/')
            response = self.session.get(root_url, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unavailable", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}

class ConversationManager:
    """
    Manages conversation context and history
    """
    
    def __init__(self, max_context_length: int = 5):
        self.conversations = {}
        self.max_context_length = max_context_length
    
    def add_message(self, user_id: str, message: str, is_user: bool = True):
        """Add message to conversation history"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            'message': message,
            'is_user': is_user,
            'timestamp': time.time()
        })
        
        # Keep only recent messages
        if len(self.conversations[user_id]) > self.max_context_length * 2:
            self.conversations[user_id] = self.conversations[user_id][-self.max_context_length * 2:]
    
    def get_context(self, user_id: str) -> str:
        """Get conversation context for user"""
        if user_id not in self.conversations:
            return ""
        
        messages = self.conversations[user_id][-self.max_context_length * 2:]
        context_parts = []
        
        for msg in messages:
            role = "Customer" if msg['is_user'] else "SafeServe AI"
            context_parts.append(f"{role}: {msg['message']}")
        
        return " | ".join(context_parts)
    
    def clear_conversation(self, user_id: str):
        """Clear conversation history for user"""
        if user_id in self.conversations:
            del self.conversations[user_id]
    
    def get_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of conversation"""
        if user_id not in self.conversations:
            return {"message_count": 0, "duration": 0}
        
        messages = self.conversations[user_id]
        if not messages:
            return {"message_count": 0, "duration": 0}
        
        start_time = messages[0]['timestamp']
        end_time = messages[-1]['timestamp']
        duration = end_time - start_time
        
        return {
            "message_count": len(messages),
            "duration": duration,
            "user_messages": sum(1 for msg in messages if msg['is_user']),
            "bot_messages": sum(1 for msg in messages if not msg['is_user'])
        }

# Example usage
if __name__ == "__main__":
    # Initialize LLM interface
    llm_interface = LLMChatInterface("http://localhost:8000/chat")
    conversation_manager = ConversationManager()
    
    print("🤖 Testing LLM Chat Interface:")
    print("=" * 50)
    
    # Test health check
    health_status = llm_interface.health_check()
    print(f"Health Status: {'✅ Online' if health_status else '❌ Offline'}")
    
    # Test service info
    service_info = llm_interface.get_service_info()
    print(f"Service Info: {service_info}")
    
    # Test conversation
    user_id = "test_user"
    test_queries = [
        "Hello, I need help with my account",
        "I noticed a suspicious transaction of $500 yesterday",
        "What should I do about this fraud?"
    ]
    
    print("\n💬 Testing Conversation:")
    print("-" * 30)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Customer: {query}")
        
        # Add to conversation
        conversation_manager.add_message(user_id, query, is_user=True)
        
        # Get context
        context = conversation_manager.get_context(user_id)
        
        # Get response
        response = llm_interface.chat(query, context)
        
        # Add response to conversation
        conversation_manager.add_message(user_id, response.response, is_user=False)
        
        print(f"   SafeServe AI: {response.response}")
        print(f"   [Processing: {response.processing_time:.2f}s, Success: {response.success}]")
        
        if not response.success:
            print(f"   [Error: {response.error_message}]")
    
    # Show conversation summary
    summary = conversation_manager.get_conversation_summary(user_id)
    print(f"\n📊 Conversation Summary: {summary}")