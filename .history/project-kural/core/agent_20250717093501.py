"""
Central Agent Orchestrator for Project Kural

This module contains the main KuralAgent class that coordinates all components:
persona management, memory integration, tool usage, and LLM interaction.
"""

import os
import logging
import requests
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatOpenRouter:
    """
    Custom ChatOpenRouter implementation for OpenRouter API integration.
    """
    
    def __init__(self, api_key: str, model: str = "mistralai/mistral-7b-instruct"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Invoke the OpenRouter API with messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dict containing the response
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Convert messages to OpenRouter format
            api_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    api_messages.append(msg)
                else:
                    # Handle LangChain message objects
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    role = "user" if "Human" in str(type(msg)) else "assistant"
                    api_messages.append({"role": role, "content": content})
            
            data = {
                "model": self.model,
                "messages": api_messages,
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return {
                    "content": response_data["choices"][0]["message"]["content"],
                    "usage": response_data.get("usage", {}),
                    "model": self.model
                }
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return {"content": "I apologize, but I'm having trouble processing your request right now. Please try again.", "error": True}
                
        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            return {"content": "I apologize, but I'm experiencing technical difficulties. Please try again.", "error": True}


class KuralAgent:
    """
    Main agent orchestrator that combines all Project Kural components.
    """
    
    def __init__(self, openrouter_api_key: str, tools: List[BaseTool]):
        """
        Initialize the Kural Agent.
        
        Args:
            openrouter_api_key (str): OpenRouter API key for LLM access
            tools (List[BaseTool]): List of tools the agent can use
        """
        self.openrouter_api_key = openrouter_api_key
        self.tools = tools
        self.llm = ChatOpenRouter(api_key=openrouter_api_key)
        
        # Define persona file mappings
        self.persona_mappings = {
            "Negative": "personas/empathetic_deescalation.txt",
            "Positive": "personas/efficient_friendly.txt", 
            "Neutral": "personas/professional_direct.txt"
        }
        
        logger.info("KuralAgent initialized successfully")
    
    def _load_persona_prompt(self, sentiment: str) -> str:
        """
        Load the appropriate persona prompt based on sentiment.
        
        Args:
            sentiment (str): Detected sentiment ('Negative', 'Positive', 'Neutral')
            
        Returns:
            str: Persona prompt content
        """
        persona_file = self.persona_mappings.get(sentiment, self.persona_mappings["Neutral"])
        
        try:
            with open(persona_file, 'r', encoding='utf-8') as f:
                persona_prompt = f.read().strip()
            
            logger.info(f"Loaded {sentiment} persona from {persona_file}")
            return persona_prompt
            
        except FileNotFoundError:
            logger.error(f"Persona file not found: {persona_file}")
            # Return a default professional persona
            return """
            You are a professional customer service representative. 
            Provide helpful, accurate, and courteous assistance to customers.
            Be clear, concise, and focus on resolving their issues effectively.
            """
        except Exception as e:
            logger.error(f"Error loading persona file {persona_file}: {e}")
            return "You are a helpful customer service representative."
    
    def _construct_master_prompt(self, persona_prompt: str, long_term_summary: str, 
                               language: str) -> str:
        """
        Construct the master system prompt combining persona, context, and instructions.
        
        Args:
            persona_prompt (str): The persona-specific prompt
            long_term_summary (str): User's conversation history summary
            language (str): Detected language for response
            
        Returns:
            str: Complete system prompt
        """
        # Language response instructions
        language_instructions = {
            "en": "Respond in English.",
            "ta": "Respond in Tamil (தமிழ்).",
            "hi": "Respond in Hindi (हिंदी).",
            "es": "Respond in Spanish.",
            "fr": "Respond in French."
        }
        
        language_instruction = language_instructions.get(language, "Respond in English.")
        
        # Construct the master prompt
        master_prompt = f"""
{persona_prompt}

IMPORTANT CONTEXT:
{f"Previous context with this user: {long_term_summary}" if long_term_summary else "This is a new customer interaction."}

LANGUAGE INSTRUCTION:
{language_instruction}

AVAILABLE TOOLS:
You have access to tools that can help you:
- get_billing_info: Retrieve customer billing information
- check_network_status: Check network status for specific area codes

RESPONSE GUIDELINES:
1. Always be helpful and professional
2. Use tools when appropriate to provide accurate information
3. Respond in the detected language ({language})
4. Follow the persona guidelines above
5. If you cannot help with something, explain why and suggest alternatives
6. Keep responses conversational and natural

Remember: You are representing the company, so maintain high standards of service while following your persona guidelines.
"""
        
        return master_prompt.strip()
    
    def _simple_agent_execution(self, user_input: str, master_prompt: str) -> str:
        """
        Execute a simple agent interaction without complex ReAct patterns.
        
        Args:
            user_input (str): User's input message
            master_prompt (str): Complete system prompt
            
        Returns:
            str: Agent's response
        """
        try:
            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": master_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            if response.get("error"):
                return response["content"]
            
            agent_response = response["content"]
            
            # Check if the agent wants to use tools
            if "get_billing_info" in agent_response.lower() or "billing" in user_input.lower():
                # Extract user ID and call billing tool
                for tool in self.tools:
                    if tool.name == "get_billing_info":
                        # For demo, use a default user ID
                        billing_info = tool.invoke({"user_id": "DEMO_USER"})
                        agent_response += f"\n\n{billing_info}"
                        break
            
            if "check_network_status" in agent_response.lower() or "network" in user_input.lower():
                # Extract area code and call network tool
                for tool in self.tools:
                    if tool.name == "check_network_status":
                        # For demo, use a default area code
                        network_info = tool.invoke({"area_code": "555"})
                        agent_response += f"\n\n{network_info}"
                        break
            
            return agent_response
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again or contact support for assistance."
    
    def run(self, user_id: str, user_input: str, language: str = "en", 
            sentiment: str = "Neutral", short_term_memory: Optional[ConversationBufferMemory] = None,
            long_term_summary: str = "") -> str:
        """
        Main agent execution method that coordinates all components.
        
        Args:
            user_id (str): User identifier
            user_input (str): User's input message
            language (str): Detected language code
            sentiment (str): Detected sentiment
            short_term_memory (Optional[ConversationBufferMemory]): Session memory
            long_term_summary (str): User's conversation history summary
            
        Returns:
            str: Agent's response
        """
        try:
            logger.info(f"Processing request for user {user_id} with sentiment {sentiment}")
            
            # Load appropriate persona
            persona_prompt = self._load_persona_prompt(sentiment)
            
            # Construct master system prompt
            master_prompt = self._construct_master_prompt(
                persona_prompt, long_term_summary, language
            )
            
            # Execute agent with simplified approach
            response = self._simple_agent_execution(user_input, master_prompt)
            
            # Update short-term memory if provided
            if short_term_memory:
                try:
                    short_term_memory.chat_memory.add_user_message(user_input)
                    short_term_memory.chat_memory.add_ai_message(response)
                except Exception as e:
                    logger.warning(f"Failed to update short-term memory: {e}")
            
            logger.info(f"Successfully processed request for user {user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Agent execution failed for user {user_id}: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again or contact our support team for assistance."
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List[str]: List of tool names
        """
        return [tool.name for tool in self.tools]
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the agent components.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        health_status = {
            "agent_initialized": True,
            "tools_available": len(self.tools),
            "tool_names": self.get_available_tools(),
            "persona_files": {},
            "api_key_present": bool(self.openrouter_api_key)
        }
        
        # Check persona files
        for sentiment, file_path in self.persona_mappings.items():
            health_status["persona_files"][sentiment] = os.path.exists(file_path)
        
        return health_status