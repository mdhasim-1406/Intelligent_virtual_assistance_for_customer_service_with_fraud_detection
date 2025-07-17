"""
Project Kural - Streamlit Frontend Application

Main application that provides the user interface for theer_service_with_fraud_detection/project-kural && python -c"
from core import PerceptionModule, MemoryModule, KuralAgent, get_billing_info, check_network_status
print('??? Core modules imported successfully!')
print('??? Tools available:', [get_billing_info.name, check_network_status.name])
print('??? Project Kural is ready to launch!')
"
Traceback (most recent call last):
  File "<string>", line 2, in <module>
    from core import PerceptionModule, MemoryModule, KuralAgent
, get_billing_info, check_network_status                         File "/home/hasim001/Intelligent_virtual_assistance_for_custo
mer_service_with_fraud_detection/project-kural/core/__init__.py", line 14, in <module>                                            from .tools import get_billing_info, check_network_status
  File "/home/hasim001/Intelligent_virtual_assistance_for_custo
mer_service_with_fraud_detection/project-kural/core/tools.py", line 9, in <module>                                                from typing import str
ImportError: cannot import name 'str' from 'typing' (/usr/lib/p
ython3.13/typing.py)                                                                                                          
 intelligent
multilingual customer service agent with voice and text capabilities.
"""

import streamlit as st
import os
import tempfile
import logging
from gtts import gTTS
from io import BytesIO
import base64
from datetime import datetime

# Import core modules
from core import PerceptionModule, MemoryModule, KuralAgent, get_billing_info, check_network_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Project Kural - AI Customer Service",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize all session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = ""
    
    if 'short_term_memory' not in st.session_state:
        st.session_state.short_term_memory = None
    
    if 'components_initialized' not in st.session_state:
        st.session_state.components_initialized = False
    
    if 'perception_module' not in st.session_state:
        st.session_state.perception_module = None
    
    if 'memory_module' not in st.session_state:
        st.session_state.memory_module = None
    
    if 'kural_agent' not in st.session_state:
        st.session_state.kural_agent = None
    
    if 'conversation_count' not in st.session_state:
        st.session_state.conversation_count = 0

def get_api_key():
    """Get OpenRouter API key from secrets or user input."""
    try:
        # Try to get from Streamlit secrets first
        if hasattr(st, 'secrets') and 'OPENROUTER_API_KEY' in st.secrets:
            return st.secrets['OPENROUTER_API_KEY']
    except:
        pass
    
    # Fall back to user input
    api_key = st.sidebar.text_input(
        "OpenRouter API Key",
        type="password",
        help="Enter your OpenRouter API key to use the AI agent"
    )
    
    if not api_key:
        st.sidebar.warning("Please enter your OpenRouter API key to continue.")
        st.sidebar.info("Get your API key from: https://openrouter.ai/")
        return None
    
    return api_key

def initialize_components(api_key):
    """Initialize all core components."""
    try:
        if not st.session_state.components_initialized:
            with st.spinner("Initializing AI components..."):
                # Initialize Perception Module
                st.session_state.perception_module = PerceptionModule()
                
                # Initialize Memory Module
                st.session_state.memory_module = MemoryModule()
                
                # Initialize tools
                tools = [get_billing_info, check_network_status]
                
                # Initialize Kural Agent
                st.session_state.kural_agent = KuralAgent(
                    openrouter_api_key=api_key,
                    tools=tools
                )
                
                # Initialize short-term memory
                st.session_state.short_term_memory = st.session_state.memory_module.get_short_term_memory()
                
                st.session_state.components_initialized = True
                
            st.success("✅ All components initialized successfully!")
            
    except Exception as e:
        st.error(f"❌ Failed to initialize components: {str(e)}")
        logger.error(f"Component initialization failed: {e}")
        return False
    
    return True

def process_voice_input(uploaded_file):
    """Process uploaded voice file and return transcription."""
    if not uploaded_file:
        return None, None, None
    
    try:
        with st.spinner("Processing voice input..."):
            # Save uploaded file
            audio_path = st.session_state.perception_module.save_uploaded_audio(uploaded_file)
            
            if not audio_path:
                st.error("Failed to save audio file")
                return None, None, None
            
            # Transcribe audio
            transcription_result = st.session_state.perception_module.transcribe_audio(audio_path)
            
            # Clean up temporary file
            st.session_state.perception_module.cleanup_temp_file(audio_path)
            
            if 'error' in transcription_result:
                st.error(f"Transcription failed: {transcription_result['error']}")
                return None, None, None
            
            text = transcription_result['text']
            language = transcription_result['language']
            language_name = transcription_result.get('language_name', language)
            
            if text:
                st.info(f"🎤 Transcribed ({language_name}): {text}")
                return text, language, language_name
            else:
                st.warning("No speech detected in the audio file")
                return None, None, None
                
    except Exception as e:
        st.error(f"Voice processing failed: {str(e)}")
        logger.error(f"Voice processing error: {e}")
        return None, None, None

def analyze_sentiment(text, api_key):
    """Analyze sentiment of the input text."""
    try:
        with st.spinner("Analyzing sentiment..."):
            sentiment = st.session_state.perception_module.analyze_sentiment(text, api_key)
            return sentiment
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return "Neutral"

def generate_audio_response(text, language="en"):
    """Generate audio response using gTTS."""
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save to BytesIO object
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        return None

def display_chat_message(role, content, audio_data=None):
    """Display a chat message with optional audio."""
    with st.chat_message(role):
        st.write(content)
        
        if audio_data:
            st.audio(audio_data, format='audio/mp3')

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🎯 Project Kural")
    st.subtitle("Intelligent Multilingual Customer Service Agent")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        st.stop()
    
    # Initialize components
    if not initialize_components(api_key):
        st.stop()
    
    # User ID input
    user_id = st.sidebar.text_input(
        "User ID",
        value=st.session_state.user_id,
        placeholder="Enter your customer ID"
    )
    
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
        # Reset chat history when user changes
        st.session_state.chat_history = []
        st.session_state.conversation_count = 0
    
    # Display user stats if user ID is provided
    if user_id:
        user_stats = st.session_state.memory_module.get_user_stats(user_id)
        st.sidebar.info(f"""
        **User Statistics:**
        - Previous conversations: {user_stats['conversation_count']}
        - Has history: {'Yes' if user_stats['has_summary'] else 'No'}
        - Last updated: {user_stats['last_updated'][:10] if user_stats['last_updated'] else 'Never'}
        """)
    
    # Agent health check
    if st.sidebar.button("🔍 System Health Check"):
        health = st.session_state.kural_agent.health_check()
        st.sidebar.json(health)
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(
            message["role"],
            message["content"],
            message.get("audio_data")
        )
    
    # Input methods
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎤 Voice Input")
        uploaded_audio = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'ogg', 'm4a'],
            help="Record and upload your voice message"
        )
        
        if uploaded_audio and st.button("Process Voice", key="process_voice"):
            if not user_id:
                st.error("Please enter a User ID first")
            else:
                text, language, language_name = process_voice_input(uploaded_audio)
                if text:
                    # Process the voice input
                    process_user_input(text, language, user_id, api_key)
    
    with col2:
        st.subheader("⌨️ Text Input")
        text_input = st.text_area(
            "Type your message",
            placeholder="Enter your message here...",
            height=100
        )
        
        if st.button("Send Message", key="send_text") and text_input:
            if not user_id:
                st.error("Please enter a User ID first")
            else:
                # Process text input (default to English)
                process_user_input(text_input, "en", user_id, api_key)
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.conversation_count = 0
        st.rerun()

def process_user_input(user_input, language, user_id, api_key):
    """Process user input through the AI pipeline."""
    try:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Analyze sentiment
        sentiment = analyze_sentiment(user_input, api_key)
        
        # Get long-term memory
        long_term_summary = st.session_state.memory_module.get_long_term_summary(user_id)
        
        # Generate response using Kural Agent
        with st.spinner("Generating response..."):
            response = st.session_state.kural_agent.run(
                user_id=user_id,
                user_input=user_input,
                language=language,
                sentiment=sentiment,
                short_term_memory=st.session_state.short_term_memory,
                long_term_summary=long_term_summary
            )
        
        # Generate audio response
        audio_data = generate_audio_response(response, language)
        
        # Add agent response to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "sentiment": sentiment,
            "language": language,
            "audio_data": audio_data,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update conversation count
        st.session_state.conversation_count += 1
        
        # Save conversation summary periodically
        if st.session_state.conversation_count % 3 == 0:
            try:
                chat_messages = st.session_state.short_term_memory.chat_memory.messages
                st.session_state.memory_module.save_conversation_summary(
                    user_id, chat_messages, api_key
                )
            except Exception as e:
                logger.warning(f"Failed to save conversation summary: {e}")
        
        # Display the new messages
        display_chat_message("user", user_input)
        display_chat_message("assistant", response, audio_data)
        
        # Show sentiment and language info
        st.info(f"📊 Detected: {sentiment} sentiment, {language} language")
        
    except Exception as e:
        st.error(f"❌ Error processing input: {str(e)}")
        logger.error(f"Input processing failed: {e}")

if __name__ == "__main__":
    main()