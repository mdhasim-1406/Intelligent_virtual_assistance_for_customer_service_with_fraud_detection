"""
Project Kural - Streamlit Frontend Application

Main application that provides the user interface for the
intelligent multilingual customer service agent with voice and text capabilities.
"""

import streamlit as st
import os
import tempfile
import logging
from gtts import gTTS
from io import BytesIO
import base64
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define BASE_DIR for cross-platform file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import core modules with explicit imports
from core.perception import PerceptionModule
from core.memory import MemoryModule
from core.agent import KuralAgent
from core.tools import get_billing_info, check_network_status
from core.vector_store import initialize_knowledge_base

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

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e8f4f8;
        border-left-color: #28a745;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left-color: #1f77b4;
    }
    .stButton > button {
        width: 100%;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_vector_store():
    """Initialize and cache the vector store for knowledge base."""
    try:
        with st.spinner("Loading knowledge base... This may take a moment on first run."):
            vector_store = initialize_knowledge_base()
            st.success("✅ Knowledge base loaded successfully!")
            return vector_store
    except ConnectionError as e:
        logger.error(f"Connection error initializing vector store: {e}")
        st.error(f"🔴 **Knowledge Base Connection Error**\n\n{str(e)}")
        st.info("💡 **Quick Fix**: Check your internet connection and try refreshing the page.")
        return None
    except RuntimeError as e:
        logger.error(f"Runtime error initializing vector store: {e}")
        st.error(f"🔴 **Knowledge Base Runtime Error**\n\n{str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error initializing vector store: {e}")
        st.error(f"🔴 **Knowledge Base Initialization Failed**\n\n{str(e)}")
        return None

@st.cache_resource
def initialize_perception_module():
    """Initialize and cache the perception module."""
    try:
        perception = PerceptionModule()
        st.success("✅ Perception module (Whisper) loaded successfully!")
        return perception
    except RuntimeError as e:
        logger.error(f"Runtime error initializing perception module: {e}")
        st.error(f"🔴 **Voice Processing Initialization Error**\n\n{str(e)}")
        st.info("💡 **Solutions**: Install missing dependencies or check your internet connection.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error initializing perception module: {e}")
        st.error(f"🔴 **Voice Processing Failed**\n\n{str(e)}")
        return None

@st.cache_resource
def initialize_memory_module():
    """Initialize and cache the memory module."""
    try:
        memory = MemoryModule()
        return memory
    except Exception as e:
        logger.error(f"Failed to initialize memory module: {e}")
        st.error(f"🔴 **Memory Module Initialization Failed**\n\n{str(e)}")
        return None

def initialize_session_state():
    """Initialize all session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = ""
    
    if 'conversation_count' not in st.session_state:
        st.session_state.conversation_count = 0
    
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False
    
    if 'kural_agent' not in st.session_state:
        st.session_state.kural_agent = None
    
    if 'short_term_memory' not in st.session_state:
        st.session_state.short_term_memory = None

def get_api_key():
    """Get OpenRouter API key from environment or user input."""
    # First try to get from environment
    api_key = os.environ.get('OPENROUTER_API_KEY')
    
    if api_key:
        st.sidebar.success("🔑 API Key loaded from environment")
        return api_key
    
    # Fall back to user input
    api_key = st.sidebar.text_input(
        "🔑 OpenRouter API Key",
        type="password",
        help="Enter your OpenRouter API key to use the AI agent",
        placeholder="sk-or-v1-..."
    )
    
    if not api_key:
        st.sidebar.error("❌ Please enter your OpenRouter API key to continue.")
        st.sidebar.info("💡 Get your API key from: https://openrouter.ai/")
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Alternative**: Set the `OPENROUTER_API_KEY` environment variable")
        return None
    
    return api_key

def initialize_agent(api_key):
    """Initialize the Kural Agent with all components."""
    try:
        if not st.session_state.agent_initialized:
            # Initialize core components with detailed error handling
            st.info("🔄 **Initializing Project Kural Components...**")
            
            # Initialize perception module
            with st.spinner("Loading voice processing (Whisper)..."):
                perception = initialize_perception_module()
                if not perception:
                    st.warning("⚠️ Voice processing unavailable - text-only mode enabled")
                    
            # Initialize memory module
            with st.spinner("Loading memory system..."):
                memory = initialize_memory_module()
                if not memory:
                    st.error("❌ Memory system failed to initialize")
                    return False
                    
            # Initialize vector store (knowledge base)
            with st.spinner("Loading knowledge base..."):
                vector_store = initialize_vector_store()
                if not vector_store:
                    st.warning("⚠️ Knowledge base unavailable - using general responses only")
            
            # Initialize tools
            tools = [get_billing_info, check_network_status]
            
            # Initialize Kural Agent
            try:
                st.session_state.kural_agent = KuralAgent(
                    openrouter_api_key=api_key,
                    tools=tools,
                    vector_store=vector_store
                )
                
                # Initialize short-term memory
                st.session_state.short_term_memory = memory.get_short_term_memory()
                
                # Store components in session state
                st.session_state.perception_module = perception
                st.session_state.memory_module = memory
                st.session_state.vector_store = vector_store
                
                st.session_state.agent_initialized = True
                
                # Show component status
                status_msg = "✅ **System Initialized Successfully!**\n\n"
                status_msg += f"- Voice Processing: {'✅ Available' if perception else '❌ Unavailable'}\n"
                status_msg += f"- Memory System: {'✅ Available' if memory else '❌ Unavailable'}\n"
                status_msg += f"- Knowledge Base: {'✅ Available' if vector_store else '❌ Unavailable'}\n"
                status_msg += f"- Tools: ✅ {len(tools)} tools loaded\n"
                status_msg += f"- Agent: ✅ Ready for interaction"
                
                st.success(status_msg)
                
            except ValueError as e:
                st.error(f"🔴 **Agent Configuration Error**\n\n{str(e)}")
                return False
            except Exception as e:
                st.error(f"🔴 **Agent Initialization Failed**\n\n{str(e)}")
                return False
            
        return True
        
    except Exception as e:
        st.error(f"🔴 **Critical System Error**\n\n{str(e)}")
        logger.error(f"Agent initialization failed: {e}")
        return False

def process_voice_input(uploaded_file):
    """Process uploaded voice file and return transcription."""
    if not uploaded_file:
        return None, None, None
    
    try:
        with st.spinner("🎤 Processing voice input..."):
            # Save uploaded file to temporary location
            audio_path = st.session_state.perception_module.save_uploaded_audio(uploaded_file)
            
            if not audio_path:
                st.error("❌ Failed to save audio file")
                return None, None, None
            
            # Transcribe audio
            transcription_result = st.session_state.perception_module.transcribe_audio(audio_path)
            
            # Clean up temporary file
            st.session_state.perception_module.cleanup_temp_file(audio_path)
            
            if 'error' in transcription_result:
                st.error(f"❌ Transcription failed: {transcription_result['error']}")
                return None, None, None
            
            text = transcription_result['text']
            language = transcription_result['language']
            language_name = transcription_result.get('language_name', language)
            
            if text:
                st.info(f"🎤 **Transcribed** ({language_name}): {text}")
                return text, language, language_name
            else:
                st.warning("⚠️ No speech detected in the audio file")
                return None, None, None
                
    except Exception as e:
        st.error(f"❌ Voice processing failed: {str(e)}")
        logger.error(f"Voice processing error: {e}")
        return None, None, None

def analyze_sentiment(text):
    """Analyze sentiment of the input text."""
    try:
        sentiment = st.session_state.perception_module.analyze_sentiment(text)
        return sentiment
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return "Neutral"

def generate_audio_response(text, language="en"):
    """Generate audio response using gTTS."""
    try:
        # Map language codes to gTTS supported codes
        lang_map = {
            "en": "en",
            "ta": "ta",
            "hi": "hi",
            "es": "es",
            "fr": "fr"
        }
        
        gtts_lang = lang_map.get(language, "en")
        
        # Create gTTS object
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        
        # Save to BytesIO object
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        return None

def display_chat_history():
    """Display the chat history with proper formatting."""
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display metadata for assistant messages
            if message["role"] == "assistant":
                col1, col2 = st.columns([1, 1])
                with col1:
                    if "sentiment" in message:
                        sentiment_color = {
                            "Positive": "🟢",
                            "Negative": "🔴", 
                            "Neutral": "🟡"
                        }
                        st.caption(f"{sentiment_color.get(message['sentiment'], '⚪')} Sentiment: {message['sentiment']}")
                
                with col2:
                    if "language" in message:
                        st.caption(f"🌍 Language: {message['language']}")
                
                # Play audio if available
                if "audio_data" in message and message["audio_data"]:
                    st.audio(message["audio_data"], format='audio/mp3')

def process_user_input(user_input, language="en"):
    """Process user input through the AI pipeline."""
    try:
        if not st.session_state.user_id:
            st.error("❌ Please enter a User ID in the sidebar first")
            return
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Process input with progress indicators
        with st.spinner("🧠 Analyzing sentiment..."):
            sentiment = analyze_sentiment(user_input)
        
        with st.spinner("🔍 Retrieving context..."):
            # Get long-term memory
            long_term_summary = st.session_state.memory_module.get_long_term_summary(st.session_state.user_id)
        
        with st.spinner("💭 Generating response..."):
            # Generate response using Kural Agent
            response = st.session_state.kural_agent.run(
                user_id=st.session_state.user_id,
                user_input=user_input,
                language=language,
                sentiment=sentiment,
                short_term_memory=st.session_state.short_term_memory,
                long_term_summary=long_term_summary
            )
        
        with st.spinner("🔊 Generating audio..."):
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
                with st.spinner("💾 Saving conversation..."):
                    chat_messages = st.session_state.short_term_memory.chat_memory.messages
                    st.session_state.memory_module.save_conversation_summary(
                        st.session_state.user_id, chat_messages
                    )
                    st.success("✅ Conversation saved to memory")
            except Exception as e:
                logger.warning(f"Failed to save conversation summary: {e}")
        
        # Show processing summary
        st.success(f"✅ Response generated • Sentiment: {sentiment} • Language: {language}")
        
        # Auto-scroll to bottom
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing input: {str(e)}")
        logger.error(f"Input processing failed: {e}")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Project Kural</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Intelligent Multilingual Customer Service Agent</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        st.warning("⚠️ Please configure your OpenRouter API key in the sidebar to continue.")
        st.info("💡 **Getting Started**: Enter your API key in the sidebar, then provide a User ID to start chatting!")
        st.stop()
    
    # Initialize agent
    if not initialize_agent(api_key):
        st.stop()
    
    # User configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 User Settings")
    
    user_id = st.sidebar.text_input(
        "User ID",
        value=st.session_state.user_id,
        placeholder="Enter your customer ID (e.g., CUST001)",
        help="Unique identifier for personalized memory and conversation history"
    )
    
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
        # Reset chat history when user changes
        st.session_state.chat_history = []
        st.session_state.conversation_count = 0
        st.rerun()
    
    # User statistics
    if user_id:
        user_stats = st.session_state.memory_module.get_user_stats(user_id)
        st.sidebar.info(f"""
        **📊 User Statistics**
        - Previous conversations: {user_stats['conversation_count']}
        - Has conversation history: {'Yes' if user_stats['has_summary'] else 'No'}
        - Last interaction: {user_stats['last_updated'][:10] if user_stats['last_updated'] else 'Never'}
        """)
    
    # System controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Controls")
    
    # Health check
    if st.sidebar.button("🔍 System Health Check"):
        with st.spinner("Checking system health..."):
            health = st.session_state.kural_agent.health_check()
            st.sidebar.success("✅ System Status: Healthy")
            with st.sidebar.expander("📋 Detailed Health Report"):
                st.json(health)
    
    # Clear chat
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.conversation_count = 0
        st.success("✅ Chat history cleared")
        st.rerun()
    
    # Main chat interface
    if not user_id:
        st.info("👋 **Welcome to Project Kural!**\n\nPlease enter your User ID in the sidebar to start chatting.")
        st.stop()
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Conversation")
        display_chat_history()
    else:
        st.info("👋 Hi! I'm Kural, your AI customer service assistant. How can I help you today?")
    
    # Input interface
    st.markdown("---")
    
    # Create two columns for input methods
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎤 Voice Input")
        uploaded_audio = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'ogg', 'm4a'],
            help="Upload your voice message (WAV, MP3, OGG, M4A formats supported)",
            key="audio_upload"
        )
        
        if st.button("🎙️ Process Voice", disabled=not uploaded_audio):
            text, language, language_name = process_voice_input(uploaded_audio)
            if text:
                process_user_input(text, language)
    
    with col2:
        st.subheader("⌨️ Text Input")
        text_input = st.text_area(
            "Type your message",
            placeholder="Enter your message here... (English, Tamil, or Hindi)",
            height=100,
            key="text_input"
        )
        
        if st.button("📤 Send Message", disabled=not text_input.strip()):
            process_user_input(text_input.strip(), "en")
    
    # Chat input (alternative modern interface)
    st.markdown("---")
    user_message = st.chat_input("Type your message here...")
    if user_message:
        process_user_input(user_message, "en")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>🎯 <strong>Project Kural</strong> - Powered by OpenRouter AI • 
            Built with ❤️ for intelligent customer service</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()