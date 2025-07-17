"""
SafeServe AI - Enhanced Streamlit UI
Modern interface for testing fraud detection, multilingual support, and voice interaction
"""

import streamlit as st
import requests
import json
import time
import base64
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from io import BytesIO
import pandas as pd

# Configure page
st.set_page_config(
    page_title="SafeServe AI - Customer Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .fraud-alert {
        background: #ffebee;
        border: 1px solid #f44336;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .safe-indicator {
        background: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-indicator {
        background: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8080"
if 'user_id' not in st.session_state:
    st.session_state.user_id = "streamlit_user"
if 'system_capabilities' not in st.session_state:
    st.session_state.system_capabilities = {}

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ SafeServe AI - Enhanced Customer Assistant</h1>
    <p>Advanced fraud detection, multilingual support, and voice interaction</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# API Configuration
st.sidebar.subheader("API Settings")
api_url = st.sidebar.text_input(
    "API URL",
    value=st.session_state.api_url,
    help="URL of the SafeServe AI API"
)
st.session_state.api_url = api_url

user_id = st.sidebar.text_input(
    "User ID",
    value=st.session_state.user_id,
    help="Unique identifier for this session"
)
st.session_state.user_id = user_id

# System health check
def check_system_health():
    """Check API health and get capabilities"""
    try:
        # Health check
        health_response = requests.get(f"{st.session_state.api_url}/health", timeout=5)
        health_data = health_response.json() if health_response.status_code == 200 else {}
        
        # Get stats
        stats_response = requests.get(f"{st.session_state.api_url}/stats", timeout=5)
        stats_data = stats_response.json() if stats_response.status_code == 200 else {}
        
        return health_data, stats_data
        
    except Exception as e:
        st.sidebar.error(f"API connection failed: {str(e)}")
        return {}, {}

# Check system status
if st.sidebar.button("🔍 Check System Status"):
    with st.sidebar:
        with st.spinner("Checking system status..."):
            health, capabilities = check_system_health()
            st.session_state.system_capabilities = capabilities
            
            if health:
                st.success("✅ System Online")
                st.json(health)
            else:
                st.error("❌ System Offline")

# Display capabilities
if st.session_state.system_capabilities:
    st.sidebar.subheader("🎯 System Capabilities")
    caps = st.session_state.system_capabilities
    
    # Languages
    if 'languages' in caps:
        st.sidebar.write("**Languages:**")
        for lang_code, lang_name in caps['languages'].get('supported', {}).items():
            st.sidebar.write(f"• {lang_name} ({lang_code})")
    
    # Voice capabilities
    if 'voice' in caps:
        voice_caps = caps['voice']
        st.sidebar.write("**Voice:**")
        st.sidebar.write(f"• Recognition: {'✅' if voice_caps.get('recognition') else '❌'}")
        st.sidebar.write(f"• Synthesis: {'✅' if voice_caps.get('synthesis') else '❌'}")
    
    # Fraud detection
    if 'fraud_detection' in caps:
        st.sidebar.write("**Fraud Detection:**")
        st.sidebar.write(f"• Enabled: {'✅' if caps['fraud_detection'].get('enabled') else '❌'}")

# Main interface tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chat Assistant", 
    "🎤 Voice Interaction", 
    "📊 Analytics", 
    "🔧 Testing"
])

def send_chat_message(text: str, language: str = "auto"):
    """Send chat message to SafeServe AI API"""
    try:
        payload = {
            "query": text,
            "user_id": st.session_state.user_id,
            "language": language
        }
        
        response = requests.post(
            f"{st.session_state.api_url}/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def send_fraud_prediction(amount: float, location: str, merchant: str, device_id: str, velocity_score: float):
    """Send fraud prediction request"""
    try:
        payload = {
            "amount": amount,
            "location": location,
            "merchant": merchant,
            "device_id": device_id,
            "velocity_score": velocity_score
        }
        
        response = requests.post(
            f"{st.session_state.api_url}/predict",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def send_combined_analysis(amount: float, location: str, merchant: str, device_id: str, velocity_score: float):
    """Send combined analysis request"""
    try:
        payload = {
            "amount": amount,
            "location": location,
            "merchant": merchant,
            "device_id": device_id,
            "velocity_score": velocity_score
        }
        
        response = requests.post(
            f"{st.session_state.api_url}/analyze-transaction-with-chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

# System health check
def check_system_health():
    """Check API health and get capabilities"""
    try:
        # Health check
        health_response = requests.get(f"{st.session_state.api_url}/health", timeout=5)
        health_data = health_response.json() if health_response.status_code == 200 else {}
        
        # Get stats
        stats_response = requests.get(f"{st.session_state.api_url}/stats", timeout=5)
        stats_data = stats_response.json() if stats_response.status_code == 200 else {}
        
        return health_data, stats_data
        
    except Exception as e:
        st.sidebar.error(f"API connection failed: {str(e)}")
        return {}, {}

def display_fraud_indicator(risk_score: float, fraud_label: str):
    """Display fraud detection results"""
    if fraud_label == "Suspicious":
        st.markdown(f"""
        <div class="fraud-alert">
            <h4>🚨 FRAUD RISK DETECTED</h4>
            <p><strong>Risk Score:</strong> {risk_score:.2%}</p>
            <p><strong>Classification:</strong> {fraud_label}</p>
            <p><em>This interaction has been flagged for manual review.</em></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="safe-indicator">
            <h4>✅ SAFE INTERACTION</h4>
            <p><strong>Risk Score:</strong> {risk_score:.2%}</p>
            <p><strong>Classification:</strong> {fraud_label}</p>
            <p><em>No fraud indicators detected.</em></p>
        </div>
        """, unsafe_allow_html=True)

# Tab 1: Chat Assistant
with tab1:
    st.header("💬 Interactive Chat Assistant")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Language selection
        language_options = {
            "auto": "Auto-detect",
            "en": "English",
            "hi": "Hindi",
            "ta": "Tamil",
            "te": "Telugu",
            "bn": "Bengali",
            "mr": "Marathi",
            "gu": "Gujarati"
        }
        
        selected_language = st.selectbox(
            "Language",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=0
        )
        
        # Message input
        user_input = st.text_area(
            "Type your message:",
            height=100,
            placeholder="Ask about account issues, report fraud, or get help..."
        )
        
        # Send button
        if st.button("Send Message", type="primary"):
            if user_input.strip():
                with st.spinner("Processing your message..."):
                    response = send_chat_message(user_input, selected_language)
                    
                    if response:
                        # Add to chat history with correct response format
                        st.session_state.chat_history.append({
                            'user': user_input,
                            'assistant': response['response'],
                            'fraud_likelihood': 0.0,  # Chat endpoint doesn't return fraud info
                            'fraud_label': "safe",
                            'language_detected': selected_language,
                            'processing_time': response['processing_time'],
                            'risk_factors': [],
                            'timestamp': response['timestamp']
                        })
                        
                        st.rerun()
    
    with col2:
        # Quick actions
        st.subheader("Quick Actions")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("📋 Get Summary"):
            try:
                summary_response = requests.get(
                    f"{st.session_state.api_url}/user/{st.session_state.user_id}/summary"
                )
                if summary_response.status_code == 200:
                    st.json(summary_response.json())
            except Exception as e:
                st.error(f"Failed to get summary: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You ({chat['language_detected']}):</strong> {chat['user']}
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant response
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>SafeServe AI:</strong> {chat['assistant']}
                <br><small>Processing time: {chat['processing_time']:.2f}s</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Fraud detection results
            display_fraud_indicator(
                chat['fraud_likelihood'],
                chat['fraud_label'],
                chat['risk_factors']
            )
            
            st.markdown("---")

# Tab 2: Voice Interaction
with tab2:
    st.header("🎤 Voice Interaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎙️ Voice Input")
        
        # Audio recording placeholder
        st.info("Voice recording functionality requires additional setup. Please refer to the documentation for Vosk and audio recording setup.")
        
        # File upload for audio
        uploaded_audio = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'ogg'],
            help="Upload an audio file to test voice recognition"
        )
        
        if uploaded_audio:
            st.audio(uploaded_audio, format='audio/wav')
            
            if st.button("Process Voice Input"):
                # Convert audio to base64
                audio_bytes = uploaded_audio.read()
                audio_base64 = base64.b64encode(audio_bytes).decode()
                
                with st.spinner("Processing voice input..."):
                    response = send_message(
                        text="[Voice Input]",
                        mode="voice",
                        audio=audio_base64
                    )
                    
                    if response:
                        st.success("Voice processed successfully!")
                        st.write(f"**Detected Text:** {response['response']}")
                        st.write(f"**Language:** {response['language_detected']}")
    
    with col2:
        st.subheader("🗣️ Voice Output")
        
        # Test text for voice synthesis
        tts_text = st.text_area(
            "Enter text for voice synthesis:",
            value="Hello, this is SafeServe AI. How can I help you today?",
            height=100
        )
        
        tts_language = st.selectbox(
            "Voice Language",
            options=["en", "hi", "ta"],
            format_func=lambda x: {"en": "English", "hi": "Hindi", "ta": "Tamil"}[x]
        )
        
        if st.button("Generate Voice"):
            with st.spinner("Generating voice..."):
                response = send_message(
                    text=tts_text,
                    language=tts_language,
                    mode="voice"
                )
                
                if response and response.get('audio_base64'):
                    # Decode and play audio
                    audio_data = base64.b64decode(response['audio_base64'])
                    st.audio(audio_data, format='audio/wav')
                    st.success("Voice generated successfully!")
                else:
                    st.error("Voice generation failed or not available")

# Tab 3: Analytics
with tab3:
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.chat_history:
        # Conversation metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", len(st.session_state.chat_history))
        
        with col2:
            avg_processing_time = sum(chat['processing_time'] for chat in st.session_state.chat_history) / len(st.session_state.chat_history)
            st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
        
        with col3:
            high_risk_count = sum(1 for chat in st.session_state.chat_history if chat['fraud_label'] == 'high_risk')
            st.metric("High Risk Interactions", high_risk_count)
        
        with col4:
            languages = set(chat['language_detected'] for chat in st.session_state.chat_history)
            st.metric("Languages Used", len(languages))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fraud Risk Distribution")
            df = pd.DataFrame(st.session_state.chat_history)
            fraud_counts = df['fraud_label'].value_counts()
            st.bar_chart(fraud_counts)
        
        with col2:
            st.subheader("Language Distribution")
            lang_counts = df['language_detected'].value_counts()
            st.bar_chart(lang_counts)
        
        # Risk timeline
        st.subheader("Risk Score Timeline")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.line_chart(df.set_index('timestamp')['fraud_likelihood'])
        
        # Detailed table
        st.subheader("Detailed Interaction Log")
        display_df = df[['timestamp', 'language_detected', 'fraud_label', 'fraud_likelihood', 'processing_time']]
        st.dataframe(display_df, use_container_width=True)
    
    else:
        st.info("No conversation data available. Start chatting to see analytics!")

# Tab 4: Testing
with tab4:
    st.header("🔧 System Testing")
    
    # Test scenarios
    st.subheader("Fraud Detection Test Scenarios")
    
    test_scenarios = {
        "Safe Inquiry": "Hello, I would like to check my account balance please.",
        "Urgent Request": "I need to transfer money urgently right now before my account gets frozen!",
        "Suspicious Refund": "I want immediate refund for all transactions from last week. This is emergency!",
        "Phishing Attempt": "Click here to verify your account details immediately or you'll lose access forever!",
        "Multiple Languages": "मुझे अपने खाते की जानकारी चाहिए। मैं पैसे निकालना चाहता हूं।",
        "Tamil Query": "எனக்கு என் கணக்கு பற்றி தெரிந்து கொள்ள வேண்டும்"
    }
    
    selected_scenario = st.selectbox(
        "Select test scenario:",
        options=list(test_scenarios.keys())
    )
    
    st.text_area(
        "Test message:",
        value=test_scenarios[selected_scenario],
        height=80,
        key="test_message"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Run Test", type="primary"):
            with st.spinner("Running test..."):
                response = send_message(st.session_state.test_message)
                
                if response:
                    st.success("Test completed!")
                    
                    # Display results
                    st.subheader("Test Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Response:**")
                        st.write(response['translated_response'])
                    
                    with col2:
                        st.write("**Fraud Analysis:**")
                        st.write(f"Risk Score: {response['fraud_likelihood']:.2%}")
                        st.write(f"Classification: {response['fraud_label']}")
                        st.write(f"Language: {response['language_detected']}")
                        
                        if response['capabilities'].get('fraud_factors'):
                            st.write("**Risk Factors:**")
                            for factor in response['capabilities']['fraud_factors']:
                                st.write(f"• {factor}")
    
    with col2:
        if st.button("Batch Test All Scenarios"):
            results = []
            progress_bar = st.progress(0)
            
            for i, (scenario, message) in enumerate(test_scenarios.items()):
                with st.spinner(f"Testing: {scenario}"):
                    response = send_message(message)
                    
                    if response:
                        results.append({
                            'scenario': scenario,
                            'fraud_likelihood': response['fraud_likelihood'],
                            'fraud_label': response['fraud_label'],
                            'language_detected': response['language_detected'],
                            'processing_time': response['processing_time']
                        })
                    
                    progress_bar.progress((i + 1) / len(test_scenarios))
            
            # Display batch results
            if results:
                st.subheader("Batch Test Results")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                st.write("**Summary:**")
                st.write(f"Average Risk Score: {results_df['fraud_likelihood'].mean():.2%}")
                st.write(f"High Risk Scenarios: {(results_df['fraud_label'] == 'high_risk').sum()}")
                st.write(f"Average Processing Time: {results_df['processing_time'].mean():.2f}s")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>SafeServe AI v2.0 - Enhanced Customer Assistant with Fraud Detection</p>
    <p>Built with ❤️ for secure customer interactions</p>
</div>
""", unsafe_allow_html=True)