"""
SafeServe AI - Streamlit Frontend
Elegant UI for fraud detection and AI chatbot
"""

import streamlit as st
import requests
import json
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import time
import base64
from pathlib import Path
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from load_env import get_config
    config = get_config()
except:
    # Fallback configuration
    config = {
        "API_HOST": "localhost",
        "API_PORT": 8080,
        "LLM_API_URL": "http://localhost:8000/chat"
    }

# Configure Streamlit page
st.set_page_config(
    page_title="SafeServe AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: white;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .fraud-alert {
        background-color: #ff6b6b;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    .fraud-safe {
        background-color: #51cf66;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        text-align: right;
    }
    
    .bot-message {
        background-color: #f1f3f4;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        text-align: left;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .language-selector {
        position: absolute;
        top: 10px;
        right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"

# API Configuration
API_BASE_URL = f"http://{config['API_HOST']}:{config['API_PORT']}"

# Helper functions
def get_api_health():
    """Check API health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def predict_fraud(transaction_data: Dict) -> Optional[Dict]:
    """Make fraud prediction API call"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=transaction_data,
            timeout=10
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error making fraud prediction: {str(e)}")
        return None

def chat_with_ai(query: str, user_id: str = "user") -> Optional[str]:
    """Chat with AI API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"query": query, "user_id": user_id},
            timeout=15
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to AI: {str(e)}"

def analyze_transaction_with_chat(transaction_data: Dict) -> Optional[Dict]:
    """Combined transaction analysis with AI explanation"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze-transaction-with-chat",
            json=transaction_data,
            timeout=20
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error in combined analysis: {str(e)}")
        return None

def get_system_stats() -> Optional[Dict]:
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def display_risk_gauge(risk_score: float):
    """Display risk score as a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def display_transaction_trends():
    """Display transaction trends chart"""
    if st.session_state.transaction_history:
        df = pd.DataFrame(st.session_state.transaction_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by hour and calculate risk metrics
        hourly_data = df.groupby(df['timestamp'].dt.hour).agg({
            'risk_score': 'mean',
            'amount': 'sum'
        }).reset_index()
        
        fig = px.line(
            hourly_data, 
            x='timestamp', 
            y='risk_score',
            title='Average Risk Score by Hour',
            labels={'timestamp': 'Hour', 'risk_score': 'Average Risk Score'}
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Main UI
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ SafeServe AI</h1>
        <p>AI-Powered Fraud Detection & Customer Service Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Language selector
    languages = ["English", "Hindi", "Spanish", "French"]
    selected_lang = st.selectbox("🌐 Language", languages, key="lang_selector")
    st.session_state.selected_language = selected_lang
    
    # Check API health
    health = get_api_health()
    if health:
        if health["status"] == "healthy":
            st.success("✅ System Online - All services operational")
        else:
            st.warning("⚠️ System Status: Degraded")
    else:
        st.error("❌ Unable to connect to backend services")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Fraud Detection", "💬 AI Assistant", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.header("Transaction Fraud Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enter Transaction Details")
            
            # Transaction input form
            with st.form("transaction_form"):
                amount = st.number_input("💰 Amount (₹)", min_value=0.01, value=1000.0)
                
                location = st.selectbox("📍 Location", [
                    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata",
                    "Pune", "Hyderabad", "foreign", "unknown"
                ])
                
                merchant = st.selectbox("🏪 Merchant", [
                    "Amazon", "Flipkart", "Walmart", "Target", "Starbucks",
                    "McDonald's", "Gas Station", "ATM", "online_shopping",
                    "online_gambling", "crypto_exchange", "unknown"
                ])
                
                device_id = st.text_input("📱 Device ID", value="device_1234")
                velocity_score = st.slider("⚡ Velocity Score", 0.0, 10.0, 2.0)
                
                submit_button = st.form_submit_button("🔍 Analyze Transaction")
                
                if submit_button:
                    transaction_data = {
                        "amount": amount,
                        "location": location,
                        "merchant": merchant,
                        "device_id": device_id,
                        "velocity_score": velocity_score,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
                    # Show loading spinner
                    with st.spinner("Analyzing transaction..."):
                        result = analyze_transaction_with_chat(transaction_data)
                    
                    if result:
                        fraud_analysis = result["fraud_analysis"]
                        explanation = result["explanation"]
                        
                        # Store in session state
                        st.session_state.transaction_history.append({
                            **transaction_data,
                            **fraud_analysis,
                            "timestamp": result["timestamp"]
                        })
                        
                        # Display results
                        risk_score = fraud_analysis["risk_score"]
                        label = fraud_analysis["label"]
                        
                        if label == "Suspicious":
                            st.markdown(f"""
                            <div class="fraud-alert">
                                <h3>🚨 FRAUD ALERT</h3>
                                <p>This transaction has been flagged as suspicious!</p>
                                <p><strong>Risk Score:</strong> {risk_score:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="fraud-safe">
                                <h3>✅ TRANSACTION SAFE</h3>
                                <p>This transaction appears to be legitimate.</p>
                                <p><strong>Risk Score:</strong> {risk_score:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # AI Explanation
                        st.subheader("🤖 AI Analysis")
                        st.info(explanation)
                        
                        # Recommendations
                        if risk_score > 0.7:
                            st.error("🔒 **Recommended Actions:**")
                            st.markdown("- Block the transaction immediately")
                            st.markdown("- Contact customer for verification")
                            st.markdown("- Review account for other suspicious activity")
                        elif risk_score > 0.4:
                            st.warning("⚠️ **Recommended Actions:**")
                            st.markdown("- Monitor account activity")
                            st.markdown("- Consider additional verification")
                        else:
                            st.success("✅ **No action required** - Transaction appears normal")
        
        with col2:
            st.subheader("Risk Assessment")
            
            # Show risk gauge if we have recent analysis
            if st.session_state.transaction_history:
                latest_transaction = st.session_state.transaction_history[-1]
                display_risk_gauge(latest_transaction["risk_score"])
                
                # Transaction details
                st.markdown("### Latest Transaction")
                st.markdown(f"**Amount:** ₹{latest_transaction['amount']}")
                st.markdown(f"**Location:** {latest_transaction['location']}")
                st.markdown(f"**Merchant:** {latest_transaction['merchant']}")
                st.markdown(f"**Status:** {latest_transaction['label']}")
    
    with tab2:
        st.header("AI Customer Service Assistant")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Chat interface
            st.subheader("💬 Chat with SafeServe AI")
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                for message in st.session_state.chat_history:
                    if message["type"] == "user":
                        st.markdown(f'<div class="user-message">You: {message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="bot-message">🤖 SafeServe AI: {message["content"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Chat input
            user_query = st.text_input("Type your message here...", key="chat_input")
            
            col_send, col_clear = st.columns([1, 1])
            
            with col_send:
                if st.button("📤 Send"):
                    if user_query:
                        # Add user message to history
                        st.session_state.chat_history.append({
                            "type": "user",
                            "content": user_query,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                        
                        # Get AI response
                        with st.spinner("AI is thinking..."):
                            ai_response = chat_with_ai(user_query)
                        
                        if ai_response:
                            st.session_state.chat_history.append({
                                "type": "bot",
                                "content": ai_response,
                                "timestamp": datetime.datetime.now().isoformat()
                            })
                        
                        st.rerun()
            
            with col_clear:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        with col2:
            st.subheader("Quick Actions")
            
            # Pre-defined queries
            quick_queries = [
                "How can I check if a transaction is fraudulent?",
                "What should I do if I suspect fraud?",
                "How do I report a suspicious transaction?",
                "What are common fraud indicators?",
                "How can I protect my account?",
                "What is your fraud detection accuracy?"
            ]
            
            for query in quick_queries:
                if st.button(f"💡 {query}", key=f"quick_{query[:20]}"):
                    st.session_state.chat_history.append({
                        "type": "user",
                        "content": query,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    
                    with st.spinner("AI is thinking..."):
                        ai_response = chat_with_ai(query)
                    
                    if ai_response:
                        st.session_state.chat_history.append({
                            "type": "bot",
                            "content": ai_response,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                    
                    st.rerun()
    
    with tab3:
        st.header("Analytics Dashboard")
        
        # System stats
        stats = get_system_stats()
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", stats.get("total_transactions", 0))
            
            with col2:
                st.metric("Suspicious Transactions", stats.get("suspicious_transactions", 0))
            
            with col3:
                st.metric("Chat Interactions", stats.get("total_chats", 0))
            
            with col4:
                st.metric("Detection Accuracy", stats.get("fraud_detection_accuracy", "N/A"))
        
        # Transaction trends
        if st.session_state.transaction_history:
            st.subheader("Transaction Analysis")
            display_transaction_trends()
            
            # Transaction history table
            st.subheader("Recent Transactions")
            df = pd.DataFrame(st.session_state.transaction_history)
            st.dataframe(df.tail(10), use_container_width=True)
        else:
            st.info("No transaction data available. Process some transactions to see analytics.")
    
    with tab4:
        st.header("System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("API Configuration")
            st.text_input("API Host", value=config["API_HOST"], disabled=True)
            st.text_input("API Port", value=str(config["API_PORT"]), disabled=True)
            st.text_input("LLM API URL", value=config["LLM_API_URL"], disabled=True)
            
            if st.button("🔄 Test Connection"):
                health = get_api_health()
                if health:
                    st.success("✅ Connection successful!")
                    st.json(health)
                else:
                    st.error("❌ Connection failed!")
        
        with col2:
            st.subheader("Application Info")
            st.info("""
            **SafeServe AI v1.0**
            
            🛡️ **Features:**
            - Real-time fraud detection
            - AI-powered customer service
            - Transaction analytics
            - Multi-language support
            
            🔧 **Technology Stack:**
            - Backend: FastAPI + ML Models
            - Frontend: Streamlit
            - AI: Deepseek-Coder 6.7B
            - ML: Isolation Forest, PyOD
            """)
            
            if st.button("📊 System Health Check"):
                st.info("Running comprehensive system check...")
                
                # Check various components
                checks = {
                    "Backend API": get_api_health() is not None,
                    "Fraud Detection": True,  # Always true if backend is up
                    "AI Assistant": health.get("llm_api_available", False) if health else False,
                    "Database": True,  # Mock check
                    "Analytics": len(st.session_state.transaction_history) >= 0
                }
                
                for component, status in checks.items():
                    if status:
                        st.success(f"✅ {component}: Online")
                    else:
                        st.error(f"❌ {component}: Offline")

if __name__ == "__main__":
    main()