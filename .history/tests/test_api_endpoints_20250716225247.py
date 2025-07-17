"""
Test suite for API endpoints
Tests all API endpoints return 200 and expected response format
"""

import pytest
import requests
import json
import time
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.load_env import get_config

class TestAPIEndpoints:
    """Test all API endpoints for proper responses"""
    
    @classmethod
    def setup_class(cls):
        """Set up test configuration"""
        cls.config = get_config()
        cls.base_url = f"http://{cls.config.get('API_HOST', 'localhost')}:{cls.config.get('API_PORT', 8080)}"
        cls.timeout = 30
    
    def test_health_endpoint(self):
        """Test /health endpoint returns 200 and proper structure"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            assert response.status_code == 200, f"Health endpoint failed: {response.status_code}"
            
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
            assert "components" in data
            assert "version" in data
            
            # Check components structure
            components = data["components"]
            assert "llm_api" in components
            assert "fraud_detection" in components
            assert "llm_api_url" in components
            
        except requests.RequestException as e:
            pytest.fail(f"Health endpoint request failed: {str(e)}")
    
    def test_root_endpoint(self):
        """Test root endpoint returns service information"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=self.timeout)
            assert response.status_code == 200, f"Root endpoint failed: {response.status_code}"
            
            data = response.json()
            assert "service" in data
            assert "version" in data
            assert "description" in data
            assert "endpoints" in data
            
            # Check endpoints structure
            endpoints = data["endpoints"]
            expected_endpoints = ["assistant", "fraud_detection", "chat", "health", "stats"]
            for endpoint in expected_endpoints:
                assert endpoint in endpoints
            
        except requests.RequestException as e:
            pytest.fail(f"Root endpoint request failed: {str(e)}")
    
    def test_stats_endpoint(self):
        """Test /stats endpoint returns statistics"""
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=self.timeout)
            assert response.status_code == 200, f"Stats endpoint failed: {response.status_code}"
            
            data = response.json()
            assert "total_transactions" in data
            assert "suspicious_transactions" in data
            assert "total_chats" in data
            assert "active_conversations" in data
            assert "llm_status" in data
            assert "timestamp" in data
            
            # Check data types
            assert isinstance(data["total_transactions"], int)
            assert isinstance(data["suspicious_transactions"], int)
            assert isinstance(data["total_chats"], int)
            assert isinstance(data["active_conversations"], int)
            
        except requests.RequestException as e:
            pytest.fail(f"Stats endpoint request failed: {str(e)}")
    
    def test_chat_endpoint(self):
        """Test /chat endpoint with valid request"""
        try:
            payload = {
                "query": "Hello, I need help with my account",
                "user_id": "test_user",
                "language": "en"
            }
            
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=self.timeout
            )
            
            assert response.status_code == 200, f"Chat endpoint failed: {response.status_code}"
            
            data = response.json()
            assert "response" in data
            assert "timestamp" in data
            assert "user_id" in data
            assert "processing_time" in data
            
            # Check data types
            assert isinstance(data["response"], str)
            assert isinstance(data["user_id"], str)
            assert isinstance(data["processing_time"], (int, float))
            assert data["user_id"] == "test_user"
            
        except requests.RequestException as e:
            pytest.fail(f"Chat endpoint request failed: {str(e)}")
    
    def test_assistant_endpoint(self):
        """Test /assistant endpoint with valid request"""
        try:
            payload = {
                "text": "I want to check my account balance",
                "lang": "en",
                "mode": "text"
            }
            
            response = requests.post(
                f"{self.base_url}/assistant",
                json=payload,
                timeout=self.timeout
            )
            
            assert response.status_code == 200, f"Assistant endpoint failed: {response.status_code}"
            
            data = response.json()
            required_fields = [
                "response", "translated_response", "language_detected",
                "fraud_likelihood", "fraud_label", "processing_time",
                "timestamp", "user_id", "capabilities"
            ]
            
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            
            # Check data types
            assert isinstance(data["response"], str)
            assert isinstance(data["fraud_likelihood"], (int, float))
            assert isinstance(data["fraud_label"], str)
            assert isinstance(data["processing_time"], (int, float))
            assert isinstance(data["capabilities"], dict)
            
            # Check fraud likelihood is in valid range
            assert 0 <= data["fraud_likelihood"] <= 1
            
        except requests.RequestException as e:
            pytest.fail(f"Assistant endpoint request failed: {str(e)}")
    
    def test_predict_endpoint(self):
        """Test /predict endpoint with valid transaction"""
        try:
            payload = {
                "amount": 100.50,
                "location": "New York, NY",
                "merchant": "Test Store",
                "device_id": "test_device_123",
                "velocity_score": 0.3
            }
            
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            
            assert response.status_code == 200, f"Predict endpoint failed: {response.status_code}"
            
            data = response.json()
            assert "risk_score" in data
            assert "label" in data
            assert "explanation" in data
            assert "timestamp" in data
            
            # Check data types and ranges
            assert isinstance(data["risk_score"], (int, float))
            assert isinstance(data["label"], str)
            assert isinstance(data["explanation"], str)
            assert 0 <= data["risk_score"] <= 1
            assert data["label"] in ["Safe", "Suspicious", "Moderate"]
            
        except requests.RequestException as e:
            pytest.fail(f"Predict endpoint request failed: {str(e)}")
    
    def test_combined_analysis_endpoint(self):
        """Test /analyze-transaction-with-chat endpoint"""
        try:
            payload = {
                "amount": 500.00,
                "location": "Los Angeles, CA",
                "merchant": "Online Store",
                "device_id": "device_456",
                "velocity_score": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/analyze-transaction-with-chat",
                json=payload,
                timeout=self.timeout
            )
            
            assert response.status_code == 200, f"Combined analysis endpoint failed: {response.status_code}"
            
            data = response.json()
            assert "fraud_analysis" in data
            assert "explanation" in data
            assert "timestamp" in data
            
            # Check fraud analysis structure
            fraud_analysis = data["fraud_analysis"]
            assert "risk_score" in fraud_analysis
            assert "label" in fraud_analysis
            assert "explanation" in fraud_analysis
            assert "timestamp" in fraud_analysis
            
        except requests.RequestException as e:
            pytest.fail(f"Combined analysis endpoint request failed: {str(e)}")
    
    def test_transactions_endpoint(self):
        """Test /transactions endpoint"""
        try:
            response = requests.get(f"{self.base_url}/transactions", timeout=self.timeout)
            assert response.status_code == 200, f"Transactions endpoint failed: {response.status_code}"
            
            data = response.json()
            assert "transactions" in data
            assert "count" in data
            assert "suspicious_count" in data
            
            # Check data types
            assert isinstance(data["transactions"], list)
            assert isinstance(data["count"], int)
            assert isinstance(data["suspicious_count"], int)
            
        except requests.RequestException as e:
            pytest.fail(f"Transactions endpoint request failed: {str(e)}")
    
    def test_chats_endpoint(self):
        """Test /chats endpoint"""
        try:
            response = requests.get(f"{self.base_url}/chats", timeout=self.timeout)
            assert response.status_code == 200, f"Chats endpoint failed: {response.status_code}"
            
            data = response.json()
            assert "chats" in data
            assert "total_chats" in data
            
            # Check data types
            assert isinstance(data["chats"], list)
            assert isinstance(data["total_chats"], int)
            
        except requests.RequestException as e:
            pytest.fail(f"Chats endpoint request failed: {str(e)}")
    
    def test_invalid_endpoint_404(self):
        """Test that invalid endpoints return 404"""
        try:
            response = requests.get(f"{self.base_url}/invalid-endpoint", timeout=self.timeout)
            assert response.status_code == 404, f"Expected 404 for invalid endpoint, got {response.status_code}"
            
        except requests.RequestException as e:
            pytest.fail(f"Invalid endpoint test failed: {str(e)}")
    
    def test_chat_endpoint_validation(self):
        """Test chat endpoint with invalid payload"""
        try:
            # Missing required field
            payload = {
                "user_id": "test_user",
                "language": "en"
                # Missing 'query' field
            }
            
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=self.timeout
            )
            
            assert response.status_code == 422, f"Expected 422 for invalid payload, got {response.status_code}"
            
        except requests.RequestException as e:
            pytest.fail(f"Chat validation test failed: {str(e)}")
    
    def test_predict_endpoint_validation(self):
        """Test predict endpoint with invalid payload"""
        try:
            # Missing required fields
            payload = {
                "amount": 100.50,
                "location": "New York, NY"
                # Missing required fields
            }
            
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            
            assert response.status_code == 422, f"Expected 422 for invalid payload, got {response.status_code}"
            
        except requests.RequestException as e:
            pytest.fail(f"Predict validation test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])