"""
SafeServe AI - Fraud Detection Engine
Real-time fraud detection using machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
import joblib
import datetime
import logging
from typing import Dict, List, Tuple, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionEngine:
    """
    Advanced fraud detection engine using multiple ML models
    """
    
    def __init__(self, model_type: str = "isolation_forest"):
        """
        Initialize the fraud detection engine
        
        Args:
            model_type: Type of model to use ('isolation_forest', 'lof', 'ocsvm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'amount', 'hour', 'day_of_week', 'location_risk', 
            'device_score', 'velocity_score', 'merchant_risk'
        ]
        
    def _initialize_model(self):
        """Initialize the ML model based on type"""
        if self.model_type == "isolation_forest":
            self.model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
        elif self.model_type == "lof":
            self.model = LOF(contamination=0.1)
        elif self.model_type == "ocsvm":
            self.model = OCSVM(contamination=0.1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def _extract_features(self, transaction_data: Dict) -> np.ndarray:
        """
        Extract features from transaction data
        
        Args:
            transaction_data: Dictionary containing transaction information
            
        Returns:
            Feature vector as numpy array
        """
        # Extract basic features
        amount = float(transaction_data.get('amount', 0))
        timestamp = transaction_data.get('timestamp', datetime.datetime.now())
        
        if isinstance(timestamp, str):
            timestamp = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Time-based features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Location risk score (simplified - in production, use geolocation data)
        location = transaction_data.get('location', 'unknown')
        location_risk = self._calculate_location_risk(location)
        
        # Device score (simplified - in production, use device fingerprinting)
        device_id = transaction_data.get('device_id', 'unknown')
        device_score = self._calculate_device_score(device_id)
        
        # Velocity score (transactions per hour)
        velocity_score = transaction_data.get('velocity_score', 0)
        
        # Merchant risk score
        merchant = transaction_data.get('merchant', 'unknown')
        merchant_risk = self._calculate_merchant_risk(merchant)
        
        features = np.array([
            amount,
            hour,
            day_of_week,
            location_risk,
            device_score,
            velocity_score,
            merchant_risk
        ])
        
        return features.reshape(1, -1)
    
    def _calculate_location_risk(self, location: str) -> float:
        """Calculate location-based risk score"""
        # Simplified location risk - in production, use geolocation analysis
        high_risk_locations = ['foreign', 'unknown', 'high_crime_area']
        medium_risk_locations = ['new_location', 'atm']
        
        if location.lower() in high_risk_locations:
            return 0.8
        elif location.lower() in medium_risk_locations:
            return 0.5
        else:
            return 0.2
    
    def _calculate_device_score(self, device_id: str) -> float:
        """Calculate device-based risk score"""
        # Simplified device scoring - in production, use device fingerprinting
        if device_id == 'unknown' or device_id == 'new_device':
            return 0.7
        else:
            return 0.3
    
    def _calculate_merchant_risk(self, merchant: str) -> float:
        """Calculate merchant-based risk score"""
        # Simplified merchant risk - in production, use merchant analysis
        high_risk_merchants = ['online_gambling', 'crypto_exchange', 'unknown']
        medium_risk_merchants = ['online_shopping', 'gas_station']
        
        if merchant.lower() in high_risk_merchants:
            return 0.9
        elif merchant.lower() in medium_risk_merchants:
            return 0.4
        else:
            return 0.1
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic transaction data for training
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic transaction data
        """
        np.random.seed(42)
        
        # Generate normal transactions
        normal_samples = int(n_samples * 0.9)
        fraud_samples = n_samples - normal_samples
        
        # Normal transactions
        normal_data = {
            'amount': np.random.lognormal(mean=3, sigma=1, size=normal_samples),
            'hour': np.random.normal(loc=14, scale=4, size=normal_samples) % 24,
            'day_of_week': np.random.randint(0, 7, normal_samples),
            'location_risk': np.random.beta(2, 8, normal_samples),
            'device_score': np.random.beta(2, 8, normal_samples),
            'velocity_score': np.random.exponential(scale=2, size=normal_samples),
            'merchant_risk': np.random.beta(2, 8, normal_samples),
            'is_fraud': np.zeros(normal_samples)
        }
        
        # Fraudulent transactions
        fraud_data = {
            'amount': np.random.lognormal(mean=5, sigma=1.5, size=fraud_samples),
            'hour': np.random.choice([2, 3, 4, 23, 0, 1], fraud_samples),
            'day_of_week': np.random.randint(0, 7, fraud_samples),
            'location_risk': np.random.beta(8, 2, fraud_samples),
            'device_score': np.random.beta(8, 2, fraud_samples),
            'velocity_score': np.random.exponential(scale=8, size=fraud_samples),
            'merchant_risk': np.random.beta(8, 2, fraud_samples),
            'is_fraud': np.ones(fraud_samples)
        }
        
        # Combine data
        data = {}
        for key in normal_data.keys():
            data[key] = np.concatenate([normal_data[key], fraud_data[key]])
        
        return pd.DataFrame(data)
    
    def train(self, data: Optional[pd.DataFrame] = None):
        """
        Train the fraud detection model
        
        Args:
            data: Training data (if None, synthetic data is generated)
        """
        if data is None:
            logger.info("Generating synthetic training data...")
            data = self.generate_synthetic_data(1000)
        
        # Prepare features
        X = data[self.feature_columns]
        
        # Use only normal transactions for unsupervised learning
        if 'is_fraud' in data.columns:
            X_normal = X[data['is_fraud'] == 0]
        else:
            X_normal = X
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_normal)
        
        # Initialize and train model
        self._initialize_model()
        self.model.fit(X_scaled)
        
        self.is_trained = True
        logger.info(f"Model trained successfully with {len(X_normal)} samples")
    
    def predict(self, transaction_data: Dict) -> Dict[str, float]:
        """
        Predict fraud probability for a transaction
        
        Args:
            transaction_data: Transaction data dictionary
            
        Returns:
            Dictionary with risk_score and label
        """
        if not self.is_trained:
            logger.warning("Model not trained, training with synthetic data...")
            self.train()
        
        # Extract features
        features = self._extract_features(transaction_data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        if self.model_type == "isolation_forest":
            prediction = self.model.predict(features_scaled)[0]
            decision_score = self.model.decision_function(features_scaled)[0]
            # Convert to 0-1 probability
            risk_score = max(0, min(1, (0.5 - decision_score) * 2))
        else:  # PyOD models
            prediction = self.model.predict(features_scaled)[0]
            risk_score = self.model.predict_proba(features_scaled)[0][1]
        
        # Determine label
        label = "Suspicious" if prediction == -1 or risk_score > 0.5 else "Safe"
        
        return {
            "risk_score": float(risk_score),
            "label": label,
            "confidence": float(abs(0.5 - risk_score) * 2)
        }
    
    def save_model(self, filepath: str):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")

def create_sample_transaction() -> Dict:
    """Create a sample transaction for testing"""
    return {
        'amount': 9000,
        'timestamp': datetime.datetime.now(),
        'location': 'foreign',
        'device_id': 'unknown',
        'velocity_score': 5,
        'merchant': 'online_shopping'
    }

# Global fraud detector instance
_fraud_detector = None

def get_fraud_detector():
    """Get or create global fraud detector instance"""
    global _fraud_detector
    if _fraud_detector is None:
        _fraud_detector = FraudDetectionEngine(model_type="isolation_forest")
        _fraud_detector.train()
    return _fraud_detector

def predict_fraud(transaction_data: Dict) -> Dict:
    """
    Predict fraud for a transaction (API-compatible function)
    
    Args:
        transaction_data: Transaction data dictionary
        
    Returns:
        Dictionary with fraud prediction results
    """
    try:
        fraud_detector = get_fraud_detector()
        result = fraud_detector.predict(transaction_data)
        
        # Add timestamp and additional fields for API compatibility
        result['timestamp'] = datetime.datetime.now().isoformat()
        result['explanation'] = f"Transaction risk analysis: {result['label']} (Risk Score: {result['risk_score']:.2f})"
        
        return result
        
    except Exception as e:
        logger.error(f"Error in fraud prediction: {str(e)}")
        return {
            "risk_score": 0.5,
            "label": "Error",
            "confidence": 0.0,
            "explanation": f"Error in fraud detection: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat()
        }

# Example usage
if __name__ == "__main__":
    # Initialize fraud detection engine
    fraud_detector = FraudDetectionEngine(model_type="isolation_forest")
    
    # Train the model
    fraud_detector.train()
    
    # Test with sample transaction
    sample_transaction = create_sample_transaction()
    result = fraud_detector.predict(sample_transaction)
    
    print(f"Transaction Analysis:")
    print(f"Amount: ₹{sample_transaction['amount']}")
    print(f"Location: {sample_transaction['location']}")
    print(f"Risk Score: {result['risk_score']:.2f}")
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.2f}")