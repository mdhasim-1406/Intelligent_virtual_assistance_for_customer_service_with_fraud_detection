"""
SafeServe AI - Conversational Fraud Detection
NLP-based fraud detection from text conversations using LLM analysis
"""

import re
import json
import datetime
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FraudAnalysis:
    """Fraud analysis result structure"""
    fraud_likelihood: float
    label: str
    explanation: str
    confidence: float
    risk_factors: List[str]
    timestamp: str

class ConversationalFraudDetector:
    """
    Intelligent conversational fraud detection using LLM analysis
    """
    
    def __init__(self, llm_api_url: str):
        self.llm_api_url = llm_api_url
        self.conversation_history = []
        self.user_risk_profile = {}
        self.fraud_patterns = self._load_fraud_patterns()
        
    def _load_fraud_patterns(self) -> Dict[str, Any]:
        """Load predefined fraud patterns and keywords"""
        return {
            "urgency_keywords": [
                "urgent", "immediately", "right now", "emergency", "quickly",
                "before it's too late", "time sensitive", "act fast"
            ],
            "pressure_tactics": [
                "limited time", "offer expires", "last chance", "don't miss out",
                "exclusive deal", "special offer", "act now"
            ],
            "financial_keywords": [
                "refund", "money back", "compensation", "claim", "lawsuit",
                "settlement", "prize", "winner", "inheritance", "lottery"
            ],
            "suspicious_requests": [
                "verify account", "update information", "confirm details",
                "click here", "download", "install", "send money"
            ],
            "emotional_manipulation": [
                "trust me", "believe me", "guaranteed", "risk-free",
                "you've been selected", "congratulations", "lucky"
            ],
            "repetitive_patterns": [
                "multiple refund attempts", "repeated complaints", "same issue",
                "asked before", "told you already"
            ]
        }
    
    def _create_fraud_detection_prompt(self, message: str, context: str = "") -> str:
        """Create a specialized prompt for fraud detection"""
        prompt = f"""You are an expert fraud detection analyst. Analyze the following customer message for potential fraud, scam, or deceptive behavior patterns.

Customer Message: "{message}"

{f"Previous Context: {context}" if context else ""}

Analyze for:
1. Urgency tactics and pressure
2. Financial manipulation attempts
3. Emotional manipulation
4. Suspicious requests for information
5. Inconsistencies in the story
6. Repetitive or rehearsed language
7. Threats or intimidation

Provide analysis in this exact JSON format:
{{
    "fraud_likelihood": 0.XX,
    "label": "safe|suspicious|high_risk",
    "explanation": "Brief explanation of the assessment",
    "confidence": 0.XX,
    "risk_factors": ["factor1", "factor2"]
}}

Response:"""
        return prompt
    
    def _analyze_with_llm(self, message: str, context: str = "") -> Dict[str, Any]:
        """Use LLM to analyze message for fraud indicators"""
        try:
            prompt = self._create_fraud_detection_prompt(message, context)
            
            response = requests.post(
                self.llm_api_url,
                json={
                    "query": prompt,
                    "temperature": 0.3,  # Lower temperature for more consistent analysis
                    "max_length": 300
                },
                timeout=15
            )
            
            if response.status_code == 200:
                llm_response = response.json()["response"]
                
                # Extract JSON from LLM response
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse LLM JSON response")
                        return self._fallback_analysis(message)
                else:
                    return self._fallback_analysis(message)
            else:
                logger.error(f"LLM API error: {response.status_code}")
                return self._fallback_analysis(message)
                
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return self._fallback_analysis(message)
    
    def _fallback_analysis(self, message: str) -> Dict[str, Any]:
        """Fallback rule-based analysis when LLM is unavailable"""
        message_lower = message.lower()
        risk_score = 0.0
        risk_factors = []
        
        # Check for urgency keywords
        urgency_count = sum(1 for keyword in self.fraud_patterns["urgency_keywords"] 
                           if keyword in message_lower)
        if urgency_count > 0:
            risk_score += 0.3
            risk_factors.append("urgency_tactics")
        
        # Check for pressure tactics
        pressure_count = sum(1 for keyword in self.fraud_patterns["pressure_tactics"] 
                            if keyword in message_lower)
        if pressure_count > 0:
            risk_score += 0.2
            risk_factors.append("pressure_tactics")
        
        # Check for financial keywords
        financial_count = sum(1 for keyword in self.fraud_patterns["financial_keywords"] 
                             if keyword in message_lower)
        if financial_count > 1:
            risk_score += 0.4
            risk_factors.append("financial_manipulation")
        
        # Check for suspicious requests
        suspicious_count = sum(1 for keyword in self.fraud_patterns["suspicious_requests"] 
                              if keyword in message_lower)
        if suspicious_count > 0:
            risk_score += 0.3
            risk_factors.append("suspicious_requests")
        
        # Check for emotional manipulation
        emotional_count = sum(1 for keyword in self.fraud_patterns["emotional_manipulation"] 
                             if keyword in message_lower)
        if emotional_count > 0:
            risk_score += 0.2
            risk_factors.append("emotional_manipulation")
        
        # Cap the risk score
        risk_score = min(risk_score, 1.0)
        
        # Determine label
        if risk_score > 0.7:
            label = "high_risk"
        elif risk_score > 0.4:
            label = "suspicious"
        else:
            label = "safe"
        
        return {
            "fraud_likelihood": risk_score,
            "label": label,
            "explanation": f"Rule-based analysis detected {len(risk_factors)} risk factors",
            "confidence": 0.6,
            "risk_factors": risk_factors
        }
    
    def _check_conversation_patterns(self, message: str) -> float:
        """Check for suspicious patterns in conversation history"""
        risk_adjustment = 0.0
        
        # Check for repetitive requests
        similar_messages = [msg for msg in self.conversation_history 
                           if self._calculate_similarity(msg, message) > 0.7]
        if len(similar_messages) > 2:
            risk_adjustment += 0.3
        
        # Check for escalating demands
        if len(self.conversation_history) > 3:
            recent_messages = self.conversation_history[-3:]
            if self._detect_escalation(recent_messages):
                risk_adjustment += 0.2
        
        return min(risk_adjustment, 0.5)
    
    def _calculate_similarity(self, msg1: str, msg2: str) -> float:
        """Simple similarity calculation between messages"""
        words1 = set(msg1.lower().split())
        words2 = set(msg2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _detect_escalation(self, messages: List[str]) -> bool:
        """Detect if conversation is escalating in aggression or urgency"""
        escalation_keywords = [
            "demand", "insist", "require", "must", "have to", "need now",
            "unacceptable", "outrageous", "ridiculous", "angry", "furious"
        ]
        
        escalation_scores = []
        for msg in messages:
            score = sum(1 for keyword in escalation_keywords if keyword in msg.lower())
            escalation_scores.append(score)
        
        # Check if escalation is increasing
        return len(escalation_scores) > 1 and escalation_scores[-1] > escalation_scores[0]
    
    def analyze_message(self, message: str, user_id: str = "anonymous") -> FraudAnalysis:
        """
        Analyze a single message for fraud indicators
        
        Args:
            message: User message to analyze
            user_id: User identifier for tracking
            
        Returns:
            FraudAnalysis object with detailed results
        """
        # Update conversation history
        self.conversation_history.append(message)
        
        # Keep only last 10 messages for context
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Create context from recent conversation
        context = " | ".join(self.conversation_history[-3:]) if len(self.conversation_history) > 1 else ""
        
        # Get LLM analysis
        llm_result = self._analyze_with_llm(message, context)
        
        # Check conversation patterns
        pattern_risk = self._check_conversation_patterns(message)
        
        # Adjust risk based on user profile
        user_risk_multiplier = self.user_risk_profile.get(user_id, {}).get("risk_multiplier", 1.0)
        
        # Calculate final risk score
        base_risk = llm_result["fraud_likelihood"]
        adjusted_risk = min((base_risk + pattern_risk) * user_risk_multiplier, 1.0)
        
        # Update user risk profile
        self._update_user_profile(user_id, adjusted_risk)
        
        # Create final analysis
        analysis = FraudAnalysis(
            fraud_likelihood=adjusted_risk,
            label=self._get_risk_label(adjusted_risk),
            explanation=llm_result["explanation"],
            confidence=llm_result["confidence"],
            risk_factors=llm_result["risk_factors"],
            timestamp=datetime.datetime.now().isoformat()
        )
        
        logger.info(f"Fraud analysis for user {user_id}: {analysis.label} ({analysis.fraud_likelihood:.2f})")
        return analysis
    
    def _get_risk_label(self, risk_score: float) -> str:
        """Convert risk score to label"""
        if risk_score > 0.7:
            return "high_risk"
        elif risk_score > 0.4:
            return "suspicious"
        else:
            return "safe"
    
    def _update_user_profile(self, user_id: str, risk_score: float):
        """Update user risk profile based on analysis"""
        if user_id not in self.user_risk_profile:
            self.user_risk_profile[user_id] = {
                "risk_history": [],
                "risk_multiplier": 1.0,
                "total_interactions": 0
            }
        
        profile = self.user_risk_profile[user_id]
        profile["risk_history"].append(risk_score)
        profile["total_interactions"] += 1
        
        # Keep only last 20 interactions
        if len(profile["risk_history"]) > 20:
            profile["risk_history"] = profile["risk_history"][-20:]
        
        # Calculate risk multiplier based on history
        if len(profile["risk_history"]) > 5:
            avg_risk = sum(profile["risk_history"]) / len(profile["risk_history"])
            high_risk_count = sum(1 for r in profile["risk_history"] if r > 0.7)
            
            if high_risk_count > 3:
                profile["risk_multiplier"] = 1.5  # Increase scrutiny
            elif avg_risk > 0.5:
                profile["risk_multiplier"] = 1.2
            else:
                profile["risk_multiplier"] = 1.0
    
    def get_user_risk_summary(self, user_id: str) -> Dict[str, Any]:
        """Get risk summary for a user"""
        if user_id not in self.user_risk_profile:
            return {"status": "new_user", "risk_level": "unknown"}
        
        profile = self.user_risk_profile[user_id]
        avg_risk = sum(profile["risk_history"]) / len(profile["risk_history"])
        
        return {
            "user_id": user_id,
            "average_risk": avg_risk,
            "total_interactions": profile["total_interactions"],
            "risk_multiplier": profile["risk_multiplier"],
            "risk_level": self._get_risk_label(avg_risk),
            "recent_interactions": len(profile["risk_history"])
        }
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = ConversationalFraudDetector("http://localhost:8000/chat")
    
    # Test messages
    test_messages = [
        "Hello, I need help with my account",
        "I DEMAND an immediate refund! This is urgent and I need my money back RIGHT NOW!",
        "I've been trying to get my refund for weeks. You people are useless!",
        "Click here to verify your account details immediately or your account will be closed",
        "Congratulations! You've won $10,000! Send us your bank details to claim your prize!"
    ]
    
    print("🔍 Testing Conversational Fraud Detection:")
    print("=" * 50)
    
    for i, message in enumerate(test_messages, 1):
        analysis = detector.analyze_message(message, user_id="test_user")
        print(f"\n{i}. Message: {message}")
        print(f"   Risk Score: {analysis.fraud_likelihood:.2f}")
        print(f"   Label: {analysis.label}")
        print(f"   Explanation: {analysis.explanation}")
        print(f"   Risk Factors: {analysis.risk_factors}")
        print("-" * 50)
    
    # Show user risk summary
    summary = detector.get_user_risk_summary("test_user")
    print(f"\n📊 User Risk Summary:")
    print(f"   Average Risk: {summary['average_risk']:.2f}")
    print(f"   Risk Level: {summary['risk_level']}")
    print(f"   Total Interactions: {summary['total_interactions']}")