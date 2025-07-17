"""
LangChain Tools for Customer Service Operations

This module provides tools that the AI agent can use to access customer data
and perform service operations like billing inquiries and network status checks.
"""

from langchain.tools import tool
from typing import str
import random
from datetime import datetime, timedelta


@tool
def get_billing_info(user_id: str) -> str:
    """
    Retrieve billing information for a specific user.
    
    Args:
        user_id (str): The unique identifier for the customer
        
    Returns:
        str: Formatted billing information including current balance and due date
    """
    if not user_id or not isinstance(user_id, str):
        return "Error: Invalid user ID provided. Please provide a valid customer ID."
    
    # Simulate realistic billing data
    bill_amounts = [45.99, 67.50, 89.99, 123.45, 156.78]
    current_bill = random.choice(bill_amounts)
    
    # Generate a due date 15-30 days from now
    days_until_due = random.randint(15, 30)
    due_date = (datetime.now() + timedelta(days=days_until_due)).strftime("%Y-%m-%d")
    
    # Simulate account status
    account_status = random.choice(["Current", "Past Due", "Credit Available"])
    
    billing_info = f"""
    Billing Information for Customer ID: {user_id}
    
    Current Bill Amount: ${current_bill:.2f}
    Due Date: {due_date}
    Account Status: {account_status}
    Last Payment: ${current_bill * 0.9:.2f} on {(datetime.now() - timedelta(days=32)).strftime("%Y-%m-%d")}
    
    Payment Options:
    - Online: www.telecom.com/pay
    - Phone: 1-800-PAY-BILL
    - Auto-pay: Available for setup
    """
    
    return billing_info.strip()


@tool
def check_network_status(area_code: str) -> str:
    """
    Check network status and outages for a specific area code.
    
    Args:
        area_code (str): The area code to check (e.g., "555", "312", "917")
        
    Returns:
        str: Network status information including any ongoing issues
    """
    if not area_code or not isinstance(area_code, str):
        return "Error: Invalid area code provided. Please provide a valid 3-digit area code."
    
    # Clean and validate area code
    area_code = area_code.strip()
    if not area_code.isdigit() or len(area_code) != 3:
        return f"Error: '{area_code}' is not a valid area code. Please provide a 3-digit area code."
    
    # Simulate network conditions
    network_conditions = [
        "optimal",
        "good", 
        "fair",
        "poor",
        "outage"
    ]
    
    current_status = random.choice(network_conditions)
    
    if current_status == "outage":
        estimated_fix = random.randint(2, 6)
        return f"""
        Network Status for Area Code {area_code}:
        
        🔴 OUTAGE DETECTED
        Status: Network service interruption
        Affected Services: Voice calls, Data, Text messaging
        Estimated Resolution: {estimated_fix} hours
        
        Our engineers are actively working to restore service. We apologize for the inconvenience.
        
        For updates: www.telecom.com/status or text STATUS to 12345
        """
    
    elif current_status == "poor":
        return f"""
        Network Status for Area Code {area_code}:
        
        🟡 DEGRADED SERVICE
        Status: Reduced network performance
        Affected Services: Slower data speeds, possible call drops
        Impact: Minor service disruptions
        
        We're monitoring the situation and working to optimize performance.
        Try connecting to WiFi for better data speeds.
        """
    
    else:
        signal_strength = random.randint(85, 99)
        return f"""
        Network Status for Area Code {area_code}:
        
        🟢 SERVICE NORMAL
        Status: All systems operational
        Signal Strength: {signal_strength}%
        Data Speed: Optimal
        Voice Quality: Excellent
        
        No known issues in your area. If you're experiencing problems, 
        try restarting your device or contact technical support.
        """


# Additional utility functions for the tools
def _validate_user_id(user_id: str) -> bool:
    """Validate user ID format (for future enhancement)"""
    return user_id and isinstance(user_id, str) and len(user_id) > 0


def _validate_area_code(area_code: str) -> bool:
    """Validate area code format"""
    return (area_code and isinstance(area_code, str) and 
            area_code.strip().isdigit() and len(area_code.strip()) == 3)