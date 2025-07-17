"""
Project Kural - Core AI Modules

This package contains the core AI modules for the intelligent customer service agent:
- PerceptionModule: Handles speech recognition and sentiment analysis
- MemoryModule: Manages short-term and long-term conversation memory
- KuralAgent: Main agent orchestrator that combines all components
- Tools: LangChain tools for accessing customer data and services
"""

from .perception import PerceptionModule
from .memory import MemoryModule
from .agent import KuralAgent
from .tools import get_billing_info, check_network_status

__all__ = [
    'PerceptionModule',
    'MemoryModule', 
    'KuralAgent',
    'get_billing_info',
    'check_network_status'
]

__version__ = '1.0.0'