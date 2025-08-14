"""
AI Agents Framework - Framework for building intelligent AI agents with LangChain.

Este paquete proporciona una estructura modular y escalable para crear
agentes de IA con diferentes capacidades como chat, análisis de datos,
Q&A y workflows complejos.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from ai_agents.config.settings import settings
from ai_agents.core.base_agent import BaseAgent
from ai_agents.core.exceptions import AgentError

# Agentes especializados
from ai_agents.agents.chat.langchain_agent import LangChainChatAgent
from ai_agents.agents.data_analysis.pandas_agent import PandasAgent

__all__ = [
    "settings",
    "BaseAgent", 
    "AgentError",
    "LangChainChatAgent",
    "PandasAgent"
]
