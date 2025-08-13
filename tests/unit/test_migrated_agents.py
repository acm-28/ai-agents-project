"""
Tests para los agentes migrados.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from ai_agents.agents.chat.langchain_agent import LangChainChatAgent
from ai_agents.agents.chat.llm_agent import LLMChatAgent
from ai_agents.agents.qa.memory_qa_agent import MemoryQAAgent
from ai_agents.core.types import AgentConfig, Message, MessageRole, AgentResponse


@pytest.mark.unit
@pytest.mark.asyncio
async def test_langchain_chat_agent_creation():
    """Test que verifica la creación del LangChainChatAgent."""
    config = AgentConfig(
        agent_type="LangChainChatAgent",
        model="gpt-3.5-turbo",
        temperature=0.7,
        system_message="Eres un agente de prueba."
    )
    
    agent = LangChainChatAgent(config=config)
    assert agent is not None
    assert agent.config.agent_type == "LangChainChatAgent"
    assert agent.system_message == "Eres un agente de prueba."


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_chat_agent_creation():
    """Test que verifica la creación del LLMChatAgent."""
    config = AgentConfig(
        agent_type="LLMChatAgent",
        model="gpt-3.5-turbo",
        temperature=0.5
    )
    
    agent = LLMChatAgent(config=config)
    assert agent is not None
    assert agent.config.agent_type == "LLMChatAgent"
    assert agent.config.temperature == 0.5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_qa_agent_creation():
    """Test que verifica la creación del MemoryQAAgent."""
    config = AgentConfig(
        agent_type="MemoryQAAgent",
        model="gpt-3.5-turbo"
    )
    
    agent = MemoryQAAgent(config=config)
    assert agent is not None
    assert agent.config.agent_type == "MemoryQAAgent"


@pytest.mark.unit
@pytest.mark.asyncio 
async def test_agent_info_methods():
    """Test que verifica los métodos de información de agentes."""
    agent = LangChainChatAgent()
    
    info = agent.get_info()
    assert "agent_id" in info
    assert "agent_type" in info
    assert "state" in info
    assert info["agent_type"] == "LangChainChatAgent"


@pytest.mark.unit
def test_session_management_langchain_agent():
    """Test que verifica la gestión de sesiones del LangChainChatAgent."""
    agent = LangChainChatAgent()
    
    # Test sesión nueva
    history = agent.get_session_history("test-session-1")
    assert history is not None
    assert agent.get_session_count() == 1
    
    # Test otra sesión
    history2 = agent.get_session_history("test-session-2")
    assert history2 is not None
    assert agent.get_session_count() == 2
    
    # Test limpiar sesión
    cleared = agent.clear_session("test-session-1")
    assert cleared is True
    assert agent.get_session_count() == 1
    
    # Test limpiar todas las sesiones
    count = agent.clear_all_sessions()
    assert count == 1
    assert agent.get_session_count() == 0


@pytest.mark.unit
def test_memory_qa_agent_memory():
    """Test que verifica la memoria del MemoryQAAgent."""
    agent = MemoryQAAgent()
    
    # La memoria debe estar inicialmente vacía
    assert agent.get_memory_size() == 0
    
    # Después de inicializar, debe poder limpiar memoria
    agent.clear_memory()
    assert agent.get_memory_size() == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_processing_mock():
    """Test de integración con mocks para verificar el procesamiento."""
    
    # Mock de respuesta de OpenAI
    mock_response = MagicMock()
    mock_response.content = "Esta es una respuesta de prueba"
    mock_response.choices = [MagicMock(
        message=MagicMock(content="Esta es una respuesta de prueba"),
        finish_reason="stop"
    )]
    mock_response.usage = MagicMock(total_tokens=50)
    
    config = AgentConfig(
        agent_type="LLMChatAgent",
        model="gpt-3.5-turbo"
    )
    
    agent = LLMChatAgent(config=config)
    
    # Mock del cliente OpenAI
    with patch('openai.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client.models.list.return_value = MagicMock()
        mock_openai.return_value = mock_client
        
        await agent.initialize()
        
        # Test procesamiento con string
        response = await agent.process("Hola, ¿cómo estás?")
        assert isinstance(response, AgentResponse)
        assert response.content == "Esta es una respuesta de prueba"
        assert response.tokens_used == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
