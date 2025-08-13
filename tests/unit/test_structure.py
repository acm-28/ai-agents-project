"""
Test básico para verificar que la nueva estructura funciona correctamente.
"""

import pytest
from ai_agents import settings, BaseAgent
from ai_agents.core.types import AgentConfig, Message, MessageRole, AgentState
from ai_agents.core.exceptions import AgentError


@pytest.mark.unit
def test_settings_import():
    """Test que verifica que la configuración se importa correctamente."""
    assert settings.default_model == "gpt-3.5-turbo"
    assert settings.temperature == 0.7
    assert settings.max_tokens == 1000


@pytest.mark.unit 
def test_types_creation():
    """Test que verifica que los tipos se crean correctamente."""
    # Test AgentConfig
    config = AgentConfig(
        agent_type="TestAgent",
        model="gpt-4",
        temperature=0.5
    )
    assert config.agent_type == "TestAgent"
    assert config.model == "gpt-4"
    assert config.temperature == 0.5
    
    # Test Message
    message = Message(
        role=MessageRole.USER,
        content="Hola mundo"
    )
    assert message.role == MessageRole.USER.value  # Comparar con el valor del enum
    assert message.content == "Hola mundo"
    assert message.id is not None


@pytest.mark.unit
def test_agent_states():
    """Test que verifica los estados de agente."""
    assert AgentState.CREATED.value == "created"  # Comparar con el valor del enum
    assert AgentState.READY.value == "ready"
    assert AgentState.PROCESSING.value == "processing"


@pytest.mark.unit
def test_base_agent_creation():
    """Test que verifica que se puede crear un BaseAgent."""
    # BaseAgent es abstracto, así que no se puede instanciar directamente
    # pero podemos verificar que la clase existe
    assert BaseAgent is not None
    assert hasattr(BaseAgent, 'initialize')
    assert hasattr(BaseAgent, 'process')


@pytest.mark.unit
def test_exceptions():
    """Test que verifica las excepciones personalizadas."""
    error = AgentError("Test error", agent_id="test-123")
    assert str(error) == "Test error"
    assert error.agent_id == "test-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
