"""
Configuración global para pytest.
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# Añadir el directorio raíz al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_agents.config.settings import settings, Settings
from ai_agents.core.types import AgentConfig, Message, MessageRole


@pytest.fixture(scope="session")
def event_loop():
    """Crear event loop para tests async."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Configuración para tests."""
    return Settings(
        openai_api_key="test-key-12345",
        default_model="gpt-3.5-turbo",
        log_level="DEBUG",
        memory_backend="mock",
        data_dir="test_data",
        cache_dir="test_data/cache"
    )


@pytest.fixture
def mock_agent_config():
    """Configuración mock para agentes."""
    return AgentConfig(
        agent_type="TestAgent",
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100,
        system_message="Eres un agente de prueba."
    )


@pytest.fixture
def test_agent_id():
    """ID fijo para agentes de prueba para evitar UUIDs aleatorios."""
    return "test_agent_fixed_id"


@pytest.fixture
def sample_message():
    """Mensaje de prueba."""
    return Message(
        role=MessageRole.USER,
        content="Este es un mensaje de prueba"
    )


@pytest.fixture
def mock_openai_client():
    """Cliente OpenAI mock."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="Respuesta de prueba",
                    role="assistant"
                )
            )
        ],
        usage=MagicMock(total_tokens=50)
    )
    return mock_client


@pytest.fixture
def temp_data_dir(tmp_path):
    """Directorio temporal para datos de test."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    cache_dir = data_dir / "cache"
    cache_dir.mkdir()
    return data_dir


@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch):
    """Mock de variables de entorno para tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MEMORY_BACKEND", "mock")


# Marcadores personalizados
def pytest_configure(config):
    """Configurar marcadores personalizados."""
    config.addinivalue_line(
        "markers", "unit: marcar test como unitario"
    )
    config.addinivalue_line(
        "markers", "integration: marcar test como de integración"
    )
    config.addinivalue_line(
        "markers", "slow: marcar test como lento"
    )
