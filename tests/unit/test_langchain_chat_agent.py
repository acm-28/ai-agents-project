"""
Tests para LangChainChatAgent - Versión consolidada.
Cubre todas las funcionalidades del agente de chat unificado.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from ai_agents.agents.chat.langchain_agent import LangChainChatAgent
from ai_agents.core.types import AgentConfig, Message, MessageRole, AgentResponse
from ai_agents.core.exceptions import InitializationError, ProcessingError


@pytest.fixture
def agent_config():
    """Configuración básica para tests."""
    return AgentConfig(
        agent_type="LangChainChatAgent",
        model="gpt-3.5-turbo",
        temperature=0.7,
        system_message="Eres un asistente de prueba."
    )


@pytest.fixture
def temp_memory_dir():
    """Directorio temporal para persistencia."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestLangChainChatAgentBasic:
    """Tests básicos de creación y configuración."""
    
    @pytest.mark.unit
    def test_agent_creation_with_persistence(self, agent_config):
        """Test creación del agente con persistencia habilitada."""
        agent = LangChainChatAgent(
            agent_id="test_agent_1",
            config=agent_config,
            enable_persistence=True
        )
        
        assert agent is not None
        assert agent.agent_id == "test_agent_1"
        assert agent.enable_persistence is True
        assert agent.system_message == "Eres un asistente de prueba."
        assert agent.config.agent_type == "LangChainChatAgent"
    
    @pytest.mark.unit
    def test_agent_creation_without_persistence(self, agent_config):
        """Test creación del agente sin persistencia."""
        agent = LangChainChatAgent(
            agent_id="test_agent_2",
            config=agent_config,
            enable_persistence=False
        )
        
        assert agent is not None
        assert agent.enable_persistence is False
        assert agent.system_message == "Eres un asistente de prueba."
    
    @pytest.mark.unit
    def test_agent_default_configuration(self, test_agent_id):
        """Test creación con configuración por defecto."""
        agent = LangChainChatAgent(agent_id=test_agent_id, enable_persistence=False)
        
        assert agent is not None
        assert agent.enable_persistence is False
        assert agent.system_message == "Eres un asistente útil y amigable."


class TestLangChainChatAgentSessionManagement:
    """Tests de gestión de sesiones."""
    
    @pytest.mark.unit
    def test_session_creation_and_management(self, agent_config, test_agent_id):
        """Test creación y gestión de sesiones."""
        agent = LangChainChatAgent(
            agent_id=test_agent_id,
            config=agent_config,
            enable_persistence=False
        )
        
        # Inicialmente sin sesiones
        assert agent.get_session_count() == 0
        
        # Crear sesiones
        history1 = agent.get_session_history("session_1")
        assert history1 is not None
        assert agent.get_session_count() == 1
        
        history2 = agent.get_session_history("session_2")
        assert history2 is not None
        assert agent.get_session_count() == 2
        
        # Verificar que son diferentes
        assert history1 is not history2
    
    @pytest.mark.unit
    def test_clear_specific_session(self, agent_config, test_agent_id):
        """Test limpieza de sesión específica."""
        agent = LangChainChatAgent(
            agent_id=test_agent_id,
            config=agent_config,
            enable_persistence=False
        )
        
        # Crear sesiones
        agent.get_session_history("session_1")
        agent.get_session_history("session_2")
        assert agent.get_session_count() == 2
        
        # Limpiar una sesión
        cleared = agent.clear_session("session_1")
        assert cleared is True
        assert agent.get_session_count() == 1
        
        # Intentar limpiar sesión inexistente
        cleared = agent.clear_session("nonexistent")
        assert cleared is False
        assert agent.get_session_count() == 1
    
    @pytest.mark.unit
    def test_clear_all_sessions(self, agent_config, test_agent_id):
        """Test limpieza de todas las sesiones."""
        agent = LangChainChatAgent(
            agent_id=test_agent_id,
            config=agent_config,
            enable_persistence=False
        )
        
        # Crear múltiples sesiones
        agent.get_session_history("session_1")
        agent.get_session_history("session_2")
        agent.get_session_history("session_3")
        assert agent.get_session_count() == 3
        
        # Limpiar todas
        count = agent.clear_all_sessions()
        assert count == 3
        assert agent.get_session_count() == 0
    
    @pytest.mark.unit
    def test_get_session_memory_size(self, agent_config, test_agent_id):
        """Test obtener tamaño de memoria de sesión."""
        agent = LangChainChatAgent(
            agent_id=test_agent_id,
            config=agent_config,
            enable_persistence=False
        )
        
        # Sesión inexistente
        size = agent.get_session_memory_size("nonexistent")
        assert size == 0
        
        # Sesión existente pero vacía
        agent.get_session_history("test_session")
        size = agent.get_session_memory_size("test_session")
        assert size == 0


class TestLangChainChatAgentSummaries:
    """Tests de funcionalidades de resúmenes."""
    
    @pytest.mark.unit
    def test_get_session_summary_empty(self, agent_config, test_agent_id):
        """Test resumen de sesión vacía."""
        agent = LangChainChatAgent(
            agent_id=test_agent_id,
            config=agent_config,
            enable_persistence=False
        )
        
        # Sesión inexistente
        summary = agent.get_session_summary("nonexistent")
        assert summary is None
        
        # Sesión existente pero vacía
        agent.get_session_history("empty_session")
        summary = agent.get_session_summary("empty_session")
        
        assert summary is not None
        assert summary["session_id"] == "empty_session"
        assert summary["total_messages"] == 0
        assert summary["user_messages"] == 0
        assert summary["ai_messages"] == 0
        assert summary["conversation_length"] == 0
        assert summary["last_question"] is None
        assert summary["last_answer"] is None
    
    @pytest.mark.unit
    def test_get_session_summary_with_messages(self, agent_config, test_agent_id):
        """Test resumen de sesión con mensajes simulados."""
        agent = LangChainChatAgent(
            agent_id=test_agent_id,
            config=agent_config,
            enable_persistence=False
        )
        
        # Crear sesión y simular mensajes
        history = agent.get_session_history("test_session")
        history.add_user_message("Hola, ¿cómo estás?")
        history.add_ai_message("¡Hola! Estoy bien, gracias.")
        history.add_user_message("¿Qué tiempo hace?")
        history.add_ai_message("Hace un día soleado.")
        
        summary = agent.get_session_summary("test_session")
        
        assert summary is not None
        assert summary["session_id"] == "test_session"
        assert summary["total_messages"] == 4
        assert summary["user_messages"] == 2
        assert summary["ai_messages"] == 2
        assert summary["last_question"] == "¿Qué tiempo hace?"
        assert summary["last_answer"] == "Hace un día soleado."
        assert summary["conversation_length"] > 0
    
    @pytest.mark.unit
    def test_get_all_sessions_summary(self, agent_config, test_agent_id):
        """Test resumen de todas las sesiones."""
        agent = LangChainChatAgent(
            agent_id=test_agent_id,
            config=agent_config,
            enable_persistence=False
        )
        
        # Sin sesiones
        summaries = agent.get_all_sessions_summary()
        assert summaries == {}
        
        # Crear sesiones
        agent.get_session_history("session_1")
        agent.get_session_history("session_2")
        
        summaries = agent.get_all_sessions_summary()
        assert len(summaries) == 2
        assert "session_1" in summaries
        assert "session_2" in summaries
        assert summaries["session_1"]["session_id"] == "session_1"
        assert summaries["session_2"]["session_id"] == "session_2"


class TestLangChainChatAgentPersistence:
    """Tests de persistencia en archivos."""
    
    @pytest.mark.unit
    def test_memory_directory_creation(self, agent_config, temp_memory_dir):
        """Test creación de directorio de memoria."""
        with patch('ai_agents.config.settings.settings.memory_dir', temp_memory_dir):
            agent = LangChainChatAgent(
                agent_id="test_agent",
                config=agent_config,
                enable_persistence=True
            )
            
            memory_dir = agent._get_memory_dir_path()
            assert memory_dir.exists()
            assert memory_dir.name == "test_agent"
    
    @pytest.mark.unit
    def test_session_file_path_generation(self, agent_config, temp_memory_dir):
        """Test generación de rutas de archivos de sesión."""
        with patch('ai_agents.config.settings.settings.memory_dir', temp_memory_dir):
            agent = LangChainChatAgent(
                agent_id="test_agent",
                config=agent_config,
                enable_persistence=True
            )
            
            session_file = agent._get_session_file_path("test_session")
            assert session_file.name == "test_session_session.json"
            assert "test_agent" in str(session_file)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_session_to_file(self, agent_config, temp_memory_dir):
        """Test guardado de sesión en archivo."""
        with patch('ai_agents.config.settings.settings.memory_dir', temp_memory_dir):
            agent = LangChainChatAgent(
                agent_id="test_agent",
                config=agent_config,
                enable_persistence=True
            )
            
            # Crear sesión con mensajes
            history = agent.get_session_history("test_session")
            history.add_user_message("Pregunta de prueba")
            history.add_ai_message("Respuesta de prueba")
            
            # Guardar sesión
            await agent._save_session_to_file("test_session")
            
            # Verificar que el archivo existe
            session_file = agent._get_session_file_path("test_session")
            assert session_file.exists()
            
            # Verificar contenido del archivo
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data["agent_id"] == "test_agent"
            assert data["session_id"] == "test_session"
            assert data["total_messages"] == 2
            assert len(data["messages"]) == 2
            assert data["messages"][0]["type"] == "human"
            assert data["messages"][1]["type"] == "ai"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_load_session_from_file(self, agent_config, temp_memory_dir):
        """Test carga de sesión desde archivo."""
        with patch('ai_agents.config.settings.settings.memory_dir', temp_memory_dir):
            agent = LangChainChatAgent(
                agent_id="test_agent",
                config=agent_config,
                enable_persistence=True
            )
            
            # Crear archivo de sesión manualmente
            session_file = agent._get_session_file_path("test_session")
            session_file.parent.mkdir(parents=True, exist_ok=True)
            
            session_data = {
                "agent_id": "test_agent",
                "session_id": "test_session",
                "total_messages": 2,
                "messages": [
                    {"type": "human", "content": "Pregunta cargada", "timestamp": ""},
                    {"type": "ai", "content": "Respuesta cargada", "timestamp": ""}
                ],
                "system_message": "Test system message"
            }
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f)
            
            # Cargar sesión
            loaded = await agent._load_session_from_file("test_session")
            assert loaded is True
            
            # Verificar que la sesión se cargó en memoria
            assert "test_session" in agent.store
            assert len(agent.store["test_session"].messages) == 2


class TestLangChainChatAgentSystemMessage:
    """Tests de configuración del mensaje del sistema."""
    
    @pytest.mark.unit
    def test_set_system_message(self, agent_config, test_agent_id):
        """Test cambio del mensaje del sistema."""
        agent = LangChainChatAgent(
            agent_id=test_agent_id,
            config=agent_config,
            enable_persistence=False
        )
        original_message = agent.system_message
        
        new_message = "Nuevo mensaje del sistema para testing"
        agent.set_system_message(new_message)
        
        assert agent.system_message == new_message
        assert agent.config.system_message == new_message
        assert agent.system_message != original_message


class TestLangChainChatAgentProcessing:
    """Tests de procesamiento de mensajes (con mocks)."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_string_input(self, agent_config):
        """Test procesamiento con entrada de string."""
        mock_response = MagicMock()
        mock_response.content = "Respuesta simulada"
        
        agent = LangChainChatAgent(config=agent_config, enable_persistence=False)
        
        with patch('ai_agents.agents.chat.langchain_agent.ChatOpenAI') as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance
            
            agent.chain_with_history = MagicMock()
            agent.chain_with_history.invoke.return_value = mock_response
            
            response = await agent.process("Mensaje de prueba")
            
            assert isinstance(response, AgentResponse)
            assert response.content == "Respuesta simulada"
            assert response.metadata["session_id"] == "default"
            assert "model" in response.metadata
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_object_input(self, agent_config):
        """Test procesamiento con objeto Message."""
        mock_response = MagicMock()
        mock_response.content = "Respuesta a mensaje"
        
        agent = LangChainChatAgent(config=agent_config, enable_persistence=False)
        
        with patch('ai_agents.agents.chat.langchain_agent.ChatOpenAI') as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance
            
            agent.chain_with_history = MagicMock()
            agent.chain_with_history.invoke.return_value = mock_response
            
            message = Message(
                content="Contenido del mensaje",
                role=MessageRole.USER,
                session_id="custom_session"
            )
            
            response = await agent.process(message)
            
            assert isinstance(response, AgentResponse)
            assert response.content == "Respuesta a mensaje"
            assert response.metadata["session_id"] == "custom_session"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_dict_input(self, agent_config):
        """Test procesamiento con entrada de diccionario."""
        mock_response = MagicMock()
        mock_response.content = "Respuesta a dict"
        
        agent = LangChainChatAgent(config=agent_config, enable_persistence=False)
        
        with patch('ai_agents.agents.chat.langchain_agent.ChatOpenAI') as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance
            
            agent.chain_with_history = MagicMock()
            agent.chain_with_history.invoke.return_value = mock_response
            
            input_dict = {
                "content": "Mensaje desde dict",
                "session_id": "dict_session"
            }
            
            response = await agent.process(input_dict)
            
            assert isinstance(response, AgentResponse)
            assert response.content == "Respuesta a dict"
            assert response.metadata["session_id"] == "dict_session"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_empty_message_error(self, agent_config, test_agent_id):
        """Test error con mensaje vacío."""
        agent = LangChainChatAgent(
            agent_id=test_agent_id,
            config=agent_config,
            enable_persistence=False
        )
        
        with pytest.raises(ProcessingError):
            await agent.process("")
        
        with pytest.raises(ProcessingError):
            await agent.process("   ")
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_invalid_input_type_error(self, agent_config, test_agent_id):
        """Test error con tipo de entrada inválido."""
        agent = LangChainChatAgent(
            agent_id=test_agent_id,
            config=agent_config,
            enable_persistence=False
        )
        
        with pytest.raises(ProcessingError):
            await agent.process(123)  # int no soportado
        
        with pytest.raises(ProcessingError):
            await agent.process(None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_cleanup(agent_config, test_agent_id):
    """Test limpieza de recursos del agente."""
    agent = LangChainChatAgent(
        agent_id=test_agent_id,
        config=agent_config,
        enable_persistence=False
    )
    
    # Crear algunas sesiones
    agent.get_session_history("session_1")
    agent.get_session_history("session_2")
    assert agent.get_session_count() == 2
    
    # Limpiar
    await agent.cleanup()
    assert agent.get_session_count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
