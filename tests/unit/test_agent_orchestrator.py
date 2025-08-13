"""
Tests para AgentOrchestrator - Orquestador unificado de agentes.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from ai_agents.agents.orchestration.agent_orchestrator import (
    AgentOrchestrator,
    TaskType,
    TaskClassification,
    AgentCapability
)
from ai_agents.core.types import AgentResponse, AgentState


class TestAgentOrchestrator:
    """Test suite para AgentOrchestrator."""

    @pytest_asyncio.fixture
    async def orchestrator(self):
        """Fixture del orquestrador."""
        orchestrator = AgentOrchestrator()
        
        # Mock agentes especializados para evitar inicialización real
        mock_agents = {}
        
        agent_names = ['pandas_agent', 'sophisticated_agent', 'memory_qa_agent', 
                      'langchain_agent', 'llm_agent']
        
        for agent_name in agent_names:
            mock_agent = MagicMock()
            mock_agent.is_ready.return_value = True
            mock_agent.state = AgentState.READY
            mock_agent.__class__.__name__ = f"Mock{agent_name.title().replace('_', '')}"
            mock_agent.process = AsyncMock(return_value=AgentResponse(
                content=f"Response from {agent_name}",
                metadata={"test": True}
            ))
            mock_agent.get_capabilities.return_value = [f"Capability of {agent_name}"]
            mock_agents[agent_name] = mock_agent
        
        # Configurar los agentes mock
        with patch.object(orchestrator, '_initialize_specialized_agents') as mock_init:
            mock_init.return_value = None
            await orchestrator.initialize()
            orchestrator.specialized_agents = mock_agents
        
        return orchestrator

    @pytest.fixture
    def sample_requests(self):
        """Fixture con solicitudes de muestra para diferentes tipos de tareas."""
        return {
            'data_analysis': {
                'message': 'analizar datos del archivo ventas.csv',
                'file_path': 'ventas.csv'
            },
            'text_analysis': {
                'message': 'clasificar este texto y extraer entidades',
                'text': 'Apple Inc. announced new iPhone model in Cupertino, California.'
            },
            'qa_memory': {
                'message': 'qué me dijiste antes sobre el análisis anterior?'
            },
            'chat': {
                'message': 'hola, cómo estás?'
            },
            'complex_workflow': {
                'message': 'ejecutar un workflow complejo con múltiples pasos'
            },
            'unknown': {
                'message': 'solicitud muy ambigua sin contexto claro'
            }
        }

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test inicialización del orquestrador."""
        assert orchestrator.state == AgentState.CREATED  # Sin _safe_initialize
        assert len(orchestrator.agent_capabilities) == 5
        assert len(orchestrator.specialized_agents) == 5
        
        # Verificar configuración de capacidades
        assert 'pandas_agent' in orchestrator.agent_capabilities
        assert 'sophisticated_agent' in orchestrator.agent_capabilities
        assert 'memory_qa_agent' in orchestrator.agent_capabilities

    @pytest.mark.asyncio
    async def test_data_analysis_classification(self, orchestrator, sample_requests):
        """Test clasificación de tareas de análisis de datos."""
        request = sample_requests['data_analysis']
        
        classification = await orchestrator._classify_task(request['message'], request)
        
        assert classification.task_type == TaskType.DATA_ANALYSIS
        assert classification.agent_name == 'pandas_agent'
        assert classification.confidence > 0

    @pytest.mark.asyncio
    async def test_text_analysis_classification(self, orchestrator, sample_requests):
        """Test clasificación de tareas de análisis de texto."""
        request = sample_requests['text_analysis']
        
        classification = await orchestrator._classify_task(request['message'], request)
        
        assert classification.task_type == TaskType.TEXT_ANALYSIS
        assert classification.agent_name == 'sophisticated_agent'
        assert classification.confidence > 0

    @pytest.mark.asyncio
    async def test_qa_memory_classification(self, orchestrator, sample_requests):
        """Test clasificación de tareas de Q&A con memoria."""
        request = sample_requests['qa_memory']
        
        classification = await orchestrator._classify_task(request['message'], request)
        
        assert classification.task_type == TaskType.QA_MEMORY
        assert classification.agent_name == 'memory_qa_agent'
        assert classification.confidence > 0

    @pytest.mark.asyncio
    async def test_chat_classification(self, orchestrator, sample_requests):
        """Test clasificación de tareas de chat."""
        request = sample_requests['chat']
        
        classification = await orchestrator._classify_task(request['message'], request)
        
        assert classification.task_type == TaskType.CHAT
        assert classification.agent_name == 'langchain_agent'
        assert classification.confidence > 0

    @pytest.mark.asyncio
    async def test_agent_selection(self, orchestrator):
        """Test selección de agente apropiado."""
        classification = TaskClassification(
            task_type=TaskType.DATA_ANALYSIS,
            confidence=0.8,
            agent_name='pandas_agent',
            reasoning="Test",
            parameters={}
        )
        
        selected_agent = await orchestrator._select_agent(classification)
        
        assert selected_agent is not None
        assert selected_agent == orchestrator.specialized_agents['pandas_agent']

    @pytest.mark.asyncio
    async def test_process_data_analysis_request(self, orchestrator, sample_requests):
        """Test procesamiento completo de solicitud de análisis de datos."""
        request = sample_requests['data_analysis']
        
        response = await orchestrator.process(request)
        
        assert isinstance(response, AgentResponse)
        assert not response.metadata.get('error', False)
        assert 'orchestrator' in response.metadata
        assert response.metadata['orchestrator']['selected_agent'] == 'MockPandasAgent'
        
        # Verificar que el agente correcto fue llamado
        pandas_agent = orchestrator.specialized_agents['pandas_agent']
        pandas_agent.process.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_process_text_analysis_request(self, orchestrator, sample_requests):
        """Test procesamiento de solicitud de análisis de texto."""
        request = sample_requests['text_analysis']
        
        response = await orchestrator.process(request)
        
        assert isinstance(response, AgentResponse)
        assert not response.metadata.get('error', False)
        assert response.metadata['orchestrator']['selected_agent'] == 'MockSophisticatedAgent'
        
        # Verificar que el agente correcto fue llamado
        sophisticated_agent = orchestrator.specialized_agents['sophisticated_agent']
        sophisticated_agent.process.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_fallback_when_agent_unavailable(self, orchestrator, sample_requests):
        """Test fallback cuando el agente principal no está disponible."""
        # Hacer que pandas_agent no esté disponible
        orchestrator.specialized_agents['pandas_agent'].is_ready.return_value = False
        
        request = sample_requests['data_analysis']
        response = await orchestrator.process(request)
        
        assert isinstance(response, AgentResponse)
        # Debería usar un agente alternativo o fallback
        assert 'orchestrator' in response.metadata

    @pytest.mark.asyncio
    async def test_empty_message_handling(self, orchestrator):
        """Test manejo de mensaje vacío."""
        response = await orchestrator.process({'message': ''})
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get('error') is True
        assert response.metadata.get('needs_message') is True

    @pytest.mark.asyncio
    async def test_context_updates(self, orchestrator, sample_requests):
        """Test actualización de contexto de sesión."""
        request = sample_requests['data_analysis']
        
        # Procesar primera solicitud
        await orchestrator.process(request)
        
        # Verificar que el contexto se actualizó
        assert orchestrator.current_session_context['last_task_type'] == TaskType.DATA_ANALYSIS
        assert orchestrator.current_session_context['last_agent'] == 'pandas_agent'
        assert len(orchestrator.interaction_history) == 1

    @pytest.mark.asyncio
    async def test_context_continuity_bonus(self, orchestrator, sample_requests):
        """Test bonificación por continuidad de contexto."""
        # Primera solicitud de análisis de datos
        await orchestrator.process(sample_requests['data_analysis'])
        
        # Segunda solicitud similar debería tener bonificación
        request2 = {'message': 'continuar con el análisis'}
        classification = await orchestrator._classify_task(request2['message'], request2)
        
        # La bonificación por continuidad debería influir en la clasificación
        assert classification.task_type == TaskType.DATA_ANALYSIS

    @pytest.mark.asyncio
    async def test_agent_error_handling(self, orchestrator, sample_requests):
        """Test manejo de errores de agentes."""
        # Configurar agente para que lance error
        orchestrator.specialized_agents['pandas_agent'].process.side_effect = Exception("Agent error")
        
        request = sample_requests['data_analysis']
        response = await orchestrator.process(request)
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get('error') is True
        assert 'agent_error' in response.metadata

    @pytest.mark.asyncio
    async def test_capabilities_aggregation(self, orchestrator):
        """Test agregación de capacidades de todos los agentes."""
        capabilities = orchestrator.get_capabilities()
        
        assert isinstance(capabilities, list)
        assert len(capabilities) > 5  # Capacidades del orquestrador + agentes
        assert any('Orquestación automática' in cap for cap in capabilities)
        assert any('pandas_agent:' in cap for cap in capabilities)

    @pytest.mark.asyncio
    async def test_orchestration_stats(self, orchestrator, sample_requests):
        """Test estadísticas de orquestación."""
        # Procesar varias solicitudes
        await orchestrator.process(sample_requests['data_analysis'])
        await orchestrator.process(sample_requests['text_analysis'])
        await orchestrator.process(sample_requests['chat'])
        
        stats = orchestrator.get_orchestration_stats()
        
        assert stats['total_interactions'] == 3
        assert stats['success_rate'] == 1.0  # Todos exitosos
        assert 'task_distribution' in stats
        assert 'agent_usage' in stats
        assert len(stats['task_distribution']) >= 2  # Al menos 2 tipos de tareas

    @pytest.mark.asyncio
    async def test_available_agents_info(self, orchestrator):
        """Test información de agentes disponibles."""
        agents_info = orchestrator.get_available_agents()
        
        assert isinstance(agents_info, dict)
        assert len(agents_info) == 5
        
        for agent_name, info in agents_info.items():
            assert 'class' in info
            assert 'state' in info
            assert 'ready' in info
            assert 'task_types' in info
            assert 'description' in info

    @pytest.mark.asyncio
    async def test_pattern_matching_priority(self, orchestrator):
        """Test prioridad en matching de patrones."""
        # Mensaje que podría coincidir con múltiples patrones
        ambiguous_request = {
            'message': 'analizar el texto de este archivo csv sobre datos financieros'
        }
        
        classification = await orchestrator._classify_task(
            ambiguous_request['message'], 
            ambiguous_request
        )
        
        # Debería clasificarse como DATA_ANALYSIS por mayor especificidad
        assert classification.task_type == TaskType.DATA_ANALYSIS

    @pytest.mark.asyncio
    async def test_parameter_influence_on_classification(self, orchestrator):
        """Test influencia de parámetros en la clasificación."""
        # Mensaje neutral pero con parámetro de archivo
        request = {
            'message': 'procesar esto',
            'file_path': 'data.csv'
        }
        
        classification = await orchestrator._classify_task(request['message'], request)
        
        # La presencia de file_path debería influir hacia DATA_ANALYSIS
        assert classification.task_type == TaskType.DATA_ANALYSIS

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, orchestrator, sample_requests):
        """Test procesamiento concurrente de múltiples solicitudes."""
        import asyncio
        
        # Procesar múltiples solicitudes concurrentemente
        tasks = [
            orchestrator.process(sample_requests['data_analysis']),
            orchestrator.process(sample_requests['text_analysis']),
            orchestrator.process(sample_requests['chat'])
        ]
        
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 3
        assert all(isinstance(r, AgentResponse) for r in responses)
        assert all(not r.metadata.get('error', False) for r in responses)

    @pytest.mark.asyncio
    async def test_history_limit(self, orchestrator):
        """Test límite del historial de interacciones."""
        # Procesar más de 50 interacciones para probar el límite
        for i in range(55):
            await orchestrator.process({'message': f'test message {i}'})
        
        # El historial debería estar limitado a 50
        assert len(orchestrator.interaction_history) == 50
        
        # Las interacciones más recientes deberían estar presentes
        last_interaction = orchestrator.interaction_history[-1]
        assert 'test message 54' in last_interaction['message']
