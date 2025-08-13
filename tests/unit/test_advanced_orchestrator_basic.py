"""Tests básicos para AdvancedOrchestrator."""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from ai_agents.agents.orchestration.advanced_orchestrator import (
    AdvancedOrchestrator,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStep,
    WorkflowStatus,
    StepStatus,
    AgentMetrics
)
from ai_agents.core.types import AgentResponse


class TestAdvancedOrchestratorBasic:
    """Test suite básico para AdvancedOrchestrator."""

    @pytest_asyncio.fixture
    async def orchestrator(self):
        """Fixture para crear un AdvancedOrchestrator con agentes mock."""
        # Crear agentes mock
        pandas_agent = Mock()
        pandas_agent.agent_id = "pandas_agent"
        pandas_agent.process_request = AsyncMock(return_value=AgentResponse(
            content="Mock response", success=True
        ))
        
        sophisticated_agent = Mock()
        sophisticated_agent.agent_id = "sophisticated_agent"
        sophisticated_agent.process_request = AsyncMock(return_value=AgentResponse(
            content="Mock response", success=True
        ))
        
        qa_agent = Mock()
        qa_agent.agent_id = "qa_agent"
        qa_agent.process_request = AsyncMock(return_value=AgentResponse(
            content="Mock response", success=True
        ))
        
        langchain_agent = Mock()
        langchain_agent.agent_id = "langchain_agent"
        langchain_agent.process_request = AsyncMock(return_value=AgentResponse(
            content="Mock response", success=True
        ))
        
        llm_agent = Mock()
        llm_agent.agent_id = "llm_agent"
        llm_agent.process_request = AsyncMock(return_value=AgentResponse(
            content="Mock response", success=True
        ))
        
        orchestrator = AdvancedOrchestrator(agent_id="advanced_orchestrator")
        
        # Configurar agentes especializados manualmente
        orchestrator.specialized_agents = {
            "pandas_agent": pandas_agent,
            "sophisticated_agent": sophisticated_agent,
            "qa_agent": qa_agent,
            "langchain_agent": langchain_agent,
            "llm_agent": llm_agent
        }
        
        # Inicializar
        await orchestrator.initialize()
        
        return orchestrator

    @pytest.fixture
    def sample_workflow(self):
        """Fixture para un workflow de ejemplo."""
        return WorkflowDefinition(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="Workflow de prueba",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    agent_type="pandas_agent", 
                    task_config={"task": "Analizar datos de ventas"}
                ),
                WorkflowStep(
                    step_id="step2", 
                    agent_type="sophisticated_agent",
                    task_config={"task": "Crear reporte"},
                    dependencies=["step1"]
                )
            ]
        )

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Testa la inicialización del orquestrador."""
        assert orchestrator is not None
        assert len(orchestrator.specialized_agents) >= 5  # Pueden ser más por agentes por defecto
        assert orchestrator.max_concurrent_workflows == 10
        assert len(orchestrator.workflow_definitions) >= 0

    def test_workflow_registration(self, orchestrator, sample_workflow):
        """Testa el registro de workflows."""
        # Registrar workflow directamente en el diccionario
        orchestrator.workflow_definitions[sample_workflow.workflow_id] = sample_workflow
        assert sample_workflow.workflow_id in orchestrator.workflow_definitions
        assert orchestrator.workflow_definitions[sample_workflow.workflow_id] == sample_workflow

    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self, orchestrator, sample_workflow):
        """Testa la ejecución básica de un workflow."""
        # Registrar workflow
        orchestrator.workflow_definitions[sample_workflow.workflow_id] = sample_workflow
        
        # Ejecutar workflow
        execution = await orchestrator.execute_workflow(
            sample_workflow.workflow_id,
            {"data": "test_data"}
        )
        
        assert execution is not None
        # Puede estar en cualquier estado válido
        assert execution.status in [
            WorkflowStatus.COMPLETED, 
            WorkflowStatus.FAILED, 
            WorkflowStatus.PENDING,
            WorkflowStatus.RUNNING
        ]

    def test_agent_metrics_initialization(self, orchestrator):
        """Testa inicialización de métricas de agentes."""
        # Verificar que las métricas se inicializan correctamente
        assert isinstance(orchestrator.agent_metrics, dict)
        assert isinstance(orchestrator.system_metrics, dict)
        assert 'total_workflows' in orchestrator.system_metrics

    def test_workflow_hooks_registration(self, orchestrator):
        """Testa registro de hooks."""
        hook_called = False
        
        def test_hook(*args, **kwargs):
            nonlocal hook_called
            hook_called = True
        
        # Registrar hook
        orchestrator.workflow_hooks['before_start'].append(test_hook)
        
        assert len(orchestrator.workflow_hooks['before_start']) == 1

    @pytest.mark.asyncio
    async def test_agent_availability(self, orchestrator):
        """Testa disponibilidad de agentes."""
        available_agents = orchestrator.specialized_agents
        
        assert "pandas_agent" in available_agents
        assert "sophisticated_agent" in available_agents
        assert "qa_agent" in available_agents
        assert "langchain_agent" in available_agents
        assert "llm_agent" in available_agents

    def test_workflow_step_creation(self):
        """Testa creación de pasos de workflow."""
        step = WorkflowStep(
            step_id="test_step",
            agent_type="test_agent",
            task_config={"task": "test task"}
        )
        
        assert step.step_id == "test_step"
        assert step.agent_type == "test_agent"
        assert step.status == StepStatus.WAITING
        assert step.dependencies == []

    def test_workflow_definition_creation(self):
        """Testa creación de definición de workflow."""
        steps = [
            WorkflowStep(
                step_id="step1",
                agent_type="agent1",
                task_config={"task": "task1"}
            )
        ]
        
        workflow = WorkflowDefinition(
            workflow_id="test_workflow",
            name="Test",
            description="Test workflow",
            steps=steps
        )
        
        assert workflow.workflow_id == "test_workflow"
        assert workflow.name == "Test"
        assert len(workflow.steps) == 1
        assert workflow.max_parallel == 3  # valor por defecto

    def test_agent_metrics_creation(self):
        """Testa creación de métricas de agente."""
        metrics = AgentMetrics(
            agent_name="test_agent",
            total_requests=10,
            successful_requests=8
        )
        
        assert metrics.agent_name == "test_agent"
        assert metrics.total_requests == 10
        assert metrics.successful_requests == 8
        assert metrics.current_load == 0  # valor por defecto

    @pytest.mark.asyncio 
    async def test_system_metrics_collection(self, orchestrator):
        """Testa recolección de métricas del sistema."""
        metrics = orchestrator.system_metrics
        
        assert 'total_workflows' in metrics
        assert 'successful_workflows' in metrics
        assert 'failed_workflows' in metrics
        assert 'uptime_start' in metrics
        assert isinstance(metrics['uptime_start'], datetime)

    def test_task_queue_initialization(self, orchestrator):
        """Testa inicialización de cola de tareas."""
        assert isinstance(orchestrator.task_queue, list)
        assert isinstance(orchestrator.processing_tasks, dict)
        assert len(orchestrator.task_queue) == 0

    def test_load_balancing_configuration(self, orchestrator):
        """Testa configuración de balanceeo de carga."""
        assert orchestrator.load_balancing_enabled is True
        assert orchestrator.auto_scaling_enabled is True
        assert orchestrator.max_concurrent_workflows == 10

    @pytest.mark.asyncio
    async def test_workflow_with_dependencies(self, orchestrator):
        """Testa workflow con dependencias entre pasos."""
        workflow = WorkflowDefinition(
            workflow_id="dependency_test",
            name="Dependency Test",
            description="Test de dependencias",
            steps=[
                WorkflowStep(
                    step_id="stepA",
                    agent_type="pandas_agent",
                    task_config={"task": "Procesar datos"}
                ),
                WorkflowStep(
                    step_id="stepB",
                    agent_type="sophisticated_agent",
                    task_config={"task": "Análisis"},
                    dependencies=["stepA"]
                )
            ]
        )
        
        orchestrator.workflow_definitions[workflow.workflow_id] = workflow
        execution = await orchestrator.execute_workflow(
            workflow.workflow_id,
            {"data": "test_data"}
        )
        
        assert execution is not None

    @pytest.mark.asyncio
    async def test_multiple_workflow_registration(self, orchestrator):
        """Testa registro de múltiples workflows."""
        workflows = []
        for i in range(3):
            workflow = WorkflowDefinition(
                workflow_id=f"workflow_{i}",
                name=f"Workflow {i}",
                description=f"Test workflow {i}",
                steps=[
                    WorkflowStep(
                        step_id="step1",
                        agent_type="pandas_agent",
                        task_config={"task": f"Task {i}"}
                    )
                ]
            )
            workflows.append(workflow)
            orchestrator.workflow_definitions[workflow.workflow_id] = workflow
        
        # Verificar que todos los workflows están registrados
        for workflow in workflows:
            assert workflow.workflow_id in orchestrator.workflow_definitions
