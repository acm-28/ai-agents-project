"""Tests para AdvancedOrchestrator."""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
from datetime import datetime, timedelta

from ai_agents.agents.orchestration.advanced_orchestrator import (
    AdvancedOrchestrator,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStep,
    WorkflowStatus,
    StepStatus,
    AgentMetrics
)
from ai_agents.agents.orchestration.agent_orchestrator import TaskType
from ai_agents.core.types import AgentResponse


class TestAdvancedOrchestrator:
    """Test suite para AdvancedOrchestrator."""

    @pytest_asyncio.fixture
    async def orchestrator(self):
        """Fixture para crear un AdvancedOrchestrator con agentes mock."""
        # Crear respuesta exitosa por defecto
        default_success_response = AgentResponse(
            content="Success",
            error=None
        )
        
        # Crear agentes mock
        pandas_agent = Mock()
        pandas_agent.agent_id = "pandas_agent"
        pandas_agent.process_request = AsyncMock(return_value=default_success_response)
        pandas_agent.is_ready = Mock(return_value=True)
        pandas_agent.state = Mock()
        pandas_agent.state.value = "READY"
        
        sophisticated_agent = Mock()
        sophisticated_agent.agent_id = "sophisticated_agent"
        sophisticated_agent.process_request = AsyncMock(return_value=default_success_response)
        sophisticated_agent.is_ready = Mock(return_value=True)
        sophisticated_agent.state = Mock()
        sophisticated_agent.state.value = "READY"
        
        langchain_agent = Mock()
        langchain_agent.agent_id = "langchain_agent"
        langchain_agent.process_request = AsyncMock()
        langchain_agent.is_ready = Mock(return_value=True)
        langchain_agent.state = Mock()
        langchain_agent.state.value = "READY"
        
        # Crear orchestrator con parámetros de configuración y sin inicialización automática
        orchestrator = AdvancedOrchestrator(max_parallel_executions=2, auto_initialize_agents=False)
        
        # Registrar agentes manualmente (3 agentes consolidados)
        orchestrator.register_agent("pandas_agent", pandas_agent)
        orchestrator.register_agent("sophisticated_agent", sophisticated_agent)
        orchestrator.register_agent("langchain_agent", langchain_agent)
        
        # Inicializar el orchestrator
        await orchestrator.initialize()
        
        yield orchestrator

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
                    task_config={"task": "Analizar datos de ventas", "data": "sample_data"},
                    dependencies=[],
                    timeout_seconds=1  # Timeout muy corto para tests
                ),
                WorkflowStep(
                    step_id="step2", 
                    agent_type="sophisticated_agent",
                    task_config={"task": "Crear reporte basado en análisis", "analysis": "{{step1.result}}"},
                    dependencies=["step1"],
                    timeout_seconds=1  # Timeout muy corto para tests
                )
            ],
            metadata={"version": "1.0"}
        )

    def test_orchestrator_initialization(self, orchestrator):
        """Testa la inicialización del orquestrador."""
        assert orchestrator.max_parallel_executions == 2
        assert len(orchestrator.specialized_agents) >= 3  # Al menos los 3 agentes consolidados registrados
        assert orchestrator.semaphore._value == 2
        assert len(orchestrator.workflow_definitions) >= 1  # Al menos los predefinidos

    def test_workflow_registration(self, orchestrator, sample_workflow):
        """Testa el registro de workflows."""
        orchestrator.register_workflow(sample_workflow)
        assert sample_workflow.workflow_id in orchestrator.workflow_definitions
        assert orchestrator.workflow_definitions[sample_workflow.workflow_id] == sample_workflow

    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self, orchestrator, sample_workflow):
        """Testa la ejecución básica de un workflow."""
        # Mock responses
        orchestrator.specialized_agents["pandas_agent"].process_request.return_value = AgentResponse(
            content="Análisis completado",
            metadata={"processed_rows": 100}
        )
        
        orchestrator.specialized_agents["sophisticated_agent"].process_request.return_value = AgentResponse(
            content="Reporte generado",
            metadata={"report_pages": 5}
        )
        
        # Registrar y ejecutar workflow
        orchestrator.register_workflow(sample_workflow)
        execution = await orchestrator.execute_workflow(
            sample_workflow.workflow_id,
            {"data": "test_data"}
        )
        
        assert execution.status == WorkflowStatus.COMPLETED
        assert len(execution.results) == 2
        assert "step1" in execution.results
        assert "step2" in execution.results

    @pytest.mark.asyncio 
    async def test_workflow_with_failure(self, orchestrator, sample_workflow):
        """Testa manejo de errores en workflow."""
        # Configurar el mock para que falle con excepción
        orchestrator.specialized_agents["pandas_agent"].process_request.side_effect = Exception("Datos inválidos")
        
        orchestrator.register_workflow(sample_workflow)
        execution = await orchestrator.execute_workflow(
            sample_workflow.workflow_id,
            {"data": "invalid_data"}
        )
        
        assert execution.status == WorkflowStatus.FAILED
        # Verificar que el primer paso falló
        step1 = next(s for s in execution.workflow_def.steps if s.step_id == "step1")
        assert step1.status == StepStatus.FAILED
        # El segundo paso no debe tener resultado porque no se ejecutó
        assert "step2" not in execution.results

    @pytest.mark.asyncio
    async def test_workflow_dependency_resolution(self, orchestrator):
        """Testa resolución de dependencias en workflow."""
        workflow = WorkflowDefinition(
            workflow_id="dependency_test",
            name="Dependency Test",
            description="Prueba de dependencias",
            steps=[
                WorkflowStep(
                    step_id="stepA",
                    agent_type="pandas_agent",
                    task_config={"task": "Procesar datos", "data": "raw_data"},
                    dependencies=[]
                ),
                WorkflowStep(
                    step_id="stepB",
                    agent_type="sophisticated_agent",
                    task_config={"task": "Análisis intermedio", "processed_data": "{{stepA.result}}"},
                    dependencies=["stepA"]
                ),
                WorkflowStep(
                    step_id="stepC",
                    agent_type="langchain_agent",
                    task_config={
                        "task": "Reporte final",
                        "data": "{{stepA.result}}",
                        "analysis": "{{stepB.result}}"
                    },
                    dependencies=["stepA", "stepB"]
                )
            ]
        )
        
        # Mock responses
        for agent in orchestrator.specialized_agents.values():
            agent.process_request.return_value = AgentResponse(
                content="Success"
            )
        
        orchestrator.register_workflow(workflow)
        execution = await orchestrator.execute_workflow(
            workflow.workflow_id,
            {"data": "test_data"}
        )
        
        assert execution.status == WorkflowStatus.COMPLETED
        
        # Verificar orden de ejecución usando los pasos del workflow
        stepA = next(s for s in execution.workflow_def.steps if s.step_id == "stepA")
        stepB = next(s for s in execution.workflow_def.steps if s.step_id == "stepB")
        stepC = next(s for s in execution.workflow_def.steps if s.step_id == "stepC")
        
        assert stepA.start_time < stepB.start_time
        assert stepB.start_time < stepC.start_time

    @pytest.mark.asyncio
    async def test_parallel_workflow_execution(self, orchestrator):
        """Testa ejecución paralela de workflows."""
        workflow1 = WorkflowDefinition(
            workflow_id="parallel1",
            name="Parallel 1",
            description="Workflow paralelo 1",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    agent_type="pandas_agent",
                    task_config={"task": "Tarea 1"},
                    dependencies=[]
                )
            ]
        )
        
        workflow2 = WorkflowDefinition(
            workflow_id="parallel2",
            name="Parallel 2", 
            description="Workflow paralelo 2",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    agent_type="sophisticated_agent",
                    task_config={"task": "Tarea 2"},
                    dependencies=[]
                )
            ]
        )
        
        # Mock responses con delay
        async def mock_response_delay(*args, **kwargs):
            await asyncio.sleep(0.1)
            return AgentResponse(content="Success")
        
        for agent in orchestrator.specialized_agents.values():
            agent.process_request = AsyncMock(side_effect=mock_response_delay)
        
        orchestrator.register_workflow(workflow1)
        orchestrator.register_workflow(workflow2)
        
        # Ejecutar en paralelo
        start_time = datetime.now()
        results = await asyncio.gather(
            orchestrator.execute_workflow("parallel1", {}),
            orchestrator.execute_workflow("parallel2", {})
        )
        end_time = datetime.now()
        
        # Debería tomar aproximadamente 0.1s (paralelo) vs 0.2s (secuencial)
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 0.15  # Margen para overhead
        
        assert all(result.status == WorkflowStatus.COMPLETED for result in results)

    @pytest.mark.asyncio
    async def test_concurrency_limits(self, orchestrator):
        """Testa límites de concurrencia."""
        # Crear 3 workflows simples (más que el límite de 2)
        workflows = []
        for i in range(3):
            workflow = WorkflowDefinition(
                workflow_id=f"concurrent_{i}",
                name=f"Concurrent {i}",
                description=f"Workflow concurrente {i}",
                steps=[
                    WorkflowStep(
                        step_id="step1",
                        agent_type="pandas_agent",
                        task_config={"task": f"Tarea {i}"},
                        dependencies=[]
                    )
                ]
            )
            workflows.append(workflow)
            orchestrator.register_workflow(workflow)
        
        # Mock response con delay significativo
        async def mock_slow_response():
            await asyncio.sleep(0.2)
            return AgentResponse(content="Success")
        
        orchestrator.specialized_agents["pandas_agent"].process_request = AsyncMock(
            side_effect=mock_slow_response
        )
        
        # Ejecutar todos los workflows
        start_time = datetime.now()
        results = await asyncio.gather(*[
            orchestrator.execute_workflow(f"concurrent_{i}", {})
            for i in range(3)
        ])
        end_time = datetime.now()
        
        # Con límite de 2, debería tomar aproximadamente 0.4s (2 x 0.2s)
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time >= 0.35  # Al menos dos tandas
        assert all(result.status == WorkflowStatus.COMPLETED for result in results)

    @pytest.mark.asyncio
    async def test_agent_metrics_collection(self, orchestrator):
        """Testa recolección de métricas de agentes."""
        # Simular algunas ejecuciones
        agent_name = "pandas_agent"
        await orchestrator._update_agent_metrics(
            agent_name, 
            success=True, 
            response_time=1.5
        )
        await orchestrator._update_agent_metrics(
            agent_name,
            success=False,
            response_time=0.8
        )
        await orchestrator._update_agent_metrics(
            agent_name,
            success=True, 
            response_time=2.1
        )
        
        metrics = orchestrator.agent_metrics[agent_name]
        
        assert metrics.total_requests == 3
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 1
        assert metrics.error_rate == pytest.approx(0.333, rel=1e-2)  # 1/3 = 0.333
        assert metrics.average_response_time == pytest.approx(1.467, rel=1e-2)  # (1.5+0.8+2.1)/3

    def test_load_balancing(self, orchestrator):
        """Testa balanceeo de carga entre agentes."""
        # Configurar métricas diferentes para agentes
        orchestrator.agent_metrics["agent1"] = AgentMetrics(
            agent_name="agent1"
        )
        orchestrator.agent_metrics["agent1"].current_load = 3
        orchestrator.agent_metrics["agent1"].average_response_time = 2.0
        
        orchestrator.agent_metrics["agent2"] = AgentMetrics(
            agent_name="agent2"
        )
        orchestrator.agent_metrics["agent2"].current_load = 1
        orchestrator.agent_metrics["agent2"].average_response_time = 1.0
        
        # Debería elegir agent2 (menor carga)
        selected = orchestrator._select_best_agent_from_list(["agent1", "agent2"])
        assert selected == "agent2"

    @pytest.mark.asyncio
    async def test_predefined_data_analysis_workflow(self, orchestrator):
        """Testa workflow predefinido de análisis de datos."""
        # Mock del pandas agent
        orchestrator.specialized_agents["pandas_agent"].process_request.return_value = AgentResponse(
            content="Análisis completado: Ventas promedio = $1000",
            metadata={"statistics": {"mean": 1000, "count": 100}}
        )
        
        # Mock del sophisticated agent
        orchestrator.specialized_agents["sophisticated_agent"].process_request.return_value = AgentResponse(
            content="Reporte generado con visualizaciones",
            metadata={"charts_created": 3}
        )
        
        execution = await orchestrator.execute_workflow(
            "data_analysis_complete",
            {"dataset": "sales_data.csv", "analysis_type": "descriptive"}
        )
        
        assert execution.status == WorkflowStatus.COMPLETED
        assert len(execution.results) == 4  # load_data + basic_analysis + text_summary + qa_validation
        assert "load_data" in execution.results
        assert "basic_analysis" in execution.results
        assert "text_summary" in execution.results
        assert "qa_validation" in execution.results

    @pytest.mark.asyncio
    async def test_error_recovery_with_retries(self, orchestrator, sample_workflow):
        """Testa recuperación de errores con reintentos."""
        # Configurar para fallar las primeras 2 veces, luego éxito
        call_count = 0
        
        async def mock_flaky_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return AgentResponse(
                    content="Error temporal",
                    error="Conexión perdida"
                )
            return AgentResponse(content="Éxito")
        
        orchestrator.specialized_agents["pandas_agent"].process_request = AsyncMock(
            side_effect=mock_flaky_response
        )
        orchestrator.specialized_agents["sophisticated_agent"].process_request.return_value = AgentResponse(
            content="Reporte generado"
        )
        
        orchestrator.register_workflow(sample_workflow)
        execution = await orchestrator.execute_workflow(
            sample_workflow.workflow_id,
            {"data": "test_data"}
        )
        
        # Debería completarse después de reintentos
        assert execution.status == WorkflowStatus.COMPLETED
        assert call_count == 3  # 2 fallos + 1 éxito

    def test_workflow_hooks(self, orchestrator, sample_workflow):
        """Testa hooks de workflow."""
        hook_calls = []
        
        def workflow_started(workflow_id, execution_id):
            hook_calls.append(f"started_{workflow_id}")
        
        def step_completed(workflow_id, execution_id, step_id, result):
            hook_calls.append(f"completed_{step_id}")
        
        def workflow_completed(workflow_id, execution_id, result):
            hook_calls.append(f"finished_{workflow_id}")
        
        orchestrator.add_hook("workflow_started", workflow_started)
        orchestrator.add_hook("step_completed", step_completed)
        orchestrator.add_hook("workflow_completed", workflow_completed)
        
        # Los hooks se verificarían en una ejecución real
        assert len(orchestrator.hooks["workflow_started"]) == 1
        assert len(orchestrator.hooks["step_completed"]) == 1
        assert len(orchestrator.hooks["workflow_completed"]) == 1

    @pytest.mark.asyncio
    async def test_system_metrics(self, orchestrator):
        """Testa métricas del sistema."""
        # Simular algunas ejecuciones
        await orchestrator._update_agent_metrics("agent1", True, 1.0)
        await orchestrator._update_agent_metrics("agent2", False, 2.0)
        
        metrics = orchestrator.get_system_metrics()
        
        assert "total_workflows" in metrics
        assert "active_workflows" in metrics
        assert "uptime_seconds" in metrics
        assert "agent_metrics" in metrics

    @pytest.mark.asyncio
    async def test_workflow_cancellation(self, orchestrator):
        """Testa cancelación de workflows."""
        workflow = WorkflowDefinition(
            workflow_id="cancellable",
            name="Cancellable Workflow",
            description="Workflow que puede ser cancelado",
            steps=[
                WorkflowStep(
                    step_id="long_step",
                    agent_type="pandas_agent",
                    task_config={"task": "Tarea larga"},
                    dependencies=[]
                )
            ]
        )
        
        # Mock response que toma tiempo
        async def mock_long_response(*args, **kwargs):
            await asyncio.sleep(1.0)
            return AgentResponse(content="Success")
        
        orchestrator.specialized_agents["pandas_agent"].process_request = AsyncMock(
            side_effect=mock_long_response
        )
        
        orchestrator.register_workflow(workflow)
        
        # Iniciar ejecución y cancelar rápidamente
        execution_task = asyncio.create_task(
            orchestrator.execute_workflow("cancellable", {})
        )
        
        await asyncio.sleep(0.1)  # Permitir que inicie
        execution_task.cancel()
        
        with pytest.raises(asyncio.CancelledError):
            await execution_task

    def test_input_template_resolution(self, orchestrator):
        """Testa resolución de templates en inputs."""
        previous_results = {
            "step1": AgentResponse(
                content="Resultado del paso 1",
                metadata={"count": 42}
            )
        }
        
        step_inputs = {
            "data": "{{step1.result}}",
            "count": "{{step1.metadata.count}}",
            "static": "valor_fijo"
        }
        
        resolved = orchestrator._resolve_step_inputs(step_inputs, previous_results)
        
        assert resolved["data"] == "Resultado del paso 1"
        assert resolved["count"] == 42
        assert resolved["static"] == "valor_fijo"
