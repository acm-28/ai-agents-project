"""
API REST principal para el framework AI Agents.
Endpoints para gestión de agentes, workflows y orquestación.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from ai_agents.agents import (
    PandasAgent,
    SophisticatedAgent,
    MemoryQAAgent,
    LangChainChatAgent,
    LLMChatAgent,
    AdvancedOrchestrator
)
from ai_agents.agents.orchestration.advanced_orchestrator import (
    WorkflowDefinition,
    WorkflowStep,
    WorkflowStatus
)
from ai_agents.config.settings import settings
from .models import *

# Logger
logger = logging.getLogger(__name__)

# Instancia global del orquestrador
orchestrator: Optional[AdvancedOrchestrator] = None


async def get_orchestrator() -> AdvancedOrchestrator:
    """Obtener instancia del orquestrador."""
    global orchestrator
    
    if orchestrator is None:
        logger.info("Inicializando orquestrador para API...")
        
        # Crear agentes especializados
        pandas_agent = PandasAgent()
        sophisticated_agent = SophisticatedAgent()
        qa_agent = MemoryQAAgent()
        langchain_agent = LangChainChatAgent()
        llm_agent = LLMChatAgent()
        
        # Crear orquestrador
        orchestrator = AdvancedOrchestrator(agent_id="api_orchestrator")
        
        # Configurar agentes
        orchestrator.specialized_agents = {
            "pandas_agent": pandas_agent,
            "sophisticated_agent": sophisticated_agent,
            "qa_agent": qa_agent,
            "langchain_agent": langchain_agent,
            "llm_agent": llm_agent
        }
        
        # Inicializar
        await orchestrator.initialize()
        
        logger.info("Orquestrador inicializado correctamente")
    
    return orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación."""
    # Startup
    logger.info("🚀 Iniciando API de AI Agents...")
    await get_orchestrator()
    logger.info("✅ API inicializada correctamente")
    
    yield
    
    # Shutdown
    logger.info("🛑 Cerrando API de AI Agents...")
    # Aquí podrías agregar limpieza si fuera necesario
    logger.info("✅ API cerrada correctamente")


# Crear aplicación FastAPI
app = FastAPI(
    title="AI Agents Framework API",
    description="API REST para gestión de agentes y workflows del framework AI Agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "name": "AI Agents Framework API",
        "version": "1.0.0",
        "description": "API REST para gestión de agentes y workflows",
        "documentation": "/docs",
        "health": "/health",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Check de salud de la API."""
    try:
        orch = await get_orchestrator()
        
        # Verificar estado de agentes
        agent_status = {}
        for agent_id, agent in orch.specialized_agents.items():
            agent_status[agent_id] = {
                "status": "healthy",
                "type": agent.__class__.__name__
            }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "orchestrator": "running",
            "agents": agent_status,
            "active_workflows": len(orch.active_workflows),
            "total_workflows_defined": len(orch.workflow_definitions)
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# ===============================
# ENDPOINTS DE AGENTES
# ===============================

@app.get("/agents", response_model=AgentsListResponse)
async def list_agents():
    """Listar todos los agentes disponibles."""
    orch = await get_orchestrator()
    
    agents = []
    for agent_id, agent in orch.specialized_agents.items():
        # Obtener métricas si están disponibles
        metrics = orch.agent_metrics.get(agent_id)
        
        agent_info = AgentInfo(
            id=agent_id,
            type=agent.__class__.__name__,
            status="ready",
            description=getattr(agent, '__doc__', '').split('\n')[0] if hasattr(agent, '__doc__') else "",
            metrics=AgentMetrics(
                total_requests=metrics.total_requests if metrics else 0,
                successful_requests=metrics.successful_requests if metrics else 0,
                failed_requests=metrics.failed_requests if metrics else 0,
                error_rate=metrics.error_rate if metrics else 0.0,
                average_response_time=metrics.average_response_time if metrics else 0.0,
                current_load=metrics.current_load if metrics else 0,
                availability=metrics.availability if metrics else 1.0
            ) if metrics else None
        )
        agents.append(agent_info)
    
    return AgentsListResponse(
        agents=agents,
        total=len(agents),
        timestamp=datetime.now()
    )


@app.post("/agents/{agent_id}/execute", response_model=AgentExecutionResponse)
async def execute_agent_task(
    agent_id: str,
    request: AgentTaskRequest,
    background_tasks: BackgroundTasks
):
    """Ejecutar una tarea con un agente específico."""
    orch = await get_orchestrator()
    
    if agent_id not in orch.specialized_agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found"
        )
    
    agent = orch.specialized_agents[agent_id]
    
    try:
        # Ejecutar tarea
        result = await agent.process_request(request.task_input)
        
        return AgentExecutionResponse(
            agent_id=agent_id,
            success=result.success,
            result=result.content,
            metadata=result.metadata,
            error=result.error,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error executing task on agent {agent_id}: {e}")
        return AgentExecutionResponse(
            agent_id=agent_id,
            success=False,
            result=None,
            error=str(e),
            timestamp=datetime.now()
        )


# ===============================
# ENDPOINTS DE WORKFLOWS
# ===============================

@app.get("/workflows", response_model=WorkflowsListResponse)
async def list_workflows():
    """Listar todos los workflows disponibles."""
    orch = await get_orchestrator()
    
    workflows = []
    for workflow_id, workflow_def in orch.workflow_definitions.items():
        workflow_info = WorkflowInfo(
            id=workflow_id,
            name=workflow_def.name,
            description=workflow_def.description,
            steps_count=len(workflow_def.steps),
            max_parallel=workflow_def.max_parallel,
            timeout_minutes=workflow_def.timeout_minutes,
            metadata=workflow_def.metadata
        )
        workflows.append(workflow_info)
    
    return WorkflowsListResponse(
        workflows=workflows,
        total=len(workflows),
        timestamp=datetime.now()
    )


@app.get("/workflows/{workflow_id}", response_model=WorkflowDetailResponse)
async def get_workflow_detail(workflow_id: str):
    """Obtener detalles de un workflow específico."""
    orch = await get_orchestrator()
    
    if workflow_id not in orch.workflow_definitions:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' not found"
        )
    
    workflow_def = orch.workflow_definitions[workflow_id]
    
    steps = []
    for step in workflow_def.steps:
        step_info = WorkflowStepInfo(
            id=step.step_id,
            agent_type=step.agent_type,
            task_config=step.task_config,
            dependencies=step.dependencies,
            max_retries=step.max_retries,
            timeout_seconds=step.timeout_seconds
        )
        steps.append(step_info)
    
    return WorkflowDetailResponse(
        id=workflow_id,
        name=workflow_def.name,
        description=workflow_def.description,
        steps=steps,
        max_parallel=workflow_def.max_parallel,
        timeout_minutes=workflow_def.timeout_minutes,
        metadata=workflow_def.metadata,
        timestamp=datetime.now()
    )


@app.post("/workflows/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    workflow_id: str,
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks
):
    """Ejecutar un workflow."""
    orch = await get_orchestrator()
    
    if workflow_id not in orch.workflow_definitions:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' not found"
        )
    
    try:
        # Ejecutar workflow
        execution = await orch.execute_workflow(workflow_id, request.parameters)
        
        return WorkflowExecutionResponse(
            workflow_id=workflow_id,
            execution_id=execution.execution_id,
            status=execution.status.value,
            progress=execution.progress,
            start_time=execution.start_time,
            end_time=execution.end_time,
            results={k: v.content for k, v in execution.results.items()},
            errors=execution.errors,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing workflow: {str(e)}"
        )


@app.get("/workflows/executions/{execution_id}", response_model=WorkflowExecutionStatusResponse)
async def get_workflow_execution_status(execution_id: str):
    """Obtener estado de una ejecución de workflow."""
    orch = await get_orchestrator()
    
    # Buscar en workflows activos
    execution = orch.active_workflows.get(execution_id)
    
    if not execution:
        # Buscar en historial
        execution = next(
            (exec for exec in orch.workflow_history if exec.execution_id == execution_id),
            None
        )
    
    if not execution:
        raise HTTPException(
            status_code=404,
            detail=f"Execution '{execution_id}' not found"
        )
    
    return WorkflowExecutionStatusResponse(
        execution_id=execution_id,
        workflow_id=execution.workflow_def.workflow_id,
        status=execution.status.value,
        progress=execution.progress,
        start_time=execution.start_time,
        end_time=execution.end_time,
        current_step=execution.current_step,
        results={k: v.content for k, v in execution.results.items()},
        errors=execution.errors,
        timestamp=datetime.now()
    )


# ===============================
# ENDPOINTS DEL ORQUESTRADOR
# ===============================

@app.get("/orchestrator/status", response_model=OrchestratorStatusResponse)
async def get_orchestrator_status():
    """Obtener estado del orquestrador."""
    orch = await get_orchestrator()
    
    # Métricas de agentes
    agent_metrics = {}
    for agent_id, metrics in orch.agent_metrics.items():
        agent_metrics[agent_id] = {
            "total_requests": metrics.total_requests,
            "successful_requests": metrics.successful_requests,
            "failed_requests": metrics.failed_requests,
            "error_rate": metrics.error_rate,
            "average_response_time": metrics.average_response_time,
            "current_load": metrics.current_load,
            "availability": metrics.availability
        }
    
    # Workflows activos
    active_workflows = {}
    for execution_id, execution in orch.active_workflows.items():
        active_workflows[execution_id] = {
            "workflow_id": execution.workflow_def.workflow_id,
            "status": execution.status.value,
            "progress": execution.progress,
            "start_time": execution.start_time
        }
    
    return OrchestratorStatusResponse(
        status="running",
        system_metrics=orch.system_metrics,
        agent_metrics=agent_metrics,
        active_workflows=active_workflows,
        configuration={
            "max_concurrent_workflows": orch.max_concurrent_workflows,
            "load_balancing_enabled": orch.load_balancing_enabled,
            "auto_scaling_enabled": orch.auto_scaling_enabled
        },
        timestamp=datetime.now()
    )


@app.get("/orchestrator/metrics", response_model=OrchestratorMetricsResponse)
async def get_orchestrator_metrics():
    """Obtener métricas detalladas del orquestrador."""
    orch = await get_orchestrator()
    
    # Métricas de agentes
    agent_metrics = {}
    for agent_id, metrics in orch.agent_metrics.items():
        agent_metrics[agent_id] = {
            "agent_name": metrics.agent_name,
            "total_requests": metrics.total_requests,
            "successful_requests": metrics.successful_requests,
            "failed_requests": metrics.failed_requests,
            "error_rate": metrics.error_rate,
            "average_response_time": metrics.average_response_time,
            "last_request_time": metrics.last_request_time,
            "availability": metrics.availability,
            "current_load": metrics.current_load,
            "max_concurrent": metrics.max_concurrent
        }
    
    # Historial de workflows (últimos 50)
    workflow_history = []
    for exec in orch.workflow_history[-50:]:
        workflow_history.append({
            "execution_id": exec.execution_id,
            "workflow_id": exec.workflow_def.workflow_id,
            "status": exec.status.value,
            "start_time": exec.start_time,
            "end_time": exec.end_time,
            "progress": exec.progress
        })
    
    return OrchestratorMetricsResponse(
        system_metrics=orch.system_metrics,
        agent_metrics=agent_metrics,
        workflow_history=workflow_history,
        timestamp=datetime.now()
    )


# ===============================
# REGISTRAR RUTAS ADICIONALES
# ===============================

# Importar y registrar rutas adicionales
try:
    from .routes import all_routers
    for router in all_routers:
        app.include_router(router)
except ImportError:
    logger.warning("No se pudieron cargar las rutas adicionales")


# ===============================
# ENDPOINT PARA EJECUTAR SERVER
# ===============================

def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Ejecutar servidor API."""
    uvicorn.run(
        "ai_agents.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_api()
