"""
Modelos Pydantic para la API REST.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


# ===============================
# MODELOS BASE
# ===============================

class ResponseBase(BaseModel):
    """Modelo base para respuestas."""
    timestamp: datetime = Field(default_factory=datetime.now)


class SuccessResponse(ResponseBase):
    """Respuesta de éxito genérica."""
    success: bool = True
    message: str


class ErrorResponse(ResponseBase):
    """Respuesta de error."""
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None


# ===============================
# MODELOS DE AGENTES
# ===============================

class AgentMetrics(BaseModel):
    """Métricas de un agente."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0
    average_response_time: float = 0.0
    current_load: int = 0
    availability: float = 1.0


class AgentInfo(BaseModel):
    """Información de un agente."""
    id: str
    type: str
    status: str
    description: Optional[str] = None
    metrics: Optional[AgentMetrics] = None


class AgentsListResponse(ResponseBase):
    """Respuesta con lista de agentes."""
    agents: List[AgentInfo]
    total: int


class AgentTaskRequest(BaseModel):
    """Solicitud de tarea para un agente."""
    task_input: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None


class AgentExecutionResponse(ResponseBase):
    """Respuesta de ejecución de agente."""
    agent_id: str
    success: bool
    result: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


# ===============================
# MODELOS DE WORKFLOWS
# ===============================

class WorkflowStepInfo(BaseModel):
    """Información de un paso de workflow."""
    id: str
    agent_type: str
    task_config: Dict[str, Any]
    dependencies: List[str] = []
    max_retries: int = 3
    timeout_seconds: int = 300


class WorkflowInfo(BaseModel):
    """Información básica de un workflow."""
    id: str
    name: str
    description: str
    steps_count: int
    max_parallel: int = 3
    timeout_minutes: int = 30
    metadata: Optional[Dict[str, Any]] = None


class WorkflowsListResponse(ResponseBase):
    """Respuesta con lista de workflows."""
    workflows: List[WorkflowInfo]
    total: int


class WorkflowDetailResponse(ResponseBase):
    """Respuesta con detalles de un workflow."""
    id: str
    name: str
    description: str
    steps: List[WorkflowStepInfo]
    max_parallel: int
    timeout_minutes: int
    metadata: Optional[Dict[str, Any]] = None


class WorkflowExecutionRequest(BaseModel):
    """Solicitud de ejecución de workflow."""
    parameters: Dict[str, Any] = {}
    options: Optional[Dict[str, Any]] = None
    priority: int = 0


class WorkflowExecutionResponse(ResponseBase):
    """Respuesta de ejecución de workflow."""
    workflow_id: str
    execution_id: str
    status: str
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    errors: List[str] = []


class WorkflowExecutionStatusResponse(ResponseBase):
    """Respuesta con estado de ejecución de workflow."""
    execution_id: str
    workflow_id: str
    status: str
    progress: float
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    current_step: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    errors: List[str] = []


# ===============================
# MODELOS DEL ORQUESTRADOR
# ===============================

class OrchestratorStatusResponse(ResponseBase):
    """Respuesta con estado del orquestrador."""
    status: str
    system_metrics: Dict[str, Any]
    agent_metrics: Dict[str, Any]
    active_workflows: Dict[str, Any]
    configuration: Dict[str, Any]


class OrchestratorMetricsResponse(ResponseBase):
    """Respuesta con métricas del orquestrador."""
    system_metrics: Dict[str, Any]
    agent_metrics: Dict[str, Any]
    workflow_history: List[Dict[str, Any]]


# ===============================
# MODELOS PARA CHAT Y CONSULTAS
# ===============================

class ChatMessage(BaseModel):
    """Mensaje de chat."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Solicitud de chat."""
    message: str
    agent_type: str = "qa"
    context: Optional[str] = None
    conversation_history: List[ChatMessage] = []
    options: Optional[Dict[str, Any]] = None


class ChatResponse(ResponseBase):
    """Respuesta de chat."""
    message: str
    agent_type: str
    success: bool
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ===============================
# MODELOS PARA ANÁLISIS DE DATOS
# ===============================

class DataAnalysisRequest(BaseModel):
    """Solicitud de análisis de datos."""
    file_path: Optional[str] = None
    data_url: Optional[str] = None
    data_content: Optional[str] = None
    operations: List[str] = ["describe"]
    options: Optional[Dict[str, Any]] = None


class DataAnalysisResponse(ResponseBase):
    """Respuesta de análisis de datos."""
    file_path: Optional[str]
    operations: List[str]
    success: bool
    results: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ===============================
# MODELOS PARA PROCESAMIENTO DE TEXTO
# ===============================

class TextProcessingRequest(BaseModel):
    """Solicitud de procesamiento de texto."""
    text: str
    operation: str = "summarize"  # summarize, sentiment, keywords, entities
    language: str = "es"
    options: Optional[Dict[str, Any]] = None


class TextProcessingResponse(ResponseBase):
    """Respuesta de procesamiento de texto."""
    text_preview: str
    operation: str
    language: str
    success: bool
    result: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ===============================
# MODELOS PARA CONFIGURACIÓN
# ===============================

class APIConfiguration(BaseModel):
    """Configuración de la API."""
    title: str
    version: str
    debug: bool = False
    cors_origins: List[str] = ["*"]
    rate_limit: Optional[Dict[str, Any]] = None


class ConfigurationResponse(ResponseBase):
    """Respuesta con configuración."""
    api: APIConfiguration
    agents: Dict[str, Any]
    orchestrator: Dict[str, Any]
