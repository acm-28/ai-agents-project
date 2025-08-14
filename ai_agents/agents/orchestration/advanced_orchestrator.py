"""
Sistema de orquestación avanzada para coordinación compleja de agentes.

Este módulo implementa capacidades avanzadas de orquestación incluyendo:
- Workflows multi-agente coordinados
- Pipelines de procesamiento complejos  
- Distribución de cargas y balanceador
- Monitoreo en tiempo real
- Recuperación automática de fallos
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

from ...core.base_agent import BaseAgent
from ...core.types import AgentResponse, AgentState
from ...utils.conversation_logger import conversation_logger

logger = logging.getLogger(__name__)


# Definiciones heredadas del AgentOrchestrator eliminado
class TaskType(Enum):
    """Tipos de tareas que puede manejar el orquestador."""
    CHAT = "chat"
    DATA_ANALYSIS = "data_analysis"
    QA_MEMORY = "qa_memory"
    TEXT_ANALYSIS = "text_analysis"
    COMPLEX_WORKFLOW = "complex_workflow"
    UNKNOWN = "unknown"


@dataclass
class TaskClassification:
    """Clasificación de una tarea."""
    task_type: TaskType
    confidence: float
    agent_name: str
    reasoning: str
    parameters: Dict[str, Any]


class WorkflowStatus(Enum):
    """Estados de un workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    """Estados de un paso del workflow."""
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Definición de un paso del workflow."""
    step_id: str
    agent_type: str
    task_config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    status: StepStatus = StepStatus.WAITING
    result: Optional[AgentResponse] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class WorkflowDefinition:
    """Definición completa de un workflow."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    max_parallel: int = 3
    timeout_minutes: int = 30
    auto_retry: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Estado de ejecución de un workflow."""
    execution_id: str
    workflow_def: WorkflowDefinition
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Dict[str, AgentResponse] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    progress: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Métricas de rendimiento de un agente."""
    agent_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    error_rate: float = 0.0
    availability: float = 1.0
    current_load: int = 0
    max_concurrent: int = 5


class AdvancedOrchestrator(BaseAgent):
    """
    Orquestador avanzado con capacidades de workflows complejos.
    
    Extiende AgentOrchestrator con:
    - Workflows multi-agente coordinados
    - Pipelines de procesamiento complejos
    - Balanceador de cargas inteligente
    - Monitoreo y métricas en tiempo real
    - Recuperación automática de fallos
    """
    
    def __init__(self, max_parallel_executions: int = 3, **kwargs):
        super().__init__(**kwargs)
        
        # Registro de agentes especializados (heredado del AgentOrchestrator básico)
        self.specialized_agents: Dict[str, BaseAgent] = {}
        
        # Configuración de paralelismo
        self.max_parallel_executions = max_parallel_executions
        self.semaphore = asyncio.Semaphore(max_parallel_executions)
        
        # Sistema de workflows
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.workflow_history: List[WorkflowExecution] = []
        
        # Métricas y monitoreo
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.system_metrics: Dict[str, Any] = {
            'total_workflows': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'average_workflow_time': 0.0,
            'peak_concurrent_workflows': 0,
            'uptime_start': datetime.now()
        }
        
        # Cola de tareas distribuidas
        self.task_queue: List[Dict[str, Any]] = []
        self.processing_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Configuración del balanceador
        self.load_balancing_enabled = True
        self.auto_scaling_enabled = True
        self.max_concurrent_workflows = 10
        
        # Hooks y callbacks
        self.workflow_hooks: Dict[str, List[Callable]] = {
            'before_start': [],
            'after_complete': [],
            'on_error': [],
            'on_step_complete': []
        }
    
    async def initialize(self) -> None:
        """Inicializa el orquestador avanzado."""
        await super().initialize()
        
        # Inicializar métricas para cada agente
        for agent_name in self.specialized_agents.keys():
            self.agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)
        
        # Cargar workflows predefinidos
        await self._load_predefined_workflows()
        
        # Iniciar tareas de fondo
        asyncio.create_task(self._workflow_monitor())
        asyncio.create_task(self._metrics_collector())
        asyncio.create_task(self._task_processor())
        
        logger.info("AdvancedOrchestrator inicializado correctamente")
    
    async def process(self, input_data: Union[str, Dict, Any]) -> AgentResponse:
        """
        Procesa entrada usando clasificación automática y derivación a agentes especializados.
        
        Args:
            input_data: Datos de entrada (string, dict o Message)
            
        Returns:
            AgentResponse: Respuesta del agente apropiado
        """
        try:
            # Convertir entrada a formato estándar
            if isinstance(input_data, str):
                task_input = {"task": input_data, "user_input": input_data}
            elif isinstance(input_data, dict):
                task_input = input_data
            else:
                task_input = {"task": str(input_data), "user_input": str(input_data)}
            
            # Clasificar automáticamente la tarea
            task_classification = await self._classify_task(task_input.get("task", ""))
            
            # Obtener agente apropiado
            if task_classification.task_type == TaskType.DATA_ANALYSIS:
                agent = self.specialized_agents.get("pandas_agent")
            elif task_classification.task_type == TaskType.TEXT_ANALYSIS:
                agent = self.specialized_agents.get("sophisticated_agent")
            elif task_classification.task_type == TaskType.COMPLEX_WORKFLOW:
                agent = self.specialized_agents.get("sophisticated_agent")
            else:  # Chat o otros
                agent = self.specialized_agents.get("langchain_agent")
            
            if not agent:
                # Fallback al primer agente disponible
                agent = next(iter(self.specialized_agents.values()))
            
            # Procesar con el agente seleccionado
            return await agent.process(task_input)
            
        except Exception as e:
            logger.error(f"Error en AdvancedOrchestrator.process: {e}")
            return AgentResponse(
                content=f"Error procesando solicitud: {str(e)}",
                agent_id=self.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _classify_task(self, task: str) -> 'TaskClassification':
        """Clasifica una tarea para determinar el agente apropiado."""
        task_lower = task.lower()
        
        # Patrones de clasificación básica
        if any(word in task_lower for word in ['analizar', 'datos', 'csv', 'excel', 'pandas']):
            return TaskClassification(
                task_type=TaskType.DATA_ANALYSIS,
                confidence=0.8,
                agent_name="pandas_agent",
                reasoning="Contiene palabras relacionadas con análisis de datos",
                parameters={}
            )
        elif any(word in task_lower for word in ['texto', 'documento', 'resumir', 'analizar texto']):
            return TaskClassification(
                task_type=TaskType.TEXT_ANALYSIS,
                confidence=0.8,
                agent_name="sophisticated_agent",
                reasoning="Contiene palabras relacionadas con procesamiento de texto",
                parameters={}
            )
        elif any(word in task_lower for word in ['workflow', 'complejo', 'múltiples pasos']):
            return TaskClassification(
                task_type=TaskType.COMPLEX_WORKFLOW,
                confidence=0.8,
                agent_name="sophisticated_agent",
                reasoning="Indica proceso complejo o workflow",
                parameters={}
            )
        else:
            return TaskClassification(
                task_type=TaskType.CHAT,
                confidence=0.6,
                agent_name="langchain_agent",
                reasoning="Chat general o consulta",
                parameters={}
            )
    
    async def _load_predefined_workflows(self) -> None:
        """Carga workflows predefinidos."""
        # Workflow de análisis completo de datos
        data_analysis_workflow = WorkflowDefinition(
            workflow_id="data_analysis_complete",
            name="Análisis Completo de Datos",
            description="Pipeline completo de análisis de datos con múltiples etapas",
            steps=[
                WorkflowStep(
                    step_id="load_data",
                    agent_type="pandas_agent",
                    task_config={"action": "load_data"}
                ),
                WorkflowStep(
                    step_id="basic_analysis",
                    agent_type="pandas_agent",
                    task_config={"action": "analyze_data"},
                    dependencies=["load_data"]
                ),
                WorkflowStep(
                    step_id="text_summary",
                    agent_type="sophisticated_agent",
                    task_config={"action": "summarize_analysis"},
                    dependencies=["basic_analysis"]
                ),
                WorkflowStep(
                    step_id="qa_validation",
                    agent_type="langchain_agent",
                    task_config={"action": "validate_results"},
                    dependencies=["text_summary"]
                )
            ],
            max_parallel=2
        )
        
        # Workflow de procesamiento de documentos
        document_processing_workflow = WorkflowDefinition(
            workflow_id="document_processing",
            name="Procesamiento Completo de Documentos",
            description="Pipeline para procesar y analizar documentos de texto",
            steps=[
                WorkflowStep(
                    step_id="classify_document",
                    agent_type="sophisticated_agent",
                    task_config={"action": "classify_text"}
                ),
                WorkflowStep(
                    step_id="extract_entities",
                    agent_type="sophisticated_agent",
                    task_config={"action": "extract_entities"},
                    dependencies=["classify_document"]
                ),
                WorkflowStep(
                    step_id="generate_summary",
                    agent_type="sophisticated_agent",
                    task_config={"action": "summarize"},
                    dependencies=["extract_entities"]
                ),
                WorkflowStep(
                    step_id="store_knowledge",
                    agent_type="langchain_agent",
                    task_config={"action": "store_information"},
                    dependencies=["generate_summary"]
                )
            ]
        )
        
        self.workflow_definitions["data_analysis_complete"] = data_analysis_workflow
        self.workflow_definitions["document_processing"] = document_processing_workflow
        
        logger.info(f"Cargados {len(self.workflow_definitions)} workflows predefinidos")
    
    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """
        Registra un nuevo workflow en el orquestador.
        
        Args:
            workflow: Definición del workflow a registrar
        """
        self.workflow_definitions[workflow.workflow_id] = workflow
        logger.info(f"Workflow '{workflow.workflow_id}' registrado correctamente")
    
    async def execute_workflow(self, workflow_id: str, 
                             input_data: Dict[str, Any]) -> WorkflowExecution:
        """
        Ejecuta un workflow definido.
        
        Args:
            workflow_id: ID del workflow a ejecutar
            input_data: Datos de entrada para el workflow
            
        Returns:
            WorkflowExecution con el estado de ejecución
        """
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Workflow '{workflow_id}' no encontrado")
        
        workflow_def = self.workflow_definitions[workflow_id]
        execution_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_def=workflow_def,
            context=input_data.copy()
        )
        
        self.active_workflows[execution_id] = execution
        self.system_metrics['total_workflows'] += 1
        
        # Iniciar logging de conversación para el workflow
        conversation_log_id = conversation_logger.log_conversation_start(
            agent_id=self.agent_id,
            session_id=execution_id,
            agent_type="AdvancedOrchestrator",
            user_id=input_data.get("user_id"),
            metadata={
                "workflow_id": workflow_id,
                "workflow_name": workflow_def.name,
                "num_steps": len(workflow_def.steps),
                "max_parallel": workflow_def.max_parallel
            }
        )
        
        # Log de inicio del workflow
        conversation_logger.log_orchestrator_action(
            conversation_log_id=conversation_log_id,
            action=f"Iniciando workflow '{workflow_def.name}'",
            reasoning=f"Workflow {workflow_id} con {len(workflow_def.steps)} pasos",
            metadata={
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "input_data": input_data
            }
        )
        
        # Ejecutar hooks antes de empezar
        await self._run_hooks('before_start', execution)
        
        # Ejecutar workflow y esperar que termine
        await self._execute_workflow_async(execution, conversation_log_id)
        
        # Log de finalización del workflow
        conversation_logger.log_orchestrator_action(
            conversation_log_id=conversation_log_id,
            action=f"Workflow completado con estado: {execution.status.value}",
            metadata={
                "execution_id": execution_id,
                "final_status": execution.status.value,
                "completed_steps": len(execution.results),
                "total_steps": len(workflow_def.steps),
                "errors": execution.errors
            }
        )
        
        # Finalizar logging de conversación
        conversation_logger.log_conversation_end(
            conversation_log_id=conversation_log_id,
            reason="workflow_completed",
            summary={
                "workflow_id": workflow_id,
                "status": execution.status.value,
                "execution_time": (execution.end_time - execution.start_time).total_seconds() if execution.start_time and execution.end_time else None,
                "steps_completed": len(execution.results),
                "steps_total": len(workflow_def.steps)
            }
        )
        
        return execution
    
    async def _execute_workflow_async(self, execution: WorkflowExecution, conversation_log_id: Optional[str] = None) -> None:
        """Ejecuta un workflow de forma asíncrona."""
        try:
            execution.status = WorkflowStatus.RUNNING
            execution.start_time = datetime.now()
            
            logger.info(f"Iniciando workflow {execution.workflow_def.name} ({execution.execution_id})")
            
            # Ejecutar pasos según dependencias
            processed_steps = set()  # Pasos que ya fueron procesados (exitosos o fallidos)
            completed_steps = set()  # Solo pasos exitosos
            
            while len(processed_steps) < len(execution.workflow_def.steps):
                # Encontrar pasos listos para ejecutar
                ready_steps = []
                for step in execution.workflow_def.steps:
                    if (step.step_id not in processed_steps and
                        all(dep in completed_steps for dep in step.dependencies)):
                        ready_steps.append(step)
                
                if not ready_steps:
                    # No hay pasos listos, verificar si hay pasos ejecutándose
                    running_steps = [s for s in execution.workflow_def.steps 
                                   if s.status == StepStatus.RUNNING]
                    if not running_steps:
                        # Deadlock o todos los pasos fallaron
                        execution.status = WorkflowStatus.FAILED
                        break
                        execution.status = WorkflowStatus.FAILED
                        execution.errors.append("Workflow bloqueado - no hay pasos ejecutables")
                        break
                    
                    # Esperar un poco antes de verificar de nuevo
                    await asyncio.sleep(1)
                    continue
                
                # Ejecutar pasos en paralelo (limitado por max_parallel)
                semaphore = asyncio.Semaphore(execution.workflow_def.max_parallel)
                tasks = []
                
                for step in ready_steps:
                    task = asyncio.create_task(
                        self._execute_step_with_semaphore(semaphore, execution, step, conversation_log_id)
                    )
                    tasks.append(task)
                
                # Esperar a que al menos uno complete
                if tasks:
                    done, pending = await asyncio.wait(
                        tasks, 
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=60  # Timeout por lote
                    )
                    
                    # Procesar resultados completados
                    for task in done:
                        step_id, success, error_message = await task
                        processed_steps.add(step_id)  # Marcar como procesado independientemente del resultado
                        
                        if success:
                            completed_steps.add(step_id)
                            await self._run_hooks('on_step_complete', execution, step_id)
                        else:
                            logger.error(f"Paso {step_id} del workflow {execution.execution_id} falló permanentemente: {error_message}")
                
                # Actualizar progreso
                execution.progress = len(processed_steps) / len(execution.workflow_def.steps)
            
            # Finalizar workflow
            execution.end_time = datetime.now()
            
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
                self.system_metrics['successful_workflows'] += 1
                await self._run_hooks('after_complete', execution)
            else:
                self.system_metrics['failed_workflows'] += 1
                await self._run_hooks('on_error', execution)
            
            # Mover a historial
            self.workflow_history.append(execution)
            if execution.execution_id in self.active_workflows:
                del self.active_workflows[execution.execution_id]
            
            logger.info(f"Workflow {execution.workflow_def.name} terminado con estado: {execution.status}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.errors.append(f"Error ejecutando workflow: {str(e)}")
            logger.error(f"Error en workflow {execution.execution_id}: {e}")
            await self._run_hooks('on_error', execution)
    
    async def _execute_step_with_semaphore(self, semaphore: asyncio.Semaphore,
                                         execution: WorkflowExecution, 
                                         step: WorkflowStep,
                                         conversation_log_id: Optional[str] = None) -> Tuple[str, bool, Optional[str]]:
        """Ejecuta un paso del workflow con control de concurrencia."""
        async with semaphore:
            return await self._execute_step(execution, step, conversation_log_id)
    
    async def _execute_step(self, execution: WorkflowExecution, 
                          step: WorkflowStep,
                          conversation_log_id: Optional[str] = None) -> Tuple[str, bool, Optional[str]]:
        """Ejecuta un paso individual del workflow."""
        last_error = None
        
        # Log de inicio del paso
        if conversation_log_id:
            conversation_logger.log_orchestrator_action(
                conversation_log_id=conversation_log_id,
                action=f"Iniciando paso '{step.step_id}'",
                agent_selected=step.agent_type,
                reasoning=f"Ejecutando paso {step.step_id} del workflow con agente {step.agent_type}",
                metadata={
                    "step_id": step.step_id,
                    "agent_type": step.agent_type,
                    "dependencies": step.dependencies,
                    "attempt": 1,
                    "max_retries": step.max_retries
                }
            )
        
        # Bucle de reintentos
        for attempt in range(step.max_retries + 1):
            try:
                step.status = StepStatus.RUNNING
                step.start_time = datetime.now()
                execution.current_step = step.step_id
                
                logger.debug(f"Ejecutando paso {step.step_id} (intento {attempt + 1}) del workflow {execution.execution_id}")
                
                # Log de intento de ejecución
                if conversation_log_id and attempt > 0:
                    conversation_logger.log_orchestrator_action(
                        conversation_log_id=conversation_log_id,
                        action=f"Reintentando paso '{step.step_id}' (intento {attempt + 1})",
                        metadata={
                            "step_id": step.step_id,
                            "attempt": attempt + 1,
                            "previous_error": last_error
                        }
                    )
                
                # Seleccionar agente apropiado
                logger.debug(f"Seleccionando agente del tipo: {step.agent_type}")
                agent = await self._select_best_agent(step.agent_type)
                logger.debug(f"Agente seleccionado: {agent}")
                if not agent:
                    raise Exception(f"No hay agente disponible del tipo {step.agent_type}")
                
                # Preparar datos del paso
                logger.debug(f"Resolviendo inputs para step.task_config: {step.task_config}")
                step_data = self._resolve_step_inputs(step.task_config, execution.results)
                logger.debug(f"step_data después de _resolve_step_inputs: {step_data}")
                step_data.update(execution.context)
                logger.debug(f"step_data después de update con context: {step_data}")
                
                logger.debug(f"Datos enviados al agente {step.agent_type}: {step_data}")
                
                # Ejecutar con timeout
                result = await asyncio.wait_for(
                    agent.process_request(step_data),
                    timeout=step.timeout_seconds
                )
                
                # Verificar si el resultado indica éxito
                if isinstance(result, AgentResponse) and not result.is_success():
                    raise Exception(result.error or "Agente reportó falla sin detalles")
                
                step.result = result
                step.status = StepStatus.COMPLETED
                execution.results[step.step_id] = result
                
                # Log de finalización exitosa del paso
                if conversation_log_id:
                    conversation_logger.log_orchestrator_action(
                        conversation_log_id=conversation_log_id,
                        action=f"Paso '{step.step_id}' completado exitosamente",
                        metadata={
                            "step_id": step.step_id,
                            "agent_type": step.agent_type,
                            "execution_time": (datetime.now() - step.start_time).total_seconds(),
                            "attempt": attempt + 1,
                            "result_content_length": len(result.content) if hasattr(result, 'content') else 0
                        }
                    )
                
                # Actualizar métricas del agente
                await self._update_agent_metrics(step.agent_type, True, 
                                               (datetime.now() - step.start_time).total_seconds())
                
                return step.step_id, True, None
                
            except asyncio.TimeoutError:
                last_error = f"Timeout ejecutando paso {step.step_id}"
                logger.warning(last_error)
                
                # Log de timeout
                if conversation_log_id:
                    conversation_logger.log_orchestrator_action(
                        conversation_log_id=conversation_log_id,
                        action=f"Timeout en paso '{step.step_id}'",
                        metadata={
                            "step_id": step.step_id,
                            "agent_type": step.agent_type,
                            "attempt": attempt + 1,
                            "timeout_seconds": step.timeout_seconds,
                            "error": last_error
                        }
                    )
            except Exception as e:
                last_error = f"Error en paso {step.step_id}: {str(e)}"
                logger.warning(last_error)
                
                # Log de error
                if conversation_log_id:
                    conversation_logger.log_orchestrator_action(
                        conversation_log_id=conversation_log_id,
                        action=f"Error en paso '{step.step_id}'",
                        metadata={
                            "step_id": step.step_id,
                            "agent_type": step.agent_type,
                            "attempt": attempt + 1,
                            "error_type": type(e).__name__,
                            "error": last_error
                        }
                    )
            
            # Si no es el último intento y auto_retry está habilitado, esperar antes del siguiente intento
            if attempt < step.max_retries and execution.workflow_def.auto_retry:
                await asyncio.sleep(2 ** attempt)  # Backoff exponencial
            else:
                break  # Salir del bucle de reintentos
        
        # Si llegamos aquí, todos los reintentos fallaron
        step.status = StepStatus.FAILED
        step.error = last_error
        execution.errors.append(f"Paso {step.step_id}: {last_error}")
        
        # Log de fallo definitivo del paso
        if conversation_log_id:
            conversation_logger.log_orchestrator_action(
                conversation_log_id=conversation_log_id,
                action=f"Paso '{step.step_id}' falló definitivamente",
                metadata={
                    "step_id": step.step_id,
                    "agent_type": step.agent_type,
                    "total_attempts": step.max_retries + 1,
                    "final_error": last_error
                }
            )
        
        # Actualizar métricas del agente
        await self._update_agent_metrics(step.agent_type, False, 0)
        
        return step.step_id, False, last_error
    
    async def _select_best_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """Selecciona el mejor agente disponible según métricas."""
        logger.debug(f"_select_best_agent iniciado para tipo: {agent_type}")
        logger.debug(f"load_balancing_enabled: {self.load_balancing_enabled}")
        
        if not self.load_balancing_enabled:
            agent = self.specialized_agents.get(agent_type)
            logger.debug(f"Load balancing deshabilitado, agente obtenido: {agent}")
            return agent
        
        # Buscar agentes del tipo solicitado
        candidates = []
        logger.debug(f"Agentes especializados disponibles: {list(self.specialized_agents.keys())}")
        
        for agent_name, agent in self.specialized_agents.items():
            logger.debug(f"Evaluando agente: {agent_name}, tipo buscado: {agent_type}")
            
            if agent_type in agent_name:
                logger.debug(f"Agente {agent_name} coincide con tipo {agent_type}")
                
                try:
                    # Aquí estaba el problema - is_ready es una propiedad, no un método
                    logger.debug(f"Tipo de agent.is_ready: {type(agent.is_ready)}")
                    is_ready_result = agent.is_ready  # Cambio: sin paréntesis
                    logger.debug(f"agent.is_ready retornó: {is_ready_result}")
                    
                    if is_ready_result:
                        metrics = self.agent_metrics.get(agent_name)
                        if metrics and metrics.current_load < metrics.max_concurrent:
                            candidates.append((agent, metrics))
                            logger.debug(f"Agente {agent_name} añadido como candidato")
                        else:
                            logger.debug(f"Agente {agent_name} no añadido: metrics={metrics}")
                    else:
                        logger.debug(f"Agente {agent_name} no está ready")
                        
                except Exception as e:
                    logger.error(f"Error accediendo is_ready en agente {agent_name}: {e}")
                    logger.error(f"Tipo de agent: {type(agent)}")
                    logger.error(f"Atributos del agente: {dir(agent)}")
                    raise
            else:
                logger.debug(f"Agente {agent_name} no coincide con tipo {agent_type}")
        
        if not candidates:
            logger.debug("No se encontraron candidatos")
            return None
        
        # Seleccionar el agente con menor carga y mejor rendimiento
        best_agent = min(candidates, key=lambda x: (
            x[1].current_load / x[1].max_concurrent,  # Factor de carga
            x[1].error_rate,  # Tasa de error
            -x[1].availability  # Disponibilidad (negativo para orden ascendente)
        ))[0]
        
        # Incrementar carga del agente seleccionado
        agent_name = None
        for name, agent in self.specialized_agents.items():
            if agent == best_agent:
                agent_name = name
                break
        
        if agent_name:
            self.agent_metrics[agent_name].current_load += 1
        
        logger.debug(f"Agente seleccionado: {best_agent}")
        return best_agent
    
    async def _update_agent_metrics(self, agent_type: str, success: bool, 
                                  response_time: float) -> None:
        """Actualiza las métricas de un agente."""
        for agent_name in self.agent_metrics:
            if agent_type in agent_name:
                metrics = self.agent_metrics[agent_name]
                
                metrics.total_requests += 1
                metrics.last_request_time = datetime.now()
                
                if success:
                    metrics.successful_requests += 1
                else:
                    metrics.failed_requests += 1
                
                # Actualizar promedio de tiempo de respuesta
                if response_time > 0:
                    total_time = metrics.average_response_time * (metrics.total_requests - 1)
                    metrics.average_response_time = (total_time + response_time) / metrics.total_requests
                
                # Calcular tasa de error
                metrics.error_rate = metrics.failed_requests / metrics.total_requests
                
                # Reducir carga actual
                metrics.current_load = max(0, metrics.current_load - 1)
                
                break
    
    def _select_best_agent_from_list(self, agent_names: List[str]) -> str:
        """
        Método de utilidad para testing: selecciona el mejor agente de una lista.
        
        Args:
            agent_names: Lista de nombres de agentes candidatos
            
        Returns:
            Nombre del agente con menor carga
        """
        if not agent_names:
            return None
            
        best_agent = None
        best_score = float('inf')
        
        for agent_name in agent_names:
            if agent_name in self.agent_metrics:
                metrics = self.agent_metrics[agent_name]
                # Score basado en carga actual y tiempo de respuesta
                score = metrics.current_load + (metrics.average_response_time * 0.1)
                
                if score < best_score:
                    best_score = score
                    best_agent = agent_name
        
        return best_agent or agent_names[0]  # Fallback al primero
    
    async def _run_hooks(self, hook_type: str, execution: WorkflowExecution, 
                        step_id: str = None) -> None:
        """Ejecuta hooks del workflow."""
        hooks = self.workflow_hooks.get(hook_type, [])
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(execution, step_id)
                else:
                    hook(execution, step_id)
            except Exception as e:
                logger.error(f"Error ejecutando hook {hook_type}: {e}")
    
    async def _workflow_monitor(self) -> None:
        """Monitor de workflows activos."""
        while True:
            try:
                current_time = datetime.now()
                
                # Verificar workflows con timeout
                for execution_id, execution in list(self.active_workflows.items()):
                    if execution.start_time:
                        elapsed = (current_time - execution.start_time).total_seconds() / 60
                        if elapsed > execution.workflow_def.timeout_minutes:
                            execution.status = WorkflowStatus.FAILED
                            execution.errors.append("Workflow timeout")
                            logger.warning(f"Workflow {execution_id} cancelado por timeout")
                
                # Actualizar métricas del sistema
                self.system_metrics['peak_concurrent_workflows'] = max(
                    self.system_metrics['peak_concurrent_workflows'],
                    len(self.active_workflows)
                )
                
                await asyncio.sleep(30)  # Verificar cada 30 segundos
                
            except Exception as e:
                logger.error(f"Error en monitor de workflows: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collector(self) -> None:
        """Recolector de métricas del sistema."""
        while True:
            try:
                # Calcular tiempo promedio de workflows
                if self.workflow_history:
                    total_time = sum(
                        (w.end_time - w.start_time).total_seconds()
                        for w in self.workflow_history[-100:]  # Últimos 100
                        if w.end_time and w.start_time
                    )
                    count = len([w for w in self.workflow_history[-100:] 
                               if w.end_time and w.start_time])
                    
                    if count > 0:
                        self.system_metrics['average_workflow_time'] = total_time / count
                
                # Limpiar historial antiguo (mantener últimos 1000)
                if len(self.workflow_history) > 1000:
                    self.workflow_history = self.workflow_history[-1000:]
                
                await asyncio.sleep(300)  # Recolectar cada 5 minutos
                
            except Exception as e:
                logger.error(f"Error recolectando métricas: {e}")
                await asyncio.sleep(300)
    
    async def _task_processor(self) -> None:
        """Procesador de cola de tareas distribuidas."""
        while True:
            try:
                if self.task_queue:
                    task = self.task_queue.pop(0)
                    task_id = str(uuid.uuid4())
                    
                    self.processing_tasks[task_id] = task
                    
                    # Procesar tarea de forma asíncrona
                    asyncio.create_task(self._process_distributed_task(task_id, task))
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error procesando tareas: {e}")
                await asyncio.sleep(5)
    
    async def _process_distributed_task(self, task_id: str, task: Dict[str, Any]) -> None:
        """Procesa una tarea distribuida."""
        try:
            # Implementar lógica de procesamiento distribuido
            result = await self.process(task)
            
            # Guardar resultado
            task['result'] = result
            task['status'] = 'completed'
            
        except Exception as e:
            task['error'] = str(e)
            task['status'] = 'failed'
            logger.error(f"Error procesando tarea {task_id}: {e}")
        
        finally:
            if task_id in self.processing_tasks:
                del self.processing_tasks[task_id]
    
    def add_workflow_hook(self, hook_type: str, callback: Callable) -> None:
        """Agrega un hook al sistema de workflows."""
        if hook_type in self.workflow_hooks:
            self.workflow_hooks[hook_type].append(callback)
    
    def get_workflow_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Obtiene el estado de un workflow."""
        return self.active_workflows.get(execution_id)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del sistema."""
        uptime = (datetime.now() - self.system_metrics['uptime_start']).total_seconds()
        
        return {
            **self.system_metrics,
            'uptime_seconds': uptime,
            'active_workflows': len(self.active_workflows),
            'agent_metrics': {name: {
                'total_requests': m.total_requests,
                'success_rate': (m.successful_requests / m.total_requests) if m.total_requests > 0 else 0,
                'average_response_time': m.average_response_time,
                'current_load': m.current_load,
                'availability': m.availability
            } for name, m in self.agent_metrics.items()},
            'processing_tasks': len(self.processing_tasks),
            'queued_tasks': len(self.task_queue)
        }
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancela un workflow en ejecución."""
        if execution_id in self.active_workflows:
            execution = self.active_workflows[execution_id]
            execution.status = WorkflowStatus.CANCELLED
            
            # Cancelar pasos en ejecución
            for step in execution.workflow_def.steps:
                if step.status == StepStatus.RUNNING:
                    step.status = StepStatus.SKIPPED
            
            return True
        return False
    
    def submit_distributed_task(self, task: Dict[str, Any]) -> str:
        """Envía una tarea a la cola distribuida."""
        task_id = str(uuid.uuid4())
        task['task_id'] = task_id
        task['submitted_at'] = datetime.now()
        
        self.task_queue.append(task)
        return task_id
    
    def add_hook(self, event_name: str, callback_func):
        """Añade un hook para un evento específico."""
        if not hasattr(self, 'hooks'):
            self.hooks = {}
        if event_name not in self.hooks:
            self.hooks[event_name] = []
        self.hooks[event_name].append(callback_func)
    
    def _resolve_step_inputs(self, step_inputs: Dict[str, Any], previous_results: Dict[str, AgentResponse]) -> Dict[str, Any]:
        """Resuelve templates en los inputs de un paso usando resultados previos."""
        resolved = {}
        
        for key, value in step_inputs.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                # Es un template, extraer la referencia
                template = value[2:-2].strip()  # Remover {{ }}
                
                if "." in template:
                    # Referencia a una propiedad: step1.result, step1.metadata.count
                    parts = template.split(".")
                    step_ref = parts[0]
                    property_path = parts[1:]
                    
                    if step_ref in previous_results:
                        result_obj = previous_results[step_ref]
                        resolved_value = result_obj
                        
                        # Navegar la ruta de propiedades
                        for prop in property_path:
                            if prop == "result":
                                resolved_value = resolved_value.content
                            elif prop == "metadata" and hasattr(resolved_value, 'metadata'):
                                resolved_value = resolved_value.metadata
                            elif isinstance(resolved_value, dict) and prop in resolved_value:
                                resolved_value = resolved_value[prop]
                            else:
                                resolved_value = None
                                break
                        
                        resolved[key] = resolved_value
                    else:
                        resolved[key] = value  # No se pudo resolver, mantener original
                else:
                    # Referencia simple al resultado completo
                    if template in previous_results:
                        resolved[key] = previous_results[template].content
                    else:
                        resolved[key] = value
            else:
                # No es un template
                resolved[key] = value
                
        return resolved
