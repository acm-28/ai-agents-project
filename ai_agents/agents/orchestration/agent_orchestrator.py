"""
Agent Orchestrator - Agente de integración y coordinación unificado.

Este agente actúa como punto de entrada único para coordinar 
múltiples agentes especializados según la intención del usuario.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from ...core.base_agent import BaseAgent
from ...core.types import AgentResponse, AgentState

# Imports condicionales para agentes especializados
try:
    from ..chat.langchain_agent import LangChainChatAgent
    from ..chat.llm_agent import LLMChatAgent
    from ..qa.memory_qa_agent import MemoryQAAgent
    from ..data_analysis.pandas_agent import PandasAgent
    from ..workflows.sophisticated_agent import SophisticatedAgent
    HAS_SPECIALIZED_AGENTS = True
except ImportError as e:
    logging.warning(f"Algunos agentes especializados no están disponibles: {e}")
    HAS_SPECIALIZED_AGENTS = False

logger = logging.getLogger(__name__)


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


@dataclass
class AgentCapability:
    """Capacidad de un agente."""
    agent_class: type
    task_types: List[TaskType]
    keywords: List[str]
    priority: int
    description: str


class AgentOrchestrator(BaseAgent):
    """
    Orquestador de agentes que dirige las solicitudes al agente más apropiado.
    
    Funcionalidades:
    - Clasificación automática de intenciones
    - Enrutamiento inteligente a agentes especializados
    - Coordinación de múltiples agentes
    - Mantenimiento de contexto global
    - Fallback a agentes genéricos
    """
    
    def __init__(self, agent_id: str = None, auto_initialize_agents: bool = True, **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        
        # Configuración
        self.auto_initialize_agents = auto_initialize_agents
        
        # Registro de agentes especializados
        self.specialized_agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        
        # Historial de interacciones
        self.interaction_history: List[Dict[str, Any]] = []
        self.current_session_context: Dict[str, Any] = {}
        
        # Patrones de clasificación
        self.classification_patterns = {
            TaskType.DATA_ANALYSIS: [
                r'analiz[ar|e|o].*(datos|csv|excel|dataframe)',
                r'(cargar|procesar|limpiar).*(archivo|dataset)',
                r'(estadísticas|correlación|visualiz)',
                r'pandas|numpy|matplotlib',
                r'(csv|excel|json).*analiz'
            ],
            TaskType.QA_MEMORY: [
                r'(recordar|memoria|histórico)',
                r'(pregunta|consulta).*anterior',
                r'(contexto|conversación).*previa',
                r'qué.*dijiste.*antes',
                r'(buscar|encontrar).*información'
            ],
            TaskType.TEXT_ANALYSIS: [
                r'(clasificar|categorizar).*texto',
                r'(extraer|identificar).*entidades',
                r'(resumir|resumen).*texto',
                r'analiz.*sentiment',
                r'(procesar|analizar).*documento'
            ],
            TaskType.COMPLEX_WORKFLOW: [
                r'(workflow|flujo).*complejo',
                r'múltiples.*pasos',
                r'(coordinar|orquestar)',
                r'pipeline.*procesamiento',
                r'automatiz.*proceso'
            ],
            TaskType.CHAT: [
                r'^(hola|hello|hi|saludos)',
                r'(conversar|charlar|hablar)',
                r'cómo.*estás',
                r'^(ayuda|help)',
                r'(explicar|contar).*general'
            ]
        }
    
    async def initialize(self) -> None:
        """Inicializa el orquestador y sus agentes especializados."""
        if not HAS_SPECIALIZED_AGENTS:
            logger.warning("Agentes especializados no disponibles. Funcionalidad limitada.")
            return
        
        if not self.auto_initialize_agents:
            logger.info("Inicialización automática de agentes deshabilitada.")
            return
        
        # Configurar capacidades de agentes
        self._setup_agent_capabilities()
        
        # Inicializar agentes especializados
        await self._initialize_specialized_agents()
        
        logger.info("AgentOrchestrator inicializado correctamente")
    
    def _setup_agent_capabilities(self) -> None:
        """Configura las capacidades de cada agente especializado."""
        self.agent_capabilities = {
            'pandas_agent': AgentCapability(
                agent_class=PandasAgent,
                task_types=[TaskType.DATA_ANALYSIS],
                keywords=['datos', 'csv', 'excel', 'pandas', 'análisis', 'estadísticas'],
                priority=1,
                description="Especialista en análisis de datos con pandas"
            ),
            'sophisticated_agent': AgentCapability(
                agent_class=SophisticatedAgent,
                task_types=[TaskType.TEXT_ANALYSIS, TaskType.COMPLEX_WORKFLOW],
                keywords=['texto', 'clasificar', 'entidades', 'resumen', 'workflow'],
                priority=2,
                description="Análisis avanzado de texto y workflows complejos"
            ),
            'memory_qa_agent': AgentCapability(
                agent_class=MemoryQAAgent,
                task_types=[TaskType.QA_MEMORY],
                keywords=['recordar', 'memoria', 'pregunta', 'anterior', 'contexto'],
                priority=1,
                description="Q&A con memoria conversacional"
            ),
            'langchain_agent': AgentCapability(
                agent_class=LangChainChatAgent,
                task_types=[TaskType.CHAT, TaskType.COMPLEX_WORKFLOW],
                keywords=['conversar', 'chat', 'general', 'ayuda'],
                priority=3,
                description="Chat general con LangChain"
            ),
            'llm_agent': AgentCapability(
                agent_class=LLMChatAgent,
                task_types=[TaskType.CHAT],
                keywords=['simple', 'básico', 'rápido'],
                priority=4,
                description="Chat simple y directo"
            )
        }
    
    async def _initialize_specialized_agents(self) -> None:
        """Inicializa todos los agentes especializados."""
        for agent_name, capability in self.agent_capabilities.items():
            try:
                agent_instance = capability.agent_class()
                await agent_instance._safe_initialize()
                self.specialized_agents[agent_name] = agent_instance
                logger.info(f"Agente '{agent_name}' inicializado correctamente")
            except Exception as e:
                logger.error(f"Error inicializando agente '{agent_name}': {e}")
                # Continuar con otros agentes
                continue
    
    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Procesa una solicitud dirigiéndola al agente más apropiado.
        
        Args:
            request: Diccionario con 'message' y parámetros opcionales
            
        Returns:
            AgentResponse con el resultado del procesamiento
        """
        message = request.get('message', '')
        
        if not message.strip():
            return AgentResponse(
                content="Por favor proporciona un mensaje para procesar.",
                metadata={"error": True, "needs_message": True}
            )
        
        try:
            # 1. Clasificar la tarea
            classification = await self._classify_task(message, request)
            
            # 2. Seleccionar agente apropiado
            selected_agent = await self._select_agent(classification)
            
            if not selected_agent:
                return await self._fallback_response(message, classification)
            
            # 3. Procesar con el agente seleccionado
            response = await self._process_with_agent(selected_agent, request, classification)
            
            # 4. Actualizar contexto e historial
            await self._update_context(message, response, classification)
            
            return response
            
        except Exception as e:
            error_msg = f"Error en orquestación: {str(e)}"
            logger.error(error_msg)
            return AgentResponse(content=error_msg, metadata={"error": True})
    
    async def _classify_task(self, message: str, request: Dict[str, Any]) -> TaskClassification:
        """Clasifica el tipo de tarea basado en el mensaje y contexto."""
        message_lower = message.lower()
        scores = {}
        
        # Analizar patrones de texto
        for task_type, patterns in self.classification_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    score += 1
            scores[task_type] = score / len(patterns) if patterns else 0
        
        # Considerar parámetros adicionales
        if 'file_path' in request or 'sample_data' in request:
            scores[TaskType.DATA_ANALYSIS] += 0.5
        
        if 'text' in request and len(request.get('text', '')) > 100:
            scores[TaskType.TEXT_ANALYSIS] += 0.3
        
        # Considerar contexto de sesión
        if self.current_session_context.get('last_task_type'):
            last_type = self.current_session_context['last_task_type']
            if last_type in scores:
                scores[last_type] += 0.2  # Bonificación por continuidad
        
        # Determinar la mejor clasificación
        if not scores or max(scores.values()) == 0:
            best_task = TaskType.CHAT
            confidence = 0.3
        else:
            best_task = max(scores, key=scores.get)
            confidence = min(scores[best_task], 1.0)
        
        # Mapear a agente
        agent_mapping = {
            TaskType.DATA_ANALYSIS: 'pandas_agent',
            TaskType.TEXT_ANALYSIS: 'sophisticated_agent',
            TaskType.QA_MEMORY: 'memory_qa_agent',
            TaskType.COMPLEX_WORKFLOW: 'sophisticated_agent',
            TaskType.CHAT: 'langchain_agent',
            TaskType.UNKNOWN: 'llm_agent'
        }
        
        return TaskClassification(
            task_type=best_task,
            confidence=confidence,
            agent_name=agent_mapping.get(best_task, 'llm_agent'),
            reasoning=f"Patrón detectado para {best_task.value} con confianza {confidence:.2f}",
            parameters=request
        )
    
    async def _select_agent(self, classification: TaskClassification) -> Optional[BaseAgent]:
        """Selecciona el agente más apropiado para la clasificación."""
        primary_agent = self.specialized_agents.get(classification.agent_name)
        
        if primary_agent and primary_agent.is_ready():
            return primary_agent
        
        # Buscar agente alternativo del mismo tipo de tarea
        for agent_name, capability in self.agent_capabilities.items():
            if (classification.task_type in capability.task_types and 
                agent_name in self.specialized_agents and
                self.specialized_agents[agent_name].is_ready()):
                return self.specialized_agents[agent_name]
        
        # Fallback a agente general
        fallback_agents = ['langchain_agent', 'llm_agent']
        for agent_name in fallback_agents:
            if (agent_name in self.specialized_agents and 
                self.specialized_agents[agent_name].is_ready()):
                return self.specialized_agents[agent_name]
        
        return None
    
    async def _process_with_agent(self, agent: BaseAgent, request: Dict[str, Any], 
                                classification: TaskClassification) -> AgentResponse:
        """Procesa la solicitud con el agente seleccionado."""
        try:
            response = await agent.process(request)
            
            # Enriquecer metadata con información de orquestación
            response.metadata.update({
                "orchestrator": {
                    "selected_agent": agent.__class__.__name__,
                    "task_classification": classification.task_type.value,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning
                }
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error procesando con {agent.__class__.__name__}: {e}")
            return AgentResponse(
                content=f"Error procesando solicitud con {agent.__class__.__name__}: {str(e)}",
                metadata={"error": True, "agent_error": str(e)}
            )
    
    async def _fallback_response(self, message: str, 
                               classification: TaskClassification) -> AgentResponse:
        """Respuesta de fallback cuando no hay agentes disponibles."""
        return AgentResponse(
            content=f"""Lo siento, no hay agentes especializados disponibles para procesar tu solicitud.

**Clasificación detectada:** {classification.task_type.value}
**Agente sugerido:** {classification.agent_name}
**Confianza:** {classification.confidence:.2f}

Por favor verifica que los agentes especializados estén correctamente inicializados.""",
            metadata={
                "error": True,
                "fallback_mode": True,
                "classification": classification.task_type.value,
                "no_agents_available": True
            }
        )
    
    async def _update_context(self, message: str, response: AgentResponse, 
                            classification: TaskClassification) -> None:
        """Actualiza el contexto de sesión e historial."""
        # Actualizar contexto de sesión
        self.current_session_context.update({
            'last_task_type': classification.task_type,
            'last_agent': classification.agent_name,
            'last_confidence': classification.confidence
        })
        
        # Agregar al historial
        interaction = {
            'timestamp': response.timestamp,
            'message': message,
            'classification': classification.task_type.value,
            'agent_used': classification.agent_name,
            'confidence': classification.confidence,
            'success': not response.metadata.get('error', False)
        }
        
        self.interaction_history.append(interaction)
        
        # Mantener historial limitado (últimas 50 interacciones)
        if len(self.interaction_history) > 50:
            self.interaction_history = self.interaction_history[-50:]
    
    def get_capabilities(self) -> List[str]:
        """Obtiene las capacidades combinadas de todos los agentes."""
        capabilities = [
            "Orquestación automática de agentes",
            "Clasificación inteligente de tareas",
            "Enrutamiento adaptativo",
            "Mantenimiento de contexto de sesión",
            "Fallback automático"
        ]
        
        # Agregar capacidades de agentes especializados
        for agent_name, agent in self.specialized_agents.items():
            if hasattr(agent, 'get_capabilities'):
                agent_caps = agent.get_capabilities()
                capabilities.extend([f"{agent_name}: {cap}" for cap in agent_caps])
        
        return capabilities
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de orquestación."""
        if not self.interaction_history:
            return {"no_interactions": True}
        
        # Calcular estadísticas
        total_interactions = len(self.interaction_history)
        successful_interactions = sum(1 for i in self.interaction_history if i['success'])
        
        # Conteo por tipo de tarea
        task_counts = {}
        agent_counts = {}
        
        for interaction in self.interaction_history:
            task_type = interaction['classification']
            agent_used = interaction['agent_used']
            
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
            agent_counts[agent_used] = agent_counts.get(agent_used, 0) + 1
        
        return {
            "total_interactions": total_interactions,
            "success_rate": successful_interactions / total_interactions,
            "task_distribution": task_counts,
            "agent_usage": agent_counts,
            "specialized_agents_count": len(self.specialized_agents),
            "current_session_context": self.current_session_context
        }
    
    def get_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene información de agentes disponibles."""
        agents_info = {}
        
        for agent_name, agent in self.specialized_agents.items():
            capability = self.agent_capabilities.get(agent_name)
            agents_info[agent_name] = {
                "class": agent.__class__.__name__,
                "state": agent.state.value,
                "ready": agent.is_ready(),
                "task_types": [t.value for t in capability.task_types] if capability else [],
                "description": capability.description if capability else "No description",
                "priority": capability.priority if capability else 999
            }
        
        return agents_info

    def register_agent(self, agent_name: str, agent_instance: BaseAgent, 
                      capability: Optional[AgentCapability] = None) -> None:
        """
        Registra un agente directamente en el orquestador.
        
        Útil para testing o inyección manual de agentes.
        
        Args:
            agent_name: Nombre del agente
            agent_instance: Instancia del agente
            capability: Capacidad del agente (opcional)
        """
        self.specialized_agents[agent_name] = agent_instance
        
        if capability:
            self.agent_capabilities[agent_name] = capability
        
        logger.info(f"Agente '{agent_name}' registrado manualmente")
