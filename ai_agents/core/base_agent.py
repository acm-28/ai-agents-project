"""
Clase base mejorada para todos los agentes del framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import uuid
import logging
import time
from datetime import datetime

from ai_agents.core.types import AgentResponse, AgentState, AgentConfig, Message
from ai_agents.core.exceptions import AgentError, InitializationError, ProcessingError
from ai_agents.config.settings import settings

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Clase base abstracta mejorada para todos los agentes.
    
    Proporciona funcionalidad común como:
    - Gestión de estado del agente
    - Configuración centralizada
    - Logging estructurado
    - Manejo de errores
    - Métricas básicas
    """
    
    def __init__(self, 
                 agent_id: Optional[str] = None, 
                 config: Optional[Union[Dict, AgentConfig]] = None):
        """
        Inicializa el agente base.
        
        Args:
            agent_id: ID único del agente
            config: Configuración del agente
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.state = AgentState.CREATED
        self.created_at = datetime.now()
        self.metadata = {}
        self._is_initialized = False
        self._initialization_time = None
        
        # Configuración
        if isinstance(config, dict):
            self.config = AgentConfig(**config)
        elif isinstance(config, AgentConfig):
            self.config = config
        else:
            self.config = AgentConfig(agent_type=self.__class__.__name__)
        
        # Logger específico del agente
        self.logger = logging.getLogger(f"ai_agents.{self.__class__.__name__}")
        
        self.logger.info(f"Agente {self.agent_id} creado con configuración: {self.config.model_dump()}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Inicializa el agente de manera asíncrona.
        
        Debe ser implementado por cada agente específico.
        """
        pass
    
    @abstractmethod
    async def process(self, input_data: Union[str, Dict, Message]) -> AgentResponse:
        """
        Procesa una entrada y retorna una respuesta.
        
        Args:
            input_data: Datos de entrada (string, dict o Message)
            
        Returns:
            AgentResponse: Respuesta del agente
        """
        pass
    
    async def _safe_initialize(self) -> None:
        """Inicialización segura con manejo de errores."""
        if self._is_initialized:
            self.logger.warning(f"Agente {self.agent_id} ya está inicializado")
            return
        
        try:
            self.state = AgentState.INITIALIZING
            start_time = time.time()
            
            await self.initialize()
            
            self._initialization_time = time.time() - start_time
            self._is_initialized = True
            self.state = AgentState.READY
            
            self.logger.info(
                f"Agente {self.agent_id} inicializado correctamente "
                f"en {self._initialization_time:.2f}s"
            )
            
        except Exception as e:
            self.state = AgentState.ERROR
            error_msg = f"Error inicializando agente {self.agent_id}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise InitializationError(error_msg, agent_id=self.agent_id) from e
    
    async def _safe_process(self, input_data: Union[str, Dict, Message]) -> AgentResponse:
        """Procesamiento seguro con manejo de errores y métricas."""
        if not self._is_initialized:
            await self._safe_initialize()
        
        if self.state != AgentState.READY:
            raise ProcessingError(
                f"Agente {self.agent_id} no está listo para procesar. Estado: {self.state}",
                agent_id=self.agent_id
            )
        
        try:
            self.state = AgentState.PROCESSING
            start_time = time.time()
            
            # Procesar entrada
            response = await self.process(input_data)
            
            # Añadir métricas a la respuesta
            processing_time = time.time() - start_time
            response.processing_time = processing_time
            response.agent_id = self.agent_id
            
            self.state = AgentState.READY
            
            self.logger.info(
                f"Agente {self.agent_id} procesó entrada en {processing_time:.2f}s"
            )
            
            return response
            
        except Exception as e:
            self.state = AgentState.ERROR
            error_msg = f"Error procesando en agente {self.agent_id}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Retornar respuesta con error
            return AgentResponse(
                content="",
                error=error_msg,
                agent_id=self.agent_id,
                processing_time=time.time() - start_time
            )
    
    async def cleanup(self) -> None:
        """Limpia recursos del agente."""
        self.state = AgentState.STOPPED
        self.logger.info(f"Agente {self.agent_id} detenido y recursos limpiados")
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna información del agente."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "state": self.state.value,
            "config": self.config.model_dump(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "is_initialized": self._is_initialized,
            "initialization_time": self._initialization_time
        }
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Actualiza metadata del agente."""
        self.metadata[key] = value
        self.logger.debug(f"Metadata actualizada: {key} = {value}")
    
    @property
    def is_initialized(self) -> bool:
        """Verifica si el agente está inicializado."""
        return self._is_initialized
    
    @property
    def is_ready(self) -> bool:
        """Verifica si el agente está listo para procesar."""
        return self.state == AgentState.READY
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, state={self.state.value})"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(agent_id='{self.agent_id}', "
                f"state={self.state}, initialized={self._is_initialized})")
