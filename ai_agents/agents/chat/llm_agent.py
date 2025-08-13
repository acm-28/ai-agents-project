"""
Agente de chat simple con OpenAI.
Versión migrada y mejorada del llm_chat_agent.py
"""

import logging
from typing import Dict, Any, Optional, Union
import openai

from ai_agents.core.base_agent import BaseAgent
from ai_agents.core.types import AgentResponse, Message, MessageRole, AgentConfig
from ai_agents.core.exceptions import InitializationError, ProcessingError
from ai_agents.config.settings import settings

logger = logging.getLogger(__name__)


class LLMChatAgent(BaseAgent):
    """
    Agente de chat simple usando OpenAI directamente.
    
    Características:
    - Integración directa con OpenAI API
    - Sin memoria persistente (stateless)
    - Configuración simple
    - Ideal para casos de uso básicos
    """
    
    def __init__(self, 
                 agent_id: Optional[str] = None,
                 config: Optional[Union[Dict, AgentConfig]] = None):
        """
        Inicializa el agente de chat simple.
        
        Args:
            agent_id: ID único del agente
            config: Configuración del agente
        """
        super().__init__(agent_id, config)
        
        self.client: Optional[openai.OpenAI] = None
        self.system_message = self.config.system_message or "Eres un asistente útil y amigable."
        
    async def initialize(self) -> None:
        """Inicializa el cliente de OpenAI."""
        try:
            self.client = openai.OpenAI(api_key=settings.openai_api_key)
            
            # Test de conexión
            test_response = self.client.models.list()
            
            self.logger.info(f"LLMChatAgent {self.agent_id} inicializado correctamente")
            
        except Exception as e:
            raise InitializationError(
                f"Error inicializando LLMChatAgent: {str(e)}",
                agent_id=self.agent_id
            )
    
    async def process(self, input_data: Union[str, Dict, Message]) -> AgentResponse:
        """
        Procesa un mensaje y retorna la respuesta del agente.
        
        Args:
            input_data: Mensaje de entrada
            
        Returns:
            AgentResponse con la respuesta del agente
        """
        try:
            # Normalizar entrada
            if isinstance(input_data, str):
                message_content = input_data
            elif isinstance(input_data, Message):
                message_content = input_data.content
            elif isinstance(input_data, dict):
                message_content = input_data.get("content", input_data.get("message", ""))
            else:
                raise ValueError(f"Tipo de entrada no soportado: {type(input_data)}")
            
            if not message_content.strip():
                raise ValueError("El mensaje no puede estar vacío")
            
            # Llamada a OpenAI
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": message_content}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Extraer información de la respuesta
            choice = response.choices[0]
            content = choice.message.content.strip()
            
            return AgentResponse(
                content=content,
                metadata={
                    "model": self.config.model,
                    "system_message": self.system_message,
                    "input_length": len(message_content),
                    "finish_reason": choice.finish_reason
                },
                tokens_used=response.usage.total_tokens if response.usage else None
            )
            
        except openai.APIError as e:
            raise ProcessingError(
                f"Error de API de OpenAI: {str(e)}",
                agent_id=self.agent_id
            )
        except Exception as e:
            raise ProcessingError(
                f"Error procesando mensaje: {str(e)}",
                agent_id=self.agent_id
            )
    
    def set_system_message(self, system_message: str) -> None:
        """
        Cambia el mensaje del sistema.
        
        Args:
            system_message: Nuevo mensaje del sistema
        """
        self.system_message = system_message
        self.config.system_message = system_message
        self.logger.info(f"Sistema actualizado para agente {self.agent_id}")
    
    async def get_available_models(self) -> list:
        """
        Obtiene la lista de modelos disponibles.
        
        Returns:
            Lista de modelos disponibles
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            self.logger.error(f"Error obteniendo modelos: {e}")
            return []
