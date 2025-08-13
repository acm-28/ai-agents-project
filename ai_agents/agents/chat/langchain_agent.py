"""
Agente de chat con memoria usando LangChain.
Versión migrada y mejorada del agent1_context_awareness.py
"""

import logging
from typing import Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from ai_agents.core.base_agent import BaseAgent
from ai_agents.core.types import AgentResponse, Message, MessageRole, AgentConfig
from ai_agents.core.exceptions import InitializationError, ProcessingError
from ai_agents.config.settings import settings

logger = logging.getLogger(__name__)


class LangChainChatAgent(BaseAgent):
    """
    Agente de chat con memoria conversacional usando LangChain.
    
    Características:
    - Memoria persistente por sesión
    - Configuración de sistema personalizable
    - Integración con OpenAI
    - Manejo de errores robusto
    """
    
    def __init__(self, 
                 agent_id: Optional[str] = None,
                 config: Optional[Union[Dict, AgentConfig]] = None):
        """
        Inicializa el agente de chat.
        
        Args:
            agent_id: ID único del agente
            config: Configuración del agente
        """
        super().__init__(agent_id, config)
        
        self.llm: Optional[ChatOpenAI] = None
        self.chain_with_history: Optional[RunnableWithMessageHistory] = None
        self.store: Dict[str, ChatMessageHistory] = {}
        
        # Configuración específica del chat
        self.system_message = self.config.system_message or "Eres un asistente útil y amigable."
        
    async def initialize(self) -> None:
        """Inicializa el agente LangChain con OpenAI y memoria de conversación."""
        try:
            # Configurar el modelo LLM
            self.llm = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                openai_api_key=settings.openai_api_key
            )
            
            # Crear el prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_message),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])
            
            # Crear la cadena
            chain = prompt | self.llm
            
            # Configurar el historial
            self.chain_with_history = RunnableWithMessageHistory(
                chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )
            
            self.logger.info(f"LangChainChatAgent {self.agent_id} inicializado correctamente")
            
        except Exception as e:
            raise InitializationError(
                f"Error inicializando LangChainChatAgent: {str(e)}",
                agent_id=self.agent_id
            )
    
    async def process(self, input_data: Union[str, Dict, Message]) -> AgentResponse:
        """
        Procesa un mensaje y retorna la respuesta del agente.
        
        Args:
            input_data: Mensaje de entrada (str, dict o Message)
            
        Returns:
            AgentResponse con la respuesta del agente
        """
        try:
            # Normalizar entrada
            if isinstance(input_data, str):
                message_content = input_data
                session_id = "default"
            elif isinstance(input_data, Message):
                message_content = input_data.content
                session_id = input_data.session_id or "default"
            elif isinstance(input_data, dict):
                message_content = input_data.get("content", input_data.get("message", ""))
                session_id = input_data.get("session_id", "default")
            else:
                raise ValueError(f"Tipo de entrada no soportado: {type(input_data)}")
            
            if not message_content.strip():
                raise ValueError("El mensaje no puede estar vacío")
            
            # Procesar con LangChain
            response = self.chain_with_history.invoke(
                {"input": message_content},
                config={"configurable": {"session_id": session_id}}
            )
            
            return AgentResponse(
                content=response.content.strip(),
                metadata={
                    "session_id": session_id,
                    "model": self.config.model,
                    "system_message": self.system_message,
                    "input_length": len(message_content)
                }
            )
            
        except Exception as e:
            raise ProcessingError(
                f"Error procesando mensaje: {str(e)}",
                agent_id=self.agent_id
            )
    
    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """
        Obtiene el historial de una sesión o crea uno nuevo.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            ChatMessageHistory de la sesión
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
            self.logger.debug(f"Nueva sesión creada: {session_id}")
        return self.store[session_id]
    
    def set_system_message(self, system_message: str) -> None:
        """
        Cambia el mensaje del sistema y reinicializa el agente.
        
        Args:
            system_message: Nuevo mensaje del sistema
        """
        self.system_message = system_message
        self.config.system_message = system_message
        self.logger.info(f"Sistema actualizado para agente {self.agent_id}")
        
        # Reinicializar si ya está inicializado
        if self._is_initialized:
            import asyncio
            asyncio.create_task(self.initialize())
    
    def get_session_count(self) -> int:
        """Retorna el número de sesiones activas."""
        return len(self.store)
    
    def clear_session(self, session_id: str) -> bool:
        """
        Limpia el historial de una sesión específica.
        
        Args:
            session_id: ID de la sesión a limpiar
            
        Returns:
            True si se limpió, False si no existía
        """
        if session_id in self.store:
            del self.store[session_id]
            self.logger.info(f"Sesión {session_id} limpiada")
            return True
        return False
    
    def clear_all_sessions(self) -> int:
        """
        Limpia todas las sesiones.
        
        Returns:
            Número de sesiones que se limpiaron
        """
        count = len(self.store)
        self.store.clear()
        self.logger.info(f"Todas las sesiones limpiadas ({count} sesiones)")
        return count
    
    async def cleanup(self) -> None:
        """Limpia recursos del agente."""
        self.clear_all_sessions()
        await super().cleanup()
