"""
Agente de chat con memoria usando LangChain.
Versión migrada y mejorada del agent1_context_awareness.py
Incluye persistencia de memoria en archivos.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage

from ai_agents.core.base_agent import BaseAgent
from ai_agents.core.types import AgentResponse, Message, MessageRole, AgentConfig
from ai_agents.core.exceptions import InitializationError, ProcessingError
from ai_agents.config.settings import settings

logger = logging.getLogger(__name__)


class LangChainChatAgent(BaseAgent):
    """
    Agente de chat con memoria conversacional usando LangChain.
    
    Características:
    - Memoria persistente por sesión con guardado en archivos
    - Configuración de sistema personalizable
    - Integración con OpenAI
    - Manejo de errores robusto
    - Persistencia de memoria entre reinicios
    """
    
    def __init__(self, 
                 agent_id: Optional[str] = None,
                 config: Optional[Union[Dict, AgentConfig]] = None,
                 enable_persistence: bool = True):
        """
        Inicializa el agente de chat.
        
        Args:
            agent_id: ID único del agente
            config: Configuración del agente
            enable_persistence: Habilita el guardado persistente de memoria
        """
        super().__init__(agent_id, config)
        
        self.llm: Optional[ChatOpenAI] = None
        self.chain_with_history: Optional[RunnableWithMessageHistory] = None
        self.store: Dict[str, ChatMessageHistory] = {}
        self.enable_persistence = enable_persistence
        
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
            
            # Cargar memorias persistentes si están habilitadas
            if self.enable_persistence:
                await self._load_all_sessions_from_files()
            
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
                message_content = input_data.get("content", input_data.get("message", input_data.get("query", "")))
                session_id = input_data.get("session_id", "default")
                
                # Si no hay mensaje pero hay una acción, generar mensaje basado en la acción
                if not message_content.strip() and "action" in input_data:
                    message_content = self._generate_message_from_action(input_data)
            else:
                raise ValueError(f"Tipo de entrada no soportado: {type(input_data)}")
            
            if not message_content.strip():
                raise ValueError("El mensaje no puede estar vacío")
            
            # Iniciar logging de conversación si no está activo
            if not self._current_conversation_log_id:
                self.start_conversation_log(session_id)
            
            # Log del mensaje del usuario
            self.log_user_message(message_content)
            
            # Procesar con LangChain
            start_time = time.time()
            response = self.chain_with_history.invoke(
                {"input": message_content},
                config={"configurable": {"session_id": session_id}}
            )
            processing_time_ms = (time.time() - start_time) * 1000
            
            response_content = response.content.strip()
            
            # Log de la respuesta del agente
            self.log_agent_response(
                response=response_content,
                processing_time_ms=processing_time_ms,
                model_used=self.config.model,
                metadata={
                    "session_id": session_id,
                    "system_message": self.system_message,
                    "input_length": len(message_content)
                }
            )
            
            # Guardar sesión en archivo si la persistencia está habilitada
            if self.enable_persistence:
                await self._save_session_to_file(session_id)
            
            return AgentResponse(
                content=response_content,
                metadata={
                    "session_id": session_id,
                    "model": self.config.model,
                    "system_message": self.system_message,
                    "input_length": len(message_content),
                    "processing_time_ms": processing_time_ms
                }
            )
            
        except Exception as e:
            error_msg = f"Error procesando mensaje: {str(e)}"
            
            # Log del error
            self.log_agent_error(
                error_message=error_msg,
                error_type=type(e).__name__
            )
            
            raise ProcessingError(error_msg, agent_id=self.agent_id)
    
    def _generate_message_from_action(self, input_data: dict) -> str:
        """
        Genera un mensaje de texto basado en la acción y los datos de entrada.
        
        Args:
            input_data: Diccionario con datos de entrada incluyendo 'action'
            
        Returns:
            Mensaje de texto generado
        """
        action = input_data.get("action", "")
        dataset = input_data.get("dataset", "")
        analysis_type = input_data.get("analysis_type", "")
        user_id = input_data.get("user_id", "")
        
        # Generar mensaje basado en la acción
        if action == "validate_results":
            if dataset and analysis_type:
                return f"Por favor valida los resultados del análisis {analysis_type} realizado en el dataset {dataset}. Revisa la calidad, coherencia y exactitud de los resultados obtenidos."
            else:
                return "Por favor valida los resultados del análisis realizado. Revisa la calidad, coherencia y exactitud de los resultados obtenidos."
        
        elif action == "summarize_analysis":
            if dataset:
                return f"Genera un resumen ejecutivo del análisis realizado en el dataset {dataset}. Incluye los hallazgos principales, conclusiones y recomendaciones."
            else:
                return "Genera un resumen ejecutivo del análisis realizado. Incluye los hallazgos principales, conclusiones y recomendaciones."
        
        elif action == "review_quality":
            return "Realiza una revisión de calidad de los resultados del análisis. Identifica posibles problemas o áreas de mejora."
        
        elif action == "generate_insights":
            return "Genera insights y recomendaciones basados en los resultados del análisis de datos."
        
        else:
            # Acción genérica
            action_desc = action.replace("_", " ").title()
            if dataset:
                return f"Realiza la acción '{action_desc}' para el dataset {dataset}."
            else:
                return f"Realiza la acción '{action_desc}'."
    
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
    
    def _get_memory_dir_path(self) -> Path:
        """Obtiene la ruta del directorio de memoria para este agente."""
        memory_dir = Path(settings.memory_dir) / self.agent_id
        memory_dir.mkdir(parents=True, exist_ok=True)
        return memory_dir
    
    def _get_session_file_path(self, session_id: str) -> Path:
        """Obtiene la ruta del archivo de memoria para una sesión específica."""
        memory_dir = self._get_memory_dir_path()
        return memory_dir / f"{session_id}_session.json"
    
    async def _save_session_to_file(self, session_id: str) -> None:
        """
        Guarda una sesión específica en archivo JSON.
        
        Args:
            session_id: ID de la sesión a guardar
        """
        if not self.enable_persistence or session_id not in self.store:
            return
        
        session_file = self._get_session_file_path(session_id)
        
        try:
            # Preparar datos para guardar
            session_history = self.store[session_id]
            messages_data = []
            
            for msg in session_history.messages:
                if isinstance(msg, HumanMessage):
                    messages_data.append({
                        "type": "human",
                        "content": msg.content,
                        "timestamp": str(msg.additional_kwargs.get('timestamp', ''))
                    })
                elif isinstance(msg, AIMessage):
                    messages_data.append({
                        "type": "ai", 
                        "content": msg.content,
                        "timestamp": str(msg.additional_kwargs.get('timestamp', ''))
                    })
            
            session_data = {
                "agent_id": self.agent_id,
                "session_id": session_id,
                "total_messages": len(messages_data),
                "messages": messages_data,
                "system_message": self.system_message,
                "last_updated": str(Path().resolve())  # timestamp placeholder
            }
            
            # Guardar archivo
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"Sesión {session_id} guardada: {len(messages_data)} mensajes")
            
        except Exception as e:
            self.logger.error(f"Error guardando sesión {session_id} en archivo: {e}")
    
    async def _load_session_from_file(self, session_id: str) -> bool:
        """
        Carga una sesión específica desde archivo JSON.
        
        Args:
            session_id: ID de la sesión a cargar
            
        Returns:
            True si se cargó exitosamente, False en caso contrario
        """
        if not self.enable_persistence:
            return False
        
        session_file = self._get_session_file_path(session_id)
        
        if not session_file.exists():
            self.logger.debug(f"No se encontró archivo de sesión para {session_id}")
            return False
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Crear nueva historia de chat
            chat_history = ChatMessageHistory()
            
            # Restaurar mensajes en la memoria
            for msg_data in session_data.get('messages', []):
                if msg_data['type'] == 'human':
                    chat_history.add_user_message(msg_data['content'])
                elif msg_data['type'] == 'ai':
                    chat_history.add_ai_message(msg_data['content'])
            
            # Almacenar en el store
            self.store[session_id] = chat_history
            
            self.logger.info(f"Sesión {session_id} cargada: {len(session_data.get('messages', []))} mensajes")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando sesión {session_id} desde archivo: {e}")
            return False
    
    async def _load_all_sessions_from_files(self) -> None:
        """Carga todas las sesiones guardadas desde archivos."""
        if not self.enable_persistence:
            return
        
        memory_dir = self._get_memory_dir_path()
        
        if not memory_dir.exists():
            self.logger.info(f"No se encontró directorio de memoria para agente {self.agent_id}")
            return
        
        session_files = list(memory_dir.glob("*_session.json"))
        loaded_count = 0
        
        for session_file in session_files:
            # Extraer session_id del nombre del archivo
            session_id = session_file.stem.replace('_session', '')
            
            if await self._load_session_from_file(session_id):
                loaded_count += 1
        
        self.logger.info(f"Agente {self.agent_id}: {loaded_count} sesiones cargadas desde archivos")
    
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
            
            # Eliminar archivo de persistencia si existe
            if self.enable_persistence:
                session_file = self._get_session_file_path(session_id)
                if session_file.exists():
                    try:
                        session_file.unlink()
                        self.logger.debug(f"Archivo de sesión {session_id} eliminado")
                    except Exception as e:
                        self.logger.error(f"Error eliminando archivo de sesión {session_id}: {e}")
            
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
        session_ids = list(self.store.keys())
        
        # Limpiar todas las sesiones (esto también eliminará los archivos)
        for session_id in session_ids:
            self.clear_session(session_id)
        
        self.logger.info(f"Todas las sesiones limpiadas ({count} sesiones)")
        return count
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene un resumen de una sesión específica.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Diccionario con estadísticas de la sesión o None si no existe
        """
        if session_id not in self.store:
            return None
        
        messages = self.store[session_id].messages
        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        
        return {
            "session_id": session_id,
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "ai_messages": len(ai_messages),
            "conversation_length": sum(len(msg.content) for msg in messages),
            "last_question": user_messages[-1].content if user_messages else None,
            "last_answer": ai_messages[-1].content if ai_messages else None
        }
    
    def get_all_sessions_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene un resumen de todas las sesiones activas.
        
        Returns:
            Diccionario con resúmenes de todas las sesiones
        """
        summaries = {}
        for session_id in self.store.keys():
            summaries[session_id] = self.get_session_summary(session_id)
        return summaries
    
    def get_session_memory_size(self, session_id: str) -> int:
        """
        Retorna el número de mensajes en memoria para una sesión específica.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Número de mensajes en la sesión
        """
        if session_id in self.store:
            return len(self.store[session_id].messages)
        return 0
    
    async def cleanup(self) -> None:
        """Limpia recursos del agente."""
        self.clear_all_sessions()
        await super().cleanup()
    
    async def process_request(self, input_data: Union[str, Dict, Message]) -> AgentResponse:
        """
        Interfaz pública para procesar requests.
        Delega al método process con manejo seguro.
        
        Args:
            input_data: Datos de entrada
            
        Returns:
            AgentResponse con la respuesta del agente
        """
        return await self._safe_process(input_data)
