"""
Agente de Q&A con memoria conversacional.
Versión migrada y mejorada del agent2_qa.py
"""

import logging
from typing import Dict, Any, Optional, Union, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

from ai_agents.core.base_agent import BaseAgent
from ai_agents.core.types import AgentResponse, Message, MessageRole, AgentConfig
from ai_agents.core.exceptions import InitializationError, ProcessingError
from ai_agents.config.settings import settings

logger = logging.getLogger(__name__)


class MemoryQAAgent(BaseAgent):
    """
    Agente de pregunta-respuesta con memoria conversacional.
    
    Características:
    - Memoria de conversación con buffer
    - Contexto histórico en las respuestas
    - Optimizado para sesiones de Q&A
    - Formateo inteligente del historial
    """
    
    def __init__(self, 
                 agent_id: Optional[str] = None,
                 config: Optional[Union[Dict, AgentConfig]] = None):
        """
        Inicializa el agente de Q&A.
        
        Args:
            agent_id: ID único del agente
            config: Configuración del agente
        """
        super().__init__(agent_id, config)
        
        self.llm: Optional[ChatOpenAI] = None
        self.memory: Optional[ConversationBufferMemory] = None
        self.prompt_template: Optional[PromptTemplate] = None
        
    async def initialize(self) -> None:
        """Inicializa el agente Q&A con memoria."""
        try:
            # Configurar el modelo LLM
            self.llm = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                openai_api_key=settings.openai_api_key
            )
            
            # Inicializar memoria conversacional
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
            
            # Configurar template de prompt
            self._setup_prompt_template()
            
            self.logger.info(f"MemoryQAAgent {self.agent_id} inicializado correctamente")
            
        except Exception as e:
            raise InitializationError(
                f"Error inicializando MemoryQAAgent: {str(e)}",
                agent_id=self.agent_id
            )
    
    def _setup_prompt_template(self) -> None:
        """Configura el template de prompt para el agente Q&A."""
        template = """Eres un asistente de IA especializado en responder preguntas de manera clara y precisa.

Contexto de la conversación:
{chat_history}

Pregunta actual: {question}

Instrucciones:
- Proporciona respuestas claras y concisas
- Usa el contexto de la conversación cuando sea relevante
- Si no sabes algo, admítelo honestamente
- Mantén un tono profesional y amigable

Respuesta:"""
        
        self.prompt_template = PromptTemplate(
            template=template,
            input_variables=["chat_history", "question"]
        )
    
    async def process(self, input_data: Union[str, Dict, Message]) -> AgentResponse:
        """
        Procesa una pregunta y retorna la respuesta del agente.
        
        Args:
            input_data: Pregunta de entrada
            
        Returns:
            AgentResponse con la respuesta del agente
        """
        try:
            # Normalizar entrada
            if isinstance(input_data, str):
                question = input_data
            elif isinstance(input_data, Message):
                question = input_data.content
            elif isinstance(input_data, dict):
                question = input_data.get("content", input_data.get("question", ""))
            else:
                raise ValueError(f"Tipo de entrada no soportado: {type(input_data)}")
            
            if not question.strip():
                raise ValueError("La pregunta no puede estar vacía")
            
            # Obtener historial de memoria
            chat_history = self.memory.chat_memory.messages
            history_text = self._format_chat_history(chat_history)
            
            # Crear prompt formateado
            formatted_prompt = self.prompt_template.format(
                chat_history=history_text,
                question=question
            )
            
            # Obtener respuesta del modelo
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            answer = response.content.strip()
            
            # Guardar la interacción en memoria
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            
            return AgentResponse(
                content=answer,
                metadata={
                    "question": question,
                    "model": self.config.model,
                    "history_length": len(chat_history),
                    "input_length": len(question)
                }
            )
            
        except Exception as e:
            raise ProcessingError(
                f"Error procesando pregunta: {str(e)}",
                agent_id=self.agent_id
            )
    
    def _format_chat_history(self, messages: List) -> str:
        """
        Formatea el historial de chat para el prompt.
        
        Args:
            messages: Lista de mensajes del historial
            
        Returns:
            String formateado del historial
        """
        if not messages:
            return "No hay conversación previa."
        
        formatted_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append(f"Usuario: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_messages.append(f"Asistente: {message.content}")
        
        return "\n".join(formatted_messages)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de la conversación actual.
        
        Returns:
            Diccionario con estadísticas de la conversación
        """
        messages = self.memory.chat_memory.messages
        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        
        return {
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "ai_messages": len(ai_messages),
            "conversation_length": sum(len(msg.content) for msg in messages),
            "last_question": user_messages[-1].content if user_messages else None,
            "last_answer": ai_messages[-1].content if ai_messages else None
        }
    
    def clear_memory(self) -> None:
        """Limpia la memoria conversacional."""
        if self.memory:
            self.memory.clear()
            self.logger.info(f"Memoria limpiada para agente {self.agent_id}")
    
    def get_memory_size(self) -> int:
        """Retorna el número de mensajes en memoria."""
        if self.memory:
            return len(self.memory.chat_memory.messages)
        return 0
    
    async def cleanup(self) -> None:
        """Limpia recursos del agente."""
        self.clear_memory()
        await super().cleanup()
