"""
Tipos personalizados y modelos de datos para el framework AI Agents.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
import uuid


class AgentState(Enum):
    """Estados posibles de un agente."""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    STOPPED = "stopped"


class MessageRole(Enum):
    """Roles de mensajes en una conversación."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Modelo para mensajes en conversaciones."""
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None


class AgentResponse(BaseModel):
    """Respuesta estándar de un agente."""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    tokens_used: Optional[int] = None
    
    def is_success(self) -> bool:
        """Verifica si la respuesta fue exitosa."""
        return self.error is None


class AgentConfig(BaseModel):
    """Configuración para un agente."""
    agent_type: str
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    system_message: Optional[str] = None
    memory_enabled: bool = True
    tools_enabled: bool = False
    custom_params: Dict[str, Any] = Field(default_factory=dict)


class ConversationHistory(BaseModel):
    """Historial de conversación."""
    session_id: str
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, message: Message) -> None:
        """Añade un mensaje al historial."""
        message.session_id = self.session_id
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_messages_by_role(self, role: MessageRole) -> List[Message]:
        """Obtiene mensajes por rol."""
        return [msg for msg in self.messages if msg.role == role]
