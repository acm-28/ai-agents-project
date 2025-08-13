"""
Validadores para el framework AI Agents.
"""

import re
from typing import Any, Dict, List, Optional, Union
from ai_agents.core.types import Message, AgentConfig, MessageRole
from ai_agents.core.exceptions import ValidationError


def validate_message(message: Union[str, Dict, Message]) -> Message:
    """
    Valida y convierte entrada a objeto Message.
    
    Args:
        message: Mensaje a validar
        
    Returns:
        Objeto Message validado
        
    Raises:
        ValidationError: Si el mensaje no es válido
    """
    if isinstance(message, Message):
        return message
    
    if isinstance(message, str):
        if not message.strip():
            raise ValidationError("El mensaje no puede estar vacío")
        
        return Message(
            role=MessageRole.USER,
            content=message.strip()
        )
    
    if isinstance(message, dict):
        required_fields = {"role", "content"}
        if not required_fields.issubset(message.keys()):
            raise ValidationError(f"El mensaje debe contener los campos: {required_fields}")
        
        try:
            return Message(**message)
        except Exception as e:
            raise ValidationError(f"Error creando Message desde dict: {e}")
    
    raise ValidationError(f"Tipo de mensaje no soportado: {type(message)}")


def validate_config(config: Union[Dict, AgentConfig]) -> AgentConfig:
    """
    Valida configuración de agente.
    
    Args:
        config: Configuración a validar
        
    Returns:
        Objeto AgentConfig validado
        
    Raises:
        ValidationError: Si la configuración no es válida
    """
    if isinstance(config, AgentConfig):
        return config
    
    if isinstance(config, dict):
        try:
            return AgentConfig(**config)
        except Exception as e:
            raise ValidationError(f"Error creando AgentConfig: {e}")
    
    raise ValidationError(f"Tipo de configuración no soportado: {type(config)}")


def validate_email(email: str) -> bool:
    """
    Valida formato de email.
    
    Args:
        email: Email a validar
        
    Returns:
        True si el email es válido
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_api_key(api_key: str, provider: str = "openai") -> bool:
    """
    Valida formato de API key.
    
    Args:
        api_key: API key a validar
        provider: Proveedor de la API key
        
    Returns:
        True si la API key tiene formato válido
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    if provider.lower() == "openai":
        # OpenAI keys start with 'sk-'
        return api_key.startswith("sk-") and len(api_key) > 10
    elif provider.lower() == "anthropic":
        # Anthropic keys start with 'sk-ant-'
        return api_key.startswith("sk-ant-") and len(api_key) > 15
    
    # Validación genérica
    return len(api_key) > 8


def validate_model_name(model: str) -> bool:
    """
    Valida nombre de modelo.
    
    Args:
        model: Nombre del modelo a validar
        
    Returns:
        True si el modelo es válido
    """
    valid_models = {
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo", 
        "claude-3-haiku",
        "claude-3-sonnet",
        "claude-3-opus"
    }
    
    return model in valid_models


def validate_temperature(temperature: float) -> bool:
    """
    Valida valor de temperature.
    
    Args:
        temperature: Valor a validar
        
    Returns:
        True si el valor es válido
    """
    return 0.0 <= temperature <= 2.0


def validate_max_tokens(max_tokens: int) -> bool:
    """
    Valida valor de max_tokens.
    
    Args:
        max_tokens: Valor a validar
        
    Returns:
        True si el valor es válido
    """
    return 1 <= max_tokens <= 32000


def validate_session_id(session_id: str) -> bool:
    """
    Valida formato de session ID.
    
    Args:
        session_id: Session ID a validar
        
    Returns:
        True si el session ID es válido
    """
    if not session_id or not isinstance(session_id, str):
        return False
    
    # Verificar que sea un UUID válido o string alfanumérico
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    alphanumeric_pattern = r'^[a-zA-Z0-9_-]{8,}$'
    
    return bool(re.match(uuid_pattern, session_id)) or bool(re.match(alphanumeric_pattern, session_id))
