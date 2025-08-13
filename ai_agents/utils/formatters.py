"""
Formateadores para el framework AI Agents.
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from ai_agents.core.types import AgentResponse, Message, ConversationHistory


def format_response(response: AgentResponse, include_metadata: bool = True) -> str:
    """
    Formatea una respuesta de agente para mostrar.
    
    Args:
        response: Respuesta a formatear
        include_metadata: Si incluir metadata en el formato
        
    Returns:
        String formateado
    """
    lines = []
    
    # Contenido principal
    lines.append(f"Contenido: {response.content}")
    
    if include_metadata:
        lines.append(f"Agente: {response.agent_id or 'Desconocido'}")
        lines.append(f"Timestamp: {response.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if response.processing_time:
            lines.append(f"Tiempo de procesamiento: {response.processing_time:.2f}s")
        
        if response.tokens_used:
            lines.append(f"Tokens utilizados: {response.tokens_used}")
        
        if response.error:
            lines.append(f"Error: {response.error}")
        
        if response.metadata:
            lines.append(f"Metadata: {json.dumps(response.metadata, indent=2)}")
    
    return "\n".join(lines)


def format_conversation(history: ConversationHistory, max_messages: Optional[int] = None) -> str:
    """
    Formatea historial de conversación.
    
    Args:
        history: Historial de conversación
        max_messages: Número máximo de mensajes a mostrar
        
    Returns:
        String formateado del historial
    """
    lines = []
    lines.append(f"=== Conversación (ID: {history.session_id}) ===")
    lines.append(f"Creada: {history.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Actualizada: {history.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Mensajes: {len(history.messages)}")
    lines.append("")
    
    messages = history.messages
    if max_messages and len(messages) > max_messages:
        messages = messages[-max_messages:]
        lines.append(f"Mostrando los últimos {max_messages} mensajes...")
        lines.append("")
    
    for i, message in enumerate(messages, 1):
        role_symbol = {
            "user": "👤",
            "assistant": "🤖", 
            "system": "⚙️"
        }.get(message.role.value, "❓")
        
        lines.append(f"{i}. {role_symbol} {message.role.value.upper()}:")
        lines.append(f"   {message.content}")
        lines.append(f"   ⏰ {message.timestamp.strftime('%H:%M:%S')}")
        lines.append("")
    
    return "\n".join(lines)


def format_agent_info(agent_info: Dict[str, Any]) -> str:
    """
    Formatea información de agente.
    
    Args:
        agent_info: Información del agente
        
    Returns:
        String formateado
    """
    lines = []
    lines.append(f"=== Información del Agente ===")
    lines.append(f"ID: {agent_info.get('agent_id', 'N/A')}")
    lines.append(f"Tipo: {agent_info.get('agent_type', 'N/A')}")
    lines.append(f"Estado: {agent_info.get('state', 'N/A')}")
    lines.append(f"Inicializado: {'Sí' if agent_info.get('is_initialized') else 'No'}")
    
    if agent_info.get('created_at'):
        lines.append(f"Creado: {agent_info['created_at']}")
    
    if agent_info.get('initialization_time'):
        lines.append(f"Tiempo de inicialización: {agent_info['initialization_time']:.2f}s")
    
    # Configuración
    config = agent_info.get('config', {})
    if config:
        lines.append("")
        lines.append("Configuración:")
        for key, value in config.items():
            lines.append(f"  {key}: {value}")
    
    # Metadata
    metadata = agent_info.get('metadata', {})
    if metadata:
        lines.append("")
        lines.append("Metadata:")
        for key, value in metadata.items():
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)


def format_error(error: Exception, include_traceback: bool = False) -> str:
    """
    Formatea error para mostrar.
    
    Args:
        error: Excepción a formatear
        include_traceback: Si incluir traceback completo
        
    Returns:
        String formateado del error
    """
    lines = []
    lines.append(f"❌ Error: {error.__class__.__name__}")
    lines.append(f"Mensaje: {str(error)}")
    
    # Si es AgentError, incluir información adicional
    if hasattr(error, 'agent_id') and error.agent_id:
        lines.append(f"Agente: {error.agent_id}")
    
    if hasattr(error, 'details') and error.details:
        lines.append(f"Detalles: {json.dumps(error.details, indent=2)}")
    
    if include_traceback:
        import traceback
        lines.append("")
        lines.append("Traceback:")
        lines.append(traceback.format_exc())
    
    return "\n".join(lines)


def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    Formatea métricas para mostrar.
    
    Args:
        metrics: Diccionario de métricas
        
    Returns:
        String formateado
    """
    lines = []
    lines.append("=== Métricas ===")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        elif isinstance(value, int):
            lines.append(f"{key}: {value:,}")
        else:
            lines.append(f"{key}: {value}")
    
    return "\n".join(lines)


def format_table(data: List[Dict[str, Any]], headers: Optional[List[str]] = None) -> str:
    """
    Formatea lista de diccionarios como tabla.
    
    Args:
        data: Lista de diccionarios con datos
        headers: Headers opcionales para la tabla
        
    Returns:
        String formateado como tabla
    """
    if not data:
        return "No hay datos para mostrar"
    
    # Obtener headers si no se proporcionan
    if not headers:
        headers = list(data[0].keys())
    
    # Calcular anchos de columnas
    widths = {}
    for header in headers:
        widths[header] = max(
            len(str(header)),
            max(len(str(row.get(header, ""))) for row in data)
        )
    
    # Crear tabla
    lines = []
    
    # Header
    header_line = " | ".join(str(header).ljust(widths[header]) for header in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # Datos
    for row in data:
        row_line = " | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers)
        lines.append(row_line)
    
    return "\n".join(lines)
