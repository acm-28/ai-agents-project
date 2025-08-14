"""
Sistema de logging de conversaciones para análisis y debugging.
Complementa el sistema de memoria existente con logs detallados.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import uuid

from ai_agents.config.settings import settings


class ConversationLogger:
    """
    Logger especializado para conversaciones de agentes.
    
    Guarda logs detallados de cada interacción en data/logs/ para:
    - Debugging del comportamiento de agentes
    - Análisis de rendimiento
    - Monitoreo del orquestador
    - Auditoría de conversaciones
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Inicializa el logger de conversaciones.
        
        Args:
            log_dir: Directorio donde guardar los logs (opcional)
        """
        self.log_dir = Path(log_dir or settings.data_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger estándar para errores internos
        self.logger = logging.getLogger(__name__)
    
    def log_conversation_start(self, 
                             agent_id: str, 
                             session_id: str, 
                             agent_type: str,
                             user_id: Optional[str] = None,
                             metadata: Optional[Dict] = None) -> str:
        """
        Registra el inicio de una conversación.
        
        Args:
            agent_id: ID del agente
            session_id: ID de la sesión
            agent_type: Tipo de agente (chat, data_analysis, etc.)
            user_id: ID del usuario (opcional)
            metadata: Metadatos adicionales
            
        Returns:
            ID único del log de conversación
        """
        conversation_log_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        log_entry = {
            "conversation_log_id": conversation_log_id,
            "event_type": "conversation_start",
            "timestamp": timestamp.isoformat(),
            "agent_id": agent_id,
            "agent_type": agent_type,
            "session_id": session_id,
            "user_id": user_id,
            "metadata": metadata or {},
            "messages": []
        }
        
        self._save_log_entry(conversation_log_id, log_entry)
        
        self.logger.info(
            f"Conversación iniciada - Agent: {agent_id}, Session: {session_id}, "
            f"Type: {agent_type}, Log: {conversation_log_id}"
        )
        
        return conversation_log_id
    
    def log_user_message(self,
                        conversation_log_id: str,
                        message: str,
                        timestamp: Optional[datetime] = None,
                        metadata: Optional[Dict] = None) -> None:
        """
        Registra un mensaje del usuario.
        
        Args:
            conversation_log_id: ID del log de conversación
            message: Contenido del mensaje
            timestamp: Timestamp del mensaje (opcional)
            metadata: Metadatos adicionales
        """
        self._add_message_to_log(
            conversation_log_id,
            message_type="user_message",
            content=message,
            timestamp=timestamp,
            metadata=metadata
        )
    
    def log_agent_response(self,
                          conversation_log_id: str,
                          response: str,
                          processing_time_ms: Optional[float] = None,
                          model_used: Optional[str] = None,
                          timestamp: Optional[datetime] = None,
                          metadata: Optional[Dict] = None) -> None:
        """
        Registra una respuesta del agente.
        
        Args:
            conversation_log_id: ID del log de conversación
            response: Contenido de la respuesta
            processing_time_ms: Tiempo de procesamiento en milisegundos
            model_used: Modelo utilizado para la respuesta
            timestamp: Timestamp de la respuesta (opcional)
            metadata: Metadatos adicionales
        """
        response_metadata = metadata or {}
        if processing_time_ms:
            response_metadata["processing_time_ms"] = processing_time_ms
        if model_used:
            response_metadata["model_used"] = model_used
        
        self._add_message_to_log(
            conversation_log_id,
            message_type="agent_response",
            content=response,
            timestamp=timestamp,
            metadata=response_metadata
        )
    
    def log_agent_error(self,
                       conversation_log_id: str,
                       error_message: str,
                       error_type: str,
                       timestamp: Optional[datetime] = None,
                       metadata: Optional[Dict] = None) -> None:
        """
        Registra un error del agente.
        
        Args:
            conversation_log_id: ID del log de conversación
            error_message: Mensaje de error
            error_type: Tipo de error
            timestamp: Timestamp del error (opcional)
            metadata: Metadatos adicionales
        """
        error_metadata = metadata or {}
        error_metadata["error_type"] = error_type
        
        self._add_message_to_log(
            conversation_log_id,
            message_type="agent_error",
            content=error_message,
            timestamp=timestamp,
            metadata=error_metadata
        )
    
    def log_orchestrator_action(self,
                               conversation_log_id: str,
                               action: str,
                               agent_selected: Optional[str] = None,
                               reasoning: Optional[str] = None,
                               timestamp: Optional[datetime] = None,
                               metadata: Optional[Dict] = None) -> None:
        """
        Registra una acción del orquestador.
        
        Args:
            conversation_log_id: ID del log de conversación
            action: Acción realizada por el orquestador
            agent_selected: Agente seleccionado (si aplica)
            reasoning: Razonamiento para la selección
            timestamp: Timestamp de la acción (opcional)
            metadata: Metadatos adicionales
        """
        orchestrator_metadata = metadata or {}
        if agent_selected:
            orchestrator_metadata["agent_selected"] = agent_selected
        if reasoning:
            orchestrator_metadata["reasoning"] = reasoning
        
        self._add_message_to_log(
            conversation_log_id,
            message_type="orchestrator_action",
            content=action,
            timestamp=timestamp,
            metadata=orchestrator_metadata
        )
    
    def log_conversation_end(self,
                           conversation_log_id: str,
                           reason: str = "normal_end",
                           summary: Optional[Dict] = None,
                           timestamp: Optional[datetime] = None) -> None:
        """
        Registra el final de una conversación.
        
        Args:
            conversation_log_id: ID del log de conversación
            reason: Razón del fin de la conversación
            summary: Resumen de la conversación
            timestamp: Timestamp del final (opcional)
        """
        end_metadata = {
            "end_reason": reason,
            "conversation_summary": summary or {}
        }
        
        self._add_message_to_log(
            conversation_log_id,
            message_type="conversation_end",
            content=f"Conversación terminada: {reason}",
            timestamp=timestamp,
            metadata=end_metadata
        )
    
    def _add_message_to_log(self,
                           conversation_log_id: str,
                           message_type: str,
                           content: str,
                           timestamp: Optional[datetime] = None,
                           metadata: Optional[Dict] = None) -> None:
        """
        Añade un mensaje al log de conversación.
        
        Args:
            conversation_log_id: ID del log de conversación
            message_type: Tipo de mensaje
            content: Contenido del mensaje
            timestamp: Timestamp del mensaje (opcional)
            metadata: Metadatos adicionales
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        message_entry = {
            "message_id": str(uuid.uuid4()),
            "message_type": message_type,
            "timestamp": timestamp.isoformat(),
            "content": content,
            "metadata": metadata or {}
        }
        
        # Cargar log existente
        log_data = self._load_log_entry(conversation_log_id)
        if log_data:
            log_data["messages"].append(message_entry)
            log_data["last_updated"] = timestamp.isoformat()
            self._save_log_entry(conversation_log_id, log_data)
        else:
            self.logger.warning(f"No se pudo encontrar log de conversación: {conversation_log_id}")
    
    def _save_log_entry(self, conversation_log_id: str, log_data: Dict) -> None:
        """
        Guarda una entrada de log en archivo.
        
        Args:
            conversation_log_id: ID del log de conversación
            log_data: Datos del log a guardar
        """
        try:
            log_file = self._get_log_file_path(conversation_log_id)
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error guardando log {conversation_log_id}: {e}")
    
    def _load_log_entry(self, conversation_log_id: str) -> Optional[Dict]:
        """
        Carga una entrada de log desde archivo.
        
        Args:
            conversation_log_id: ID del log de conversación
            
        Returns:
            Datos del log o None si no existe
        """
        try:
            log_file = self._get_log_file_path(conversation_log_id)
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error cargando log {conversation_log_id}: {e}")
        return None
    
    def _get_log_file_path(self, conversation_log_id: str) -> Path:
        """
        Obtiene la ruta del archivo de log para una conversación.
        
        Args:
            conversation_log_id: ID del log de conversación
            
        Returns:
            Ruta del archivo de log
        """
        # Organizar por fecha para facilitar la búsqueda
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = self.log_dir / date_str
        date_dir.mkdir(exist_ok=True)
        
        return date_dir / f"{conversation_log_id}.json"
    
    def get_conversation_logs(self, 
                             agent_id: Optional[str] = None,
                             date: Optional[str] = None,
                             limit: int = 100) -> List[Dict]:
        """
        Obtiene logs de conversaciones con filtros opcionales.
        
        Args:
            agent_id: Filtrar por ID de agente (opcional)
            date: Filtrar por fecha (YYYY-MM-DD) (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de logs de conversaciones
        """
        logs = []
        
        try:
            if date:
                # Buscar en directorio específico de fecha
                date_dir = self.log_dir / date
                if date_dir.exists():
                    log_files = list(date_dir.glob("*.json"))
                else:
                    return []
            else:
                # Buscar en todos los directorios de fecha
                log_files = list(self.log_dir.glob("*/*.json"))
            
            for log_file in sorted(log_files, reverse=True)[:limit]:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    
                    # Aplicar filtro por agent_id si se especifica
                    if agent_id and log_data.get("agent_id") != agent_id:
                        continue
                    
                    logs.append(log_data)
                except Exception as e:
                    self.logger.error(f"Error leyendo log {log_file}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error obteniendo logs de conversaciones: {e}")
        
        return logs
    
    def generate_conversation_report(self, 
                                   agent_id: Optional[str] = None,
                                   date: Optional[str] = None) -> Dict[str, Any]:
        """
        Genera un reporte de las conversaciones.
        
        Args:
            agent_id: Filtrar por ID de agente (opcional)
            date: Filtrar por fecha (YYYY-MM-DD) (opcional)
            
        Returns:
            Reporte con estadísticas de conversaciones
        """
        logs = self.get_conversation_logs(agent_id=agent_id, date=date)
        
        total_conversations = len(logs)
        total_messages = 0
        total_user_messages = 0
        total_agent_responses = 0
        total_errors = 0
        agent_types = {}
        processing_times = []
        
        for log in logs:
            messages = log.get("messages", [])
            total_messages += len(messages)
            
            agent_type = log.get("agent_type", "unknown")
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
            
            for message in messages:
                msg_type = message.get("message_type")
                if msg_type == "user_message":
                    total_user_messages += 1
                elif msg_type == "agent_response":
                    total_agent_responses += 1
                    # Recopilar tiempos de procesamiento
                    proc_time = message.get("metadata", {}).get("processing_time_ms")
                    if proc_time:
                        processing_times.append(proc_time)
                elif msg_type == "agent_error":
                    total_errors += 1
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "report_generated_at": datetime.now().isoformat(),
            "filter_agent_id": agent_id,
            "filter_date": date,
            "statistics": {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "total_user_messages": total_user_messages,
                "total_agent_responses": total_agent_responses,
                "total_errors": total_errors,
                "agent_types": agent_types,
                "average_processing_time_ms": avg_processing_time,
                "success_rate": (total_agent_responses / (total_agent_responses + total_errors)) * 100 if (total_agent_responses + total_errors) > 0 else 100
            }
        }


# Instancia global del logger de conversaciones
conversation_logger = ConversationLogger()
