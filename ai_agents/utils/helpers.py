"""
Funciones de utilidad mejoradas para el framework AI Agents.
"""

import logging
import json
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


def preprocess_data(data: Union[str, Dict, List]) -> Dict[str, Any]:
    """
    Preprocesa datos de entrada para agentes.
    
    Args:
        data: Datos a preprocesar
        
    Returns:
        Dict con datos preprocesados
    """
    if isinstance(data, str):
        return {
            "type": "text",
            "content": data.strip(),
            "length": len(data),
            "processed_at": datetime.now().isoformat()
        }
    elif isinstance(data, dict):
        return {
            "type": "dict",
            "content": data,
            "keys": list(data.keys()),
            "processed_at": datetime.now().isoformat()
        }
    elif isinstance(data, list):
        return {
            "type": "list", 
            "content": data,
            "length": len(data),
            "processed_at": datetime.now().isoformat()
        }
    else:
        return {
            "type": "unknown",
            "content": str(data),
            "processed_at": datetime.now().isoformat()
        }


def log_message(message: str, level: str = "INFO", extra_data: Optional[Dict] = None) -> None:
    """
    Registra un mensaje con información adicional.
    
    Args:
        message: Mensaje a registrar
        level: Nivel de logging
        extra_data: Datos adicionales para el log
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if extra_data:
        message = f"{message} | Extra: {json.dumps(extra_data, default=str)}"
    
    logger.log(log_level, message)


def calculate_metrics(predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
    """
    Calcula métricas básicas entre predicciones y targets.
    
    Args:
        predictions: Lista de predicciones
        targets: Lista de valores objetivo
        
    Returns:
        Dict con métricas calculadas
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    if not predictions or not targets:
        return {"accuracy": 0.0, "count": 0}
    
    # Métrica básica de exactitud
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    accuracy = correct / len(predictions)
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions),
        "calculated_at": datetime.now().timestamp()
    }


def clean_text(text: str) -> str:
    """
    Limpia y normaliza texto.
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio
    """
    if not text:
        return ""
    
    # Eliminar espacios extra y normalizar
    text = " ".join(text.split())
    text = text.strip()
    
    return text


def chunk_text(text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
    """
    Divide texto en chunks con overlap.
    
    Args:
        text: Texto a dividir
        max_length: Tamaño máximo de cada chunk
        overlap: Solapamiento entre chunks
        
    Returns:
        Lista de chunks de texto
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        chunk = text[start:end]
        
        # Intentar cortar en una palabra completa
        if end < len(text):
            last_space = chunk.rfind(' ')
            if last_space > start + max_length // 2:
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks


def safe_json_parse(text: str, default: Any = None) -> Any:
    """
    Parsea JSON de manera segura.
    
    Args:
        text: Texto JSON a parsear
        default: Valor por defecto si falla el parsing
        
    Returns:
        Objeto parseado o valor por defecto
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Error parsing JSON: {e}")
        return default


def merge_dicts(*dicts: Dict) -> Dict:
    """
    Combina múltiples diccionarios.
    
    Args:
        *dicts: Diccionarios a combinar
        
    Returns:
        Diccionario combinado
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result
