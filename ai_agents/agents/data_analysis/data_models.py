"""
Modelos de datos y estados para el agente de análisis de datos.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
from enum import Enum


class AnalysisType(Enum):
    """Tipos de análisis disponibles."""
    DESCRIPTIVE = "descriptive"
    STATISTICAL = "statistical"
    VISUALIZATION = "visualization"
    QUERY = "query"
    COMPARISON = "comparison"
    TREND = "trend"


class DataAnalysisState(BaseModel):
    """Estado del análisis de datos para workflows."""
    
    # Entrada del usuario
    query: str = Field(..., description="Consulta del usuario")
    session_id: str = Field(default="default", description="ID de sesión")
    
    # Información del dataset
    dataset_info: Dict[str, Any] = Field(default_factory=dict, description="Información del dataset")
    dataframe_id: Optional[str] = Field(None, description="ID del DataFrame cargado")
    
    # Estado del análisis
    analysis_type: Optional[AnalysisType] = Field(None, description="Tipo de análisis detectado")
    analysis_result: str = Field(default="", description="Resultado del análisis")
    
    # Contexto conversacional
    conversation_history: List[str] = Field(default_factory=list, description="Historial de conversación")
    
    # Resultado final
    final_response: str = Field(default="", description="Respuesta final formateada")
    
    # Metadatos
    dataframe: Optional[Any] = Field(None, description="DataFrame actual", exclude=True)
    pandas_agent: Optional[Any] = Field(None, description="Agente pandas", exclude=True)
    error_message: Optional[str] = Field(None, description="Mensaje de error si existe")
    
    class Config:
        arbitrary_types_allowed = True


class DatasetInfo(BaseModel):
    """Información sobre un dataset."""
    
    name: str = Field(..., description="Nombre del dataset")
    shape: tuple = Field(..., description="Forma del dataset (filas, columnas)")
    columns: List[str] = Field(..., description="Lista de columnas")
    dtypes: Dict[str, str] = Field(..., description="Tipos de datos por columna")
    memory_usage: float = Field(..., description="Uso de memoria en MB")
    null_counts: Dict[str, int] = Field(..., description="Conteo de valores nulos")
    created_at: datetime = Field(default_factory=datetime.now, description="Fecha de creación")
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str = "dataset") -> "DatasetInfo":
        """Crea DatasetInfo desde un DataFrame de pandas."""
        return cls(
            name=name,
            shape=df.shape,
            columns=df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            memory_usage=df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            null_counts=df.isnull().sum().to_dict()
        )


class AnalysisRequest(BaseModel):
    """Solicitud de análisis de datos."""
    
    query: str = Field(..., description="Consulta del usuario")
    session_id: str = Field(default="default", description="ID de sesión")
    dataset_id: Optional[str] = Field(None, description="ID del dataset a usar")
    analysis_type: Optional[AnalysisType] = Field(None, description="Tipo de análisis específico")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parámetros adicionales")


class AnalysisResult(BaseModel):
    """Resultado de un análisis de datos."""
    
    query: str = Field(..., description="Consulta original")
    analysis_type: AnalysisType = Field(..., description="Tipo de análisis realizado")
    result: str = Field(..., description="Resultado textual")
    data: Optional[Dict[str, Any]] = Field(None, description="Datos estructurados del resultado")
    visualizations: List[str] = Field(default_factory=list, description="URLs/paths de visualizaciones")
    execution_time: float = Field(..., description="Tiempo de ejecución en segundos")
    tokens_used: Optional[int] = Field(None, description="Tokens utilizados")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadatos adicionales")
    
    def is_success(self) -> bool:
        """Verifica si el análisis fue exitoso."""
        return bool(self.result and not self.result.startswith("Error"))


class DataSource(BaseModel):
    """Fuente de datos."""
    
    source_type: str = Field(..., description="Tipo de fuente (csv, json, database, etc.)")
    path: Optional[str] = Field(None, description="Ruta del archivo")
    connection_string: Optional[str] = Field(None, description="String de conexión para BD")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parámetros de carga")
    
    def validate_source(self) -> bool:
        """Valida que la fuente de datos sea accesible."""
        # Implementar validación específica por tipo
        return True
