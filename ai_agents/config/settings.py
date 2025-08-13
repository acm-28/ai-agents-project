"""
Configuración centralizada del proyecto AI Agents.

Este módulo maneja toda la configuración del proyecto usando Pydantic Settings
para validación de tipos y carga automática desde variables de entorno.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Configuración centralizada del proyecto."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # API Keys
    openai_api_key: str = Field(..., description="OpenAI API Key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API Key")
    
    # Configuración de modelos
    default_model: str = Field("gpt-3.5-turbo", description="Modelo por defecto")
    max_tokens: int = Field(1000, description="Máximo número de tokens")
    temperature: float = Field(0.7, description="Temperatura del modelo")
    
    # Configuración de memoria
    memory_backend: str = Field("local", description="Backend de memoria")
    redis_url: Optional[str] = Field(None, description="URL de Redis")
    
    # Configuración de logging
    log_level: str = Field("INFO", description="Nivel de logging")
    log_file: str = Field("agents.log", description="Archivo de logs")
    
    # Configuración de datos
    data_dir: str = Field("data", description="Directorio de datos")
    cache_dir: str = Field("data/cache", description="Directorio de cache")

# Instancia global de configuración
settings = Settings()
