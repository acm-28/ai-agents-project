"""
Excepciones personalizadas para el framework AI Agents.
"""

class AgentError(Exception):
    """Excepción base para errores de agentes."""
    
    def __init__(self, message: str, agent_id: str = None, details: dict = None):
        super().__init__(message)
        self.agent_id = agent_id
        self.details = details or {}


class InitializationError(AgentError):
    """Error durante la inicialización del agente."""
    pass


class ProcessingError(AgentError):
    """Error durante el procesamiento."""
    pass


class MemoryError(AgentError):
    """Error en operaciones de memoria."""
    pass


class ConfigurationError(AgentError):
    """Error de configuración."""
    pass


class ModelError(AgentError):
    """Error relacionado con el modelo de IA."""
    pass


class ValidationError(AgentError):
    """Error de validación de datos."""
    pass
