from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Clase base abstracta para todos los agentes usando LangChain."""
    
    @abstractmethod
    def initialize(self):
        """Inicializa el agente."""
        pass
    
    @abstractmethod
    def respond(self, message: str) -> str:
        """Responde a un mensaje del usuario."""
        pass
    
    def act(self):
        """Realiza una acción (para agentes más complejos)."""
        pass
    
    def learn(self):
        """Aprende de la experiencia (para agentes con memoria)."""
        pass