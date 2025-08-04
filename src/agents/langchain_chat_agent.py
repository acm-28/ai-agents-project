import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent

class LangChainChatAgent(BaseAgent):
    def __init__(self):
        self.llm = None
        self.system_message = "Eres un asistente útil y amigable."
    
    def initialize(self):
        """Inicializa el agente LangChain con OpenAI."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("No se encontró OPENAI_API_KEY en las variables de entorno.")
        
        # Crear el modelo LangChain
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=150,
            openai_api_key=api_key
        )
        
        print("LangChainChatAgent inicializado correctamente.")
    
    def respond(self, message: str) -> str:
        """Responde a un mensaje usando LangChain."""
        try:
            # Crear los mensajes para el chat
            messages = [
                SystemMessage(content=self.system_message),
                HumanMessage(content=message)
            ]
            
            # Invocar el modelo
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            return f"Error al procesar el mensaje: {str(e)}"
    
    def set_system_message(self, system_message: str):
        """Permite cambiar el mensaje del sistema."""
        self.system_message = system_message
