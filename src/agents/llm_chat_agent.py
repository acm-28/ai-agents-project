import openai
import os
from dotenv import load_dotenv
from agents.base_agent import BaseAgent

class LLMChatAgent(BaseAgent):
    def initialize(self):
        """Carga la clave de API y configura el cliente de OpenAI."""
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No se encontró la clave de API de OpenAI. Asegúrate de configurarla en el archivo .env.")
        self.client = openai.OpenAI(api_key=self.api_key)
        print("LLMChatAgent inicializado y listo para usar OpenAI.")

    def respond(self, message):
        """Envía el mensaje al modelo de lenguaje y devuelve la respuesta."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # O "gpt-4" si tienes acceso
                messages=[
                    {"role": "system", "content": "Eres un asistente útil y amigable."},
                    {"role": "user", "content": message},
                ],
                max_tokens=150,
                temperature=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Hubo un error al procesar tu mensaje: {e}"