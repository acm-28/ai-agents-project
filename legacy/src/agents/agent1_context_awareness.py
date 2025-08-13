import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from agents.base_agent import BaseAgent

class LangChainChatAgent(BaseAgent):
    def __init__(self):
        self.llm = None
        self.system_message = "Eres un asistente útil y amigable."
        self.chain_with_history = None
        self.store = {}

    def initialize(self):
        """Inicializa el agente LangChain con OpenAI y memoria de conversación."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("No se encontró OPENAI_API_KEY en las variables de entorno.")

        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=150,
            openai_api_key=api_key
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        chain = prompt | self.llm

        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        print("LangChainChatAgent con memoria inicializado correctamente.")

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """Obtiene el historial de una sesión o crea uno nuevo."""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def respond(self, message: str, session_id: str) -> str:
        """Responde a un mensaje usando LangChain y el historial de la sesión."""
        try:
            response = self.chain_with_history.invoke(
                {"input": message},
                config={"configurable": {"session_id": session_id}}
            )
            return response.content.strip()
        except Exception as e:
            return f"Error al procesar el mensaje: {str(e)}"

    def set_system_message(self, system_message: str):
        """Permite cambiar el mensaje del sistema."""
        self.system_message = system_message
        # Es necesario reinicializar para que el prompt se actualice
        self.initialize()
