import os
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from agents.base_agent import BaseAgent

class QAAgent(BaseAgent):
    """
    Simple Question-Answering Agent with memory capabilities.
    This agent can understand user queries and provide relevant, concise answers
    while maintaining context from previous interactions.
    """
    
    def __init__(self):
        self.llm = None
        self.memory = None
        self.prompt_template = None
        
    def initialize(self):
        """Initialize the QA agent with OpenAI model and memory."""
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError('No se encontró OPENAI_API_KEY')
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            max_tokens=1000, 
            temperature=0.7,
            openai_api_key=api_key
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Define the prompt template
        self._setup_prompt_template()
        
        print('QAAgent inicializado correctamente')
    
    def _setup_prompt_template(self):
        """Setup the prompt template for the QA agent."""
        template = """
Eres un asistente de IA útil y conversacional. Tu tarea es responder las preguntas del usuario de la mejor manera posible.

Historial de la conversación:
{chat_history}

Pregunta del usuario: {question}

Por favor, proporciona una respuesta clara y concisa:
"""
        
        self.prompt_template = PromptTemplate(
            template=template,
            input_variables=["chat_history", "question"]
        )
    
    def get_answer(self, question: str) -> str:
        """
        Get an answer to the given question using the QA chain with memory.
        
        Args:
            question (str): The user's question
            
        Returns:
            str: The agent's response
        """
        try:
            # Get chat history from memory
            chat_history = self.memory.chat_memory.messages
            
            # Format chat history for the prompt
            history_text = self._format_chat_history(chat_history)
            
            # Create the prompt with current question and history
            formatted_prompt = self.prompt_template.format(
                chat_history=history_text,
                question=question
            )
            
            # Get response from the language model
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            answer = response.content
            
            # Save the interaction to memory
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            
            return answer
            
        except Exception as e:
            return f'Error al procesar la pregunta: {str(e)}'
    
    def _format_chat_history(self, messages: List) -> str:
        """Format chat history for display in the prompt."""
        if not messages:
            return "No hay historial previo."
        
        formatted_history = []
        for message in messages[-10:]:  # Limit to last 10 messages to avoid token limits
            if isinstance(message, HumanMessage):
                formatted_history.append(f"Usuario: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Asistente: {message.content}")
        
        return "\n".join(formatted_history)
    
    def respond(self, message: str) -> str:
        """
        Main interface method that processes user input and returns response.
        
        Args:
            message (str): User's input message/question
            
        Returns:
            str: Agent's response
        """
        return self.get_answer(message)
    
    def clear_memory(self):
        """Clear the conversation memory."""
        if self.memory:
            self.memory.clear()
            print("Memoria de conversación limpiada.")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the current conversation history.
        
        Returns:
            List[Dict]: List of conversation messages
        """
        history = []
        messages = self.memory.chat_memory.messages
        
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        
        return history
    
    def set_system_context(self, context: str):
        """
        Set additional system context for the agent.
        
        Args:
            context (str): Additional context or instructions for the agent
        """
        template = f"""
Eres un asistente de IA útil y conversacional. Tu tarea es responder las preguntas del usuario de la mejor manera posible.

Contexto adicional: {context}

Historial de la conversación:
{{chat_history}}

Pregunta del usuario: {{question}}

Por favor, proporciona una respuesta clara y concisa:
"""
        
        self.prompt_template = PromptTemplate(
            template=template,
            input_variables=["chat_history", "question"]
        )


def run_qa_agent_interactive():
    """
    Run the QA agent in interactive mode for testing.
    """
    agent = QAAgent()
    agent.initialize()
    
    print("Agente QA con memoria iniciado. Escribe 'salir' para terminar.")
    print("Comandos especiales:")
    print("- 'limpiar' para limpiar la memoria")
    print("- 'historial' para ver el historial de conversación")
    print("-" * 60)
    
    while True:
        user_question = input("Tú: ")
        
        if user_question.lower() == 'salir':
            print("Agente: ¡Hasta luego!")
            break
        elif user_question.lower() == 'limpiar':
            agent.clear_memory()
            continue
        elif user_question.lower() == 'historial':
            history = agent.get_conversation_history()
            print("\n--- Historial de Conversación ---")
            for msg in history:
                role = "Tú" if msg["role"] == "user" else "Agente"
                print(f"{role}: {msg['content']}")
            print("-" * 30 + "\n")
            continue
        
        # Get response from agent
        ai_answer = agent.respond(user_question)
        print(f"Agente: {ai_answer}")


if __name__ == "__main__":
    # Example usage and testing
    run_qa_agent_interactive()
