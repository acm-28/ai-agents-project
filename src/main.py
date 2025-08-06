from agents.agent1_context_awareness import LangChainChatAgent
from agents.agent0_langGraph import SophisticatedAgent  # Fix this import
from agents.agent2_qa import QAAgent
import uuid

def test_sophisticated_agent():
    """Ejemplo de uso del SophisticatedAgent como se muestra en la guía"""
    
    # Texto de ejemplo de la guía
    sample_text = """
OpenAI has announced the GPT-4 model, which is a large multimodal model that exhibits human-level performance on various professional benchmarks. It is developed to improve the alignment and safety of AI systems.
additionally, the model is designed to be more efficient and scalable than its predecessor, GPT-3. The GPT-4 model is expected to be released in the coming months and will be available to the public for research and development purposes.
"""
    
    print("=== SophisticatedAgent Demo ===")
    print("Texto a analizar:")
    print(sample_text)
    print("\n" + "="*50)
    
    # Crear e inicializar el agente
    agent = SophisticatedAgent()
    agent.initialize()
    
    # Procesar el texto
    state_input = {"text": sample_text}
    result = agent.app.invoke(state_input)
    
    # Mostrar resultados como en la guía
    print("Classification:", result["classification"])
    print("\nEntities:", result["entities"])
    print("\nSummary:", result["summary"])
    
    return result

def chat_demo():
    """Demo del chat agent con memoria de conversación."""
    agent = LangChainChatAgent()
    agent.initialize()
    
    # Generar un ID de sesión único para esta conversación
    session_id = str(uuid.uuid4())
    
    print("¡Hola! Soy tu asistente de chat con memoria.")
    print(f"ID de sesión: {session_id}")
    print("Escribe 'salir' para terminar la conversación.")
    
    while True:
        user_input = input("\nTú: ")
        
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("Agente: ¡Hasta luego! Que tengas un buen día.")
            
            # Opcional: Mostrar el historial de la conversación al final
            print("\nHistorial de la conversación:")
            history = agent.get_session_history(session_id)
            for msg in history.messages:
                print(f"- {msg.type.capitalize()}: {msg.content}")
            break
        
        if user_input.strip() == "":
            print("Agente: Por favor, escribe un mensaje.")
            continue
            
        response = agent.respond(user_input, session_id)
        print(f"Agente: {response}")

def qa_demo():
    """Demo del agente QA con memoria."""
    agent = QAAgent()
    agent.initialize()
    
    print("\n=== Agente QA con Memoria ===")
    print("¡Hola! Soy un agente de preguntas y respuestas con memoria.")
    print("Puedo recordar nuestra conversación anterior.")
    print("\nComandos especiales:")
    print("- 'limpiar' para limpiar la memoria")
    print("- 'historial' para ver el historial de conversación")
    print("- 'salir' para terminar")
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

def main():
    print("Selecciona qué demo ejecutar:")
    print("1. SophisticatedAgent (Análisis de texto)")
    print("2. ChatAgent (Conversación con memoria)")
    print("3. QAAgent (Preguntas y respuestas con memoria)")
    
    choice = input("\nElige una opción (1, 2 o 3): ").strip()
    
    if choice == "1":
        test_sophisticated_agent()
    elif choice == "2":
        chat_demo()
    elif choice == "3":
        qa_demo()
    else:
        print("Opción inválida. Ejecutando SophisticatedAgent por defecto...")
        test_sophisticated_agent()

if __name__ == "__main__":
    main()