from agents.langchain_chat_agent import LangChainChatAgent
from agents.sophisticated_agent import SophisticatedAgent
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

def main():
    print("Selecciona qué demo ejecutar:")
    print("1. SophisticatedAgent (Análisis de texto)")
    print("2. ChatAgent (Conversación con memoria)")
    
    choice = input("\nElige una opción (1 o 2): ").strip()
    
    if choice == "1":
        test_sophisticated_agent()
    elif choice == "2":
        chat_demo()
    else:
        print("Opción inválida. Ejecutando SophisticatedAgent por defecto...")
        test_sophisticated_agent()

if __name__ == "__main__":
    main()