from agents.langchain_chat_agent import LangChainChatAgent
from agents.sophisticated_agent import SophisticatedAgent

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
    """Demo del chat agent original"""
    # Crear e inicializar el agente
    agent = LangChainChatAgent()
    agent.initialize()
    
    # Personalizar el comportamiento del agente (opcional)
    agent.set_system_message("Eres un asistente de programación especializado en Python y AI.")
    
    print("¡Hola! Soy tu asistente de chat con LangChain.")
    print("Escribe 'salir' para terminar la conversación.")
    
    while True:
        user_input = input("\nTú: ")
        
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("Agente: ¡Hasta luego! Que tengas un buen día.")
            break
        
        if user_input.strip() == "":
            print("Agente: Por favor, escribe un mensaje.")
            continue
            
        response = agent.respond(user_input)
        print(f"Agente: {response}")

def main():
    print("Selecciona qué demo ejecutar:")
    print("1. SophisticatedAgent (Análisis de texto)")
    print("2. ChatAgent (Conversación)")
    
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