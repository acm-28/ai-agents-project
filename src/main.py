from agents.langchain_chat_agent import LangChainChatAgent

def main():
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

if __name__ == "__main__":
    main()