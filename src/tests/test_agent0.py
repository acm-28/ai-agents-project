from src.agents.agent0_langGraph import SophisticatedAgent

def test_sophisticated_agent():
    # Texto de ejemplo de la guía
    sample_text = """
OpenAI has announced the GPT-4 model, which is a large multimodal model that exhibits human-level performance on various professional benchmarks. It is developed to improve the alignment and safety of AI systems.
additionally, the model is designed to be more efficient and scalable than its predecessor, GPT-3. The GPT-4 model is expected to be released in the coming months and will be available to the public for research and development purposes.
"""

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

if __name__ == "__main__":
    test_sophisticated_agent()
