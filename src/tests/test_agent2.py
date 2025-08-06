import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent2_qa import QAAgent

def test_qa_agent_basic(agent):
    """Test básico del agente QA."""
    print("=== Test Básico del Agente QA ===")
    print("DEBUG: Iniciando test básico")
    
    # Preguntas de prueba
    test_questions = [
        "¿Cuál es la capital de Francia?",
        "¿Recuerdas mi pregunta anterior?",
        "¿Puedes explicarme qué es la inteligencia artificial?",
        "¿De qué hablamos antes?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Pregunta {i} ---")
        print(f"Usuario: {question}")
        response = agent.get_answer(question)
        print(f"Agente: {response}")
    
    print("DEBUG: Test básico completado")

def test_qa_agent_memory(agent):
    """Test específico de la funcionalidad de memoria."""
    print("\n=== Test de Memoria del Agente QA ===")
    print("DEBUG: Iniciando test de memoria")
    
    # Establecer contexto
    agent.set_system_context("Eres un experto en programación Python.")
    
    # Secuencia de preguntas relacionadas
    questions = [
        "Mi nombre es Juan y soy programador Python",
        "¿Cuál es mi nombre?",
        "¿Qué lenguaje de programación uso?",
        "¿Puedes crear una función simple en Python?"
    ]
    
    for question in questions:
        response = agent.get_answer(question)
        print(f"Usuario: {question}")
        print(f"Agente: {response}\n")
    
    # Test de limpieza de memoria
    print("--- Limpiando memoria ---")
    agent.clear_memory()
    
    response = agent.get_answer("¿Recuerdas mi nombre?")
    print(f"Usuario: ¿Recuerdas mi nombre?")
    print(f"Agente: {response}")
    print("DEBUG: Test de memoria completado")

def test_qa_agent_edge_cases(agent):
    """Test de casos especiales."""
    print("\n=== Test de Casos Especiales ===")
    print("DEBUG: Iniciando test de casos especiales")
    
    # Casos especiales (reducidos para debugging)
    test_cases = [
        "",  # Pregunta vacía
        "   ",  # Solo espacios
        "¿" * 10,  # Pregunta corta para test
        "¿Qué es esto?",  # Pregunta simple
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nDEBUG: Procesando caso {i}")
        print(f"--- Caso Especial {i} ---")
        print(f"Input: '{test_case}'")
        try:
            response = agent.get_answer(test_case)
            print(f"Respuesta: {response[:100]}{'...' if len(response) > 100 else ''}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("DEBUG: Test de casos especiales completado")

if __name__ == "__main__":
    print("DEBUG: Iniciando programa principal")
    agent = QAAgent()
    print("DEBUG: Agente creado")
    agent.initialize()
    print("DEBUG: Agente inicializado")
    
    test_qa_agent_basic(agent)
    print("DEBUG: Test básico terminado")
    
    test_qa_agent_memory(agent)
    print("DEBUG: Test de memoria terminado")
    
    test_qa_agent_edge_cases(agent)
    print("DEBUG: Test de casos especiales terminado")
    
    print("DEBUG: Programa completado")
