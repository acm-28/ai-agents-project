"""
Demostración del DataAnalysisAgent
=====================================

Este script muestra las capacidades avanzadas del DataAnalysisAgent que combina:
- Memoria conversacional (inspirado en agent1_context_awareness)
- Estructura LangGraph por nodos (inspirado en agent0_langGraph)
- Análisis de datos con pandas (basado en guia_agent3_data_analysis)

Características principales:
- Análisis de datos con lenguaje natural
- Memoria persistente entre consultas en la misma sesión
- Flujo estructurado de procesamiento con LangGraph
- Manejo de múltiples sesiones
- Carga de datasets personalizados
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.agent3_data_analysis import DataAnalysisAgent


def demo_basic_analysis():
    """Demuestra análisis básicos con el dataset de ejemplo."""
    print("=== DEMO: Análisis Básico ===")
    
    agent = DataAnalysisAgent()
    agent.initialize()
    
    session_id = "demo_basic"
    
    queries = [
        "¿Cuántas filas y columnas tiene el dataset?",
        "¿Cuáles son las columnas disponibles?",
        "¿Cuál es el precio promedio de los autos?",
        "¿Qué marca de auto aparece más frecuentemente?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Pregunta: {query}")
        response = agent.respond(query, session_id)
        print(f"   Respuesta: {response}")
    
    print("\n" + "="*60 + "\n")


def demo_memory_functionality():
    """Demuestra la funcionalidad de memoria entre consultas."""
    print("=== DEMO: Memoria Conversacional ===")
    
    agent = DataAnalysisAgent()
    agent.initialize()
    
    session_id = "demo_memory"
    
    # Secuencia de consultas que demuestran memoria
    conversation = [
        "¿Cuál es el precio promedio de los autos?",
        "¿Y cuál es el precio más alto?",
        "¿Puedes comparar ese precio promedio con la mediana?",
        "Ahora analiza solo los autos de la marca BMW"
    ]
    
    print("Simulando una conversación con memoria:")
    
    for i, query in enumerate(conversation, 1):
        print(f"\n{i}. Usuario: {query}")
        response = agent.respond(query, session_id)
        print(f"   Agente: {response}")
    
    print("\n" + "="*60 + "\n")


def demo_multiple_sessions():
    """Demuestra el manejo de múltiples sesiones independientes."""
    print("=== DEMO: Múltiples Sesiones ===")
    
    agent = DataAnalysisAgent()
    agent.initialize()
    
    # Sesión 1: Análisis de precios
    print("Sesión 1 - Análisis de precios:")
    response1 = agent.respond("¿Cuál es el precio promedio?", "sesion_precios")
    print(f"Respuesta: {response1}")
    
    # Sesión 2: Análisis de marcas
    print("\nSesión 2 - Análisis de marcas:")
    response2 = agent.respond("¿Qué marcas hay disponibles?", "sesion_marcas")
    print(f"Respuesta: {response2}")
    
    # Continuar sesión 1
    print("\nContinuando Sesión 1:")
    response3 = agent.respond("¿Y cuál es el precio máximo?", "sesion_precios")
    print(f"Respuesta: {response3}")
    
    # Mostrar sesiones disponibles
    sessions = agent.get_available_sessions()
    print(f"\nSesiones activas: {sessions}")
    
    print("\n" + "="*60 + "\n")


def demo_custom_dataset():
    """Demuestra la carga de un dataset personalizado."""
    print("=== DEMO: Dataset Personalizado ===")
    
    agent = DataAnalysisAgent()
    agent.initialize()
    
    # Crear un dataset de ejemplo diferente
    print("Creando dataset personalizado (ventas de productos)...")
    
    np.random.seed(123)
    custom_data = {
        'producto': ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Webcam'] * 20,
        'categoria': ['Electrónicos'] * 50 + ['Accesorios'] * 50,
        'precio': np.random.uniform(50, 2000, 100),
        'cantidad_vendida': np.random.randint(1, 50, 100),
        'mes': np.random.choice(['Enero', 'Febrero', 'Marzo', 'Abril'], 100)
    }
    
    custom_df = pd.DataFrame(custom_data)
    
    # Cargar el dataset personalizado
    result = agent.load_dataset(dataframe=custom_df)
    print(f"Resultado de carga: {result}")
    
    # Hacer consultas sobre el nuevo dataset
    session_id = "demo_custom"
    
    queries = [
        "¿Qué productos están en el dataset?",
        "¿Cuál es el precio promedio por categoría?",
        "¿Qué producto se vendió más?",
        "Analiza las ventas por mes"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Pregunta: {query}")
        response = agent.respond(query, session_id)
        print(f"   Respuesta: {response}")
    
    print("\n" + "="*60 + "\n")


def demo_dataset_info():
    """Demuestra las funciones de información del dataset."""
    print("=== DEMO: Información del Dataset ===")
    
    agent = DataAnalysisAgent()
    agent.initialize()
    
    # Obtener información detallada
    info = agent.get_dataset_info()
    
    print("Información del dataset actual:")
    print(f"- Forma: {info['shape']}")
    print(f"- Columnas: {info['columns']}")
    print(f"- Uso de memoria: {info['memory_usage']}")
    print(f"- Tipos de datos: {info['dtypes']}")
    print(f"- Valores nulos: {info['null_counts']}")
    
    print("\nPrimeras filas del dataset:")
    sample_data = info['sample_data']
    df_sample = pd.DataFrame(sample_data)
    print(df_sample.to_string())
    
    print("\n" + "="*60 + "\n")


def demo_error_handling():
    """Demuestra el manejo de errores y casos límite."""
    print("=== DEMO: Manejo de Errores ===")
    
    agent = DataAnalysisAgent()
    agent.initialize()
    
    session_id = "demo_errors"
    
    # Test con consulta vacía
    print("1. Consulta vacía:")
    response = agent.respond("", session_id)
    print(f"   Respuesta: {response}")
    
    # Test con consulta ambigua
    print("\n2. Consulta ambigua:")
    response = agent.respond("dime algo", session_id)
    print(f"   Respuesta: {response}")
    
    # Test con consulta compleja
    print("\n3. Consulta compleja:")
    response = agent.respond("Haz un análisis de regresión multivariante con validación cruzada", session_id)
    print(f"   Respuesta: {response}")
    
    print("\n" + "="*60 + "\n")


def interactive_demo():
    """Demo interactivo donde el usuario puede hacer preguntas."""
    print("=== DEMO INTERACTIVO ===")
    print("¡Ahora puedes interactuar directamente con el agente!")
    print("Comandos especiales:")
    print("- 'info' para ver información del dataset")
    print("- 'nueva_sesion' para crear una nueva sesión")
    print("- 'sesiones' para ver sesiones activas")
    print("- 'salir' para terminar")
    print("-" * 50)
    
    agent = DataAnalysisAgent()
    agent.initialize()
    
    current_session = "sesion_interactiva"
    
    while True:
        user_input = input(f"\nTú [{current_session}]: ")
        
        if user_input.lower() == 'salir':
            print("¡Hasta luego!")
            break
        elif user_input.lower() == 'info':
            info = agent.get_dataset_info()
            print(f"Dataset: {info['shape'][0]} filas x {info['shape'][1]} columnas")
            print(f"Columnas: {info['columns']}")
            continue
        elif user_input.lower() == 'nueva_sesion':
            current_session = f"sesion_{datetime.now().strftime('%H%M%S')}"
            print(f"Nueva sesión: {current_session}")
            continue
        elif user_input.lower() == 'sesiones':
            sessions = agent.get_available_sessions()
            print(f"Sesiones activas: {sessions}")
            continue
        
        response = agent.respond(user_input, current_session)
        print(f"Agente: {response}")


def main():
    """Función principal que ejecuta todas las demostraciones."""
    print("🤖 DEMOSTRACIÓN DEL DATA ANALYSIS AGENT 🤖")
    print("=" * 60)
    print()
    
    # Verificar variables de entorno
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY no encontrada.")
        print("Por favor, configura tu API key de OpenAI en un archivo .env")
        return
    
    print("Selecciona qué demostración ejecutar:")
    print("1. Análisis básico")
    print("2. Memoria conversacional")
    print("3. Múltiples sesiones")
    print("4. Dataset personalizado")
    print("5. Información del dataset")
    print("6. Manejo de errores")
    print("7. Demo interactivo")
    print("8. Ejecutar todas las demos")
    
    choice = input("\nElige una opción (1-8): ").strip()
    
    try:
        if choice == "1":
            demo_basic_analysis()
        elif choice == "2":
            demo_memory_functionality()
        elif choice == "3":
            demo_multiple_sessions()
        elif choice == "4":
            demo_custom_dataset()
        elif choice == "5":
            demo_dataset_info()
        elif choice == "6":
            demo_error_handling()
        elif choice == "7":
            interactive_demo()
        elif choice == "8":
            print("Ejecutando todas las demostraciones...\n")
            demo_basic_analysis()
            demo_memory_functionality()
            demo_multiple_sessions()
            demo_custom_dataset()
            demo_dataset_info()
            demo_error_handling()
            print("¡Todas las demostraciones completadas!")
        else:
            print("Opción inválida. Ejecutando demo interactivo...")
            interactive_demo()
            
    except Exception as e:
        print(f"❌ Error durante la demostración: {e}")
        print("Verifica que todas las dependencias estén instaladas correctamente.")


if __name__ == "__main__":
    main()
