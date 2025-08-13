"""
Ejemplo de uso del AgentOrchestrator - Demostración del Step 2.11

Este ejemplo muestra cómo el AgentOrchestrator coordina automáticamente
múltiples agentes especializados según el tipo de solicitud.
"""

import asyncio
import json
from ai_agents.agents.orchestration.agent_orchestrator import AgentOrchestrator


async def demo_orchestrator():
    """Demostración completa del AgentOrchestrator."""
    
    print("🎯 Iniciando demostración del AgentOrchestrator")
    print("=" * 50)
    
    # Inicializar el orquestrador
    orchestrator = AgentOrchestrator()
    
    try:
        await orchestrator._safe_initialize()
        print("✅ AgentOrchestrator inicializado correctamente")
    except Exception as e:
        print(f"⚠️  Error inicializando orquestrador: {e}")
        print("📝 Continuando con modo mock para demostración...")
        # En un entorno real, aquí configurarías los agentes reales
    
    print(f"📊 Estado del orquestrador: {orchestrator.state}")
    print()
    
    # Mostrar agentes disponibles
    print("🤖 Agentes especializados configurados:")
    for agent_name, capability in orchestrator.agent_capabilities.items():
        print(f"  • {agent_name}: {capability.description}")
        print(f"    Tipos de tareas: {[t.value for t in capability.task_types]}")
        print(f"    Prioridad: {capability.priority}")
        print()
    
    # Ejemplos de solicitudes que serán dirigidas a diferentes agentes
    test_requests = [
        {
            "name": "Análisis de datos CSV",
            "request": {
                "message": "analizar los datos del archivo ventas.csv y crear estadísticas",
                "file_path": "ventas.csv"
            },
            "expected_agent": "PandasAgent"
        },
        {
            "name": "Análisis de texto",
            "request": {
                "message": "clasificar este texto y extraer entidades importantes",
                "text": "Apple Inc. anunció hoy en Cupertino, California, el lanzamiento de su nuevo iPhone con tecnología AI avanzada."
            },
            "expected_agent": "SophisticatedAgent"
        },
        {
            "name": "Consulta con memoria",
            "request": {
                "message": "recordar lo que analizamos anteriormente sobre las ventas"
            },
            "expected_agent": "MemoryQAAgent"
        },
        {
            "name": "Chat general",
            "request": {
                "message": "hola, ¿cómo puedes ayudarme hoy?"
            },
            "expected_agent": "LangChainChatAgent"
        },
        {
            "name": "Workflow complejo",
            "request": {
                "message": "ejecutar un pipeline complejo de procesamiento con múltiples etapas"
            },
            "expected_agent": "SophisticatedAgent"
        }
    ]
    
    print("🔍 Probando clasificación automática de tareas:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_requests, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"📥 Solicitud: {test_case['request']['message']}")
        
        # Clasificar la tarea
        try:
            classification = await orchestrator._classify_task(
                test_case['request']['message'], 
                test_case['request']
            )
            
            print(f"🎯 Clasificación: {classification.task_type.value}")
            print(f"🤖 Agente seleccionado: {classification.agent_name}")
            print(f"📊 Confianza: {classification.confidence:.2f}")
            print(f"💭 Razonamiento: {classification.reasoning}")
            
            # Simular procesamiento (en entorno real llamaría al agente)
            print(f"⚙️  Dirigiendo a: {test_case['expected_agent']} (esperado)")
            
        except Exception as e:
            print(f"❌ Error clasificando: {e}")
    
    print("\n" + "=" * 50)
    print("📈 Capacidades del orquestrador:")
    capabilities = orchestrator.get_capabilities()
    for cap in capabilities[:10]:  # Mostrar primeras 10
        print(f"  • {cap}")
    if len(capabilities) > 10:
        print(f"  ... y {len(capabilities) - 10} más")
    
    # Mostrar estadísticas si hay historial
    stats = orchestrator.get_orchestration_stats()
    if not stats.get('no_interactions'):
        print("\n📊 Estadísticas de orquestación:")
        print(f"  • Total de interacciones: {stats.get('total_interactions', 0)}")
        print(f"  • Tasa de éxito: {stats.get('success_rate', 0):.1%}")
        print(f"  • Agentes especializados: {stats.get('specialized_agents_count', 0)}")
    
    print("\n🎉 Demostración completada!")
    print("💡 El AgentOrchestrator puede coordinar automáticamente múltiples agentes")
    print("   según el tipo de tarea, mantener contexto y proporcionar fallbacks.")


async def demo_context_continuity():
    """Demostración de continuidad de contexto."""
    print("\n🔄 Demostración de continuidad de contexto:")
    print("=" * 40)
    
    orchestrator = AgentOrchestrator()
    await orchestrator.initialize()
    
    # Simular secuencia de solicitudes relacionadas
    sequence = [
        "analizar datos de ventas del Q1",
        "continuar con el análisis del trimestre",
        "comparar con los datos anteriores"
    ]
    
    for i, message in enumerate(sequence, 1):
        print(f"\n{i}. Mensaje: {message}")
        
        classification = await orchestrator._classify_task(message, {"message": message})
        print(f"   Clasificación: {classification.task_type.value}")
        print(f"   Confianza: {classification.confidence:.2f}")
        
        # Simular actualización de contexto
        orchestrator.current_session_context['last_task_type'] = classification.task_type
    
    print("\n✅ El contexto influye en las clasificaciones posteriores!")


if __name__ == "__main__":
    # Ejecutar demostraciones
    asyncio.run(demo_orchestrator())
    asyncio.run(demo_context_continuity())
