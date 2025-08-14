"""
Demo del sistema de logging de conversaciones.
Muestra cómo los agentes y el orquestador registran sus interacciones.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from ai_agents.agents.chat.langchain_agent import LangChainChatAgent
from ai_agents.agents.orchestration.advanced_orchestrator import AdvancedOrchestrator
from ai_agents.agents.data_analysis.pandas_agent import PandasAgent
from ai_agents.utils.conversation_logger import conversation_logger
from ai_agents.config.settings import settings


async def demo_chat_agent_logging():
    """Demuestra el logging de conversaciones con el chat agent."""
    print("🤖 Demo: Logging de Chat Agent")
    print("=" * 50)
    
    # Crear agente de chat
    chat_agent = LangChainChatAgent(
        agent_id="demo_chat_agent",
        enable_persistence=True
    )
    
    await chat_agent.initialize()
    
    # Simular conversación
    messages = [
        "Hola, soy Santiago",
        "¿Cuál es mi nombre?", 
        "Tengo 30 años",
        "¿Cuántos años tengo?"
    ]
    
    session_id = "demo_session_001"
    
    # Iniciar logging de conversación
    chat_agent.start_conversation_log(session_id, user_id="santiago")
    
    for i, message in enumerate(messages, 1):
        print(f"\n📝 Mensaje {i}: {message}")
        
        try:
            response = await chat_agent.process({
                "content": message,
                "session_id": session_id
            })
            print(f"🤖 Respuesta: {response.content}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Conversación completada. Logs guardados en: data/logs/{datetime.now().strftime('%Y-%m-%d')}/")


async def demo_orchestrator_logging():
    """Demuestra el logging del orquestador con workflows."""
    print("\n🎼 Demo: Logging de Orquestador")
    print("=" * 50)
    
    # Crear orquestador
    orchestrator = AdvancedOrchestrator(
        agent_id="demo_orchestrator"
    )
    
    await orchestrator.initialize()
    
    # Simular ejecución de workflow
    print("\n🚀 Ejecutando workflow de análisis de datos...")
    
    try:
        execution = await orchestrator.execute_workflow(
            "data_analysis_complete",
            {
                "dataset": "demo_sales_data.csv",
                "analysis_type": "comprehensive",
                "user_id": "santiago"
            }
        )
        
        print(f"📊 Workflow completado con estado: {execution.status.value}")
        print(f"🔧 Pasos ejecutados: {len(execution.results)}")
        
        if execution.errors:
            print(f"⚠️ Errores encontrados: {len(execution.errors)}")
        
    except Exception as e:
        print(f"❌ Error ejecutando workflow: {e}")
    
    print(f"\n✅ Workflow completado. Logs guardados en: data/logs/{datetime.now().strftime('%Y-%m-%d')}/")


def show_log_structure():
    """Muestra la estructura de los logs generados."""
    print("\n📋 Estructura de Logs")
    print("=" * 50)
    
    logs_dir = Path(settings.data_dir) / "logs"
    
    if not logs_dir.exists():
        print("❌ No se encontraron logs. Ejecuta primero los demos.")
        return
    
    print(f"📁 Directorio de logs: {logs_dir}")
    
    # Mostrar estructura de directorios
    for date_dir in sorted(logs_dir.iterdir()):
        if date_dir.is_dir():
            print(f"\n📅 {date_dir.name}:")
            log_files = list(date_dir.glob("*.json"))
            print(f"   📄 {len(log_files)} archivos de log")
            
            # Mostrar ejemplo de log
            if log_files:
                example_file = log_files[0]
                try:
                    with open(example_file, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    
                    print(f"\n   🔍 Ejemplo de log ({example_file.name}):")
                    print(f"      Agente: {log_data.get('agent_type', 'N/A')}")
                    print(f"      Sesión: {log_data.get('session_id', 'N/A')}")
                    print(f"      Mensajes: {len(log_data.get('messages', []))}")
                    print(f"      Inicio: {log_data.get('timestamp', 'N/A')}")
                    
                    # Mostrar tipos de mensajes
                    message_types = {}
                    for msg in log_data.get('messages', []):
                        msg_type = msg.get('message_type', 'unknown')
                        message_types[msg_type] = message_types.get(msg_type, 0) + 1
                    
                    if message_types:
                        print(f"      Tipos de mensaje: {dict(message_types)}")
                
                except Exception as e:
                    print(f"      ❌ Error leyendo log: {e}")


def demo_log_analysis():
    """Demuestra el análisis de logs."""
    print("\n📊 Demo: Análisis de Logs")
    print("=" * 50)
    
    # Generar reporte de conversaciones
    print("📈 Generando reporte de conversaciones...")
    
    report = conversation_logger.generate_conversation_report()
    
    print(f"\n📋 Reporte de Conversaciones:")
    print(f"   Fecha del reporte: {report['report_generated_at']}")
    
    stats = report['statistics']
    print(f"\n📊 Estadísticas:")
    print(f"   Total de conversaciones: {stats['total_conversations']}")
    print(f"   Total de mensajes: {stats['total_messages']}")
    print(f"   Mensajes de usuario: {stats['total_user_messages']}")
    print(f"   Respuestas de agentes: {stats['total_agent_responses']}")
    print(f"   Errores: {stats['total_errors']}")
    print(f"   Tiempo promedio de procesamiento: {stats['average_processing_time_ms']:.2f}ms")
    print(f"   Tasa de éxito: {stats['success_rate']:.1f}%")
    
    print(f"\n🤖 Tipos de agentes:")
    for agent_type, count in stats['agent_types'].items():
        print(f"   {agent_type}: {count} conversaciones")
    
    # Obtener logs recientes
    print(f"\n📄 Logs recientes:")
    recent_logs = conversation_logger.get_conversation_logs(limit=5)
    
    for i, log in enumerate(recent_logs, 1):
        print(f"   {i}. Agente: {log.get('agent_type', 'N/A')} | "
              f"Sesión: {log.get('session_id', 'N/A')[:8]}... | "
              f"Mensajes: {len(log.get('messages', []))}")


async def main():
    """Función principal del demo."""
    print("🔍 DEMO: Sistema de Logging de Conversaciones")
    print("=" * 60)
    print("Este demo muestra cómo los agentes registran automáticamente")
    print("sus interacciones en archivos JSON para análisis y debugging.")
    print("=" * 60)
    
    # Demo 1: Chat Agent
    await demo_chat_agent_logging()
    
    # Demo 2: Orchestrator
    await demo_orchestrator_logging()
    
    # Demo 3: Estructura de logs
    show_log_structure()
    
    # Demo 4: Análisis de logs
    demo_log_analysis()
    
    print("\n" + "=" * 60)
    print("✅ Demo completado!")
    print(f"📁 Revisa los logs en: {Path(settings.data_dir) / 'logs'}")
    print("💡 Los logs están organizados por fecha en formato JSON")
    print("🔍 Cada conversación tiene un ID único para seguimiento")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
