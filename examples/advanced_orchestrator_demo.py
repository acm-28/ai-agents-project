"""
Demostración del AdvancedOrchestrator con workflows complejos.
"""

import asyncio
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

from ai_agents.agents.orchestration.advanced_orchestrator import (
    AdvancedOrchestrator,
    WorkflowDefinition,
    WorkflowStep,
    WorkflowStatus
)
from ai_agents.agents.chat.langchain_agent import LangChainChatAgent
from ai_agents.agents.chat.llm_agent import LLMChatAgent
from ai_agents.agents.qa.memory_qa_agent import MemoryQAAgent
from ai_agents.agents.data_analysis.pandas_agent import PandasAgent
from ai_agents.agents.workflows.sophisticated_agent import SophisticatedAgent


async def create_sample_data():
    """Crear datos de ejemplo para la demostración."""
    # Crear directorio de datos si no existe
    data_dir = Path("demo_data")
    data_dir.mkdir(exist_ok=True)
    
    # Crear dataset de ventas
    sales_data = pd.DataFrame({
        'fecha': pd.date_range('2024-01-01', periods=100, freq='D'),
        'producto': ['A', 'B', 'C'] * 34 + ['A', 'B'],
        'ventas': [100 + i*2 + (i%7)*10 for i in range(100)],
        'region': ['Norte', 'Sur', 'Este', 'Oeste'] * 25,
        'vendedor': [f'Vendedor_{i%10}' for i in range(100)]
    })
    
    sales_file = data_dir / "ventas_demo.csv"
    sales_data.to_csv(sales_file, index=False)
    
    # Crear documentos de texto
    documents = [
        "El análisis de ventas muestra tendencias positivas en Q1 2024.",
        "Las regiones Norte y Este lideran en performance de ventas.",
        "Los productos A y B son los más rentables según los datos históricos.",
        "Se recomienda aumentar la inversión en marketing digital.",
        "Los vendedores top han mostrado 25% más eficiencia este trimestre."
    ]
    
    docs_file = data_dir / "documentos_demo.txt"
    with open(docs_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(documents))
    
    return str(sales_file), str(docs_file)


async def setup_orchestrator():
    """Configurar el orquestrador con agentes reales."""
    print("🔧 Configurando agentes...")
    
    # Crear agentes (usando versiones simplificadas para demo)
    pandas_agent = PandasAgent()
    sophisticated_agent = SophisticatedAgent()
    qa_agent = MemoryQAAgent()
    langchain_agent = LangChainChatAgent()
    llm_agent = LLMChatAgent()
    
    # Crear orquestrador avanzado
    orchestrator = AdvancedOrchestrator(
        pandas_agent=pandas_agent,
        sophisticated_agent=sophisticated_agent,
        qa_agent=qa_agent,
        langchain_agent=langchain_agent,
        llm_agent=llm_agent,
        max_parallel_executions=3
    )
    
    print("✅ Orquestrador configurado con 5 agentes especializados")
    return orchestrator


def create_business_intelligence_workflow():
    """Crear un workflow complejo de Business Intelligence."""
    return WorkflowDefinition(
        id="business_intelligence_complete",
        name="Análisis Completo de Business Intelligence",
        description="Workflow integral que combina análisis de datos, procesamiento de documentos y generación de insights",
        steps=[
            # Paso 1: Análisis estadístico de datos de ventas
            WorkflowStep(
                id="sales_analysis",
                name="Análisis Estadístico de Ventas",
                agent_id="pandas_agent",
                task="Realizar análisis estadístico completo de los datos de ventas",
                inputs={
                    "file_path": "{{workflow_input.sales_file}}",
                    "analysis_type": "comprehensive",
                    "operations": ["describe", "correlations", "trends"]
                },
                dependencies=[]
            ),
            
            # Paso 2: Análisis de documentos en paralelo
            WorkflowStep(
                id="document_analysis",
                name="Análisis de Documentos Corporativos",
                agent_id="sophisticated_agent",
                task="Extraer insights y patrones de documentos corporativos",
                inputs={
                    "documents_path": "{{workflow_input.docs_file}}",
                    "analysis_focus": ["trends", "recommendations", "performance"]
                },
                dependencies=[]
            ),
            
            # Paso 3: Q&A sobre los resultados (depende de ambos análisis)
            WorkflowStep(
                id="insights_qa",
                name="Generación de Insights via Q&A",
                agent_id="qa_agent",
                task="Responder preguntas específicas basadas en los análisis realizados",
                inputs={
                    "context_sales": "{{sales_analysis.result}}",
                    "context_docs": "{{document_analysis.result}}",
                    "questions": [
                        "¿Cuáles son las principales tendencias de ventas?",
                        "¿Qué productos tienen mejor performance?",
                        "¿Cuáles son las recomendaciones clave?"
                    ]
                },
                dependencies=["sales_analysis", "document_analysis"]
            ),
            
            # Paso 4: Reporte ejecutivo
            WorkflowStep(
                id="executive_report",
                name="Generación de Reporte Ejecutivo",
                agent_id="langchain_agent",
                task="Crear un reporte ejecutivo consolidado",
                inputs={
                    "sales_data": "{{sales_analysis.result}}",
                    "document_insights": "{{document_analysis.result}}",
                    "qa_insights": "{{insights_qa.result}}",
                    "report_format": "executive_summary"
                },
                dependencies=["sales_analysis", "document_analysis", "insights_qa"]
            ),
            
            # Paso 5: Recomendaciones estratégicas
            WorkflowStep(
                id="strategic_recommendations",
                name="Generación de Recomendaciones Estratégicas",
                agent_id="llm_agent",
                task="Generar recomendaciones estratégicas basadas en todos los análisis",
                inputs={
                    "comprehensive_analysis": "{{executive_report.result}}",
                    "focus_areas": ["growth", "optimization", "risk_management"]
                },
                dependencies=["executive_report"]
            )
        ],
        metadata={
            "version": "2.0",
            "category": "business_intelligence",
            "estimated_duration": "5-10 minutes",
            "complexity": "high"
        }
    )


def create_parallel_processing_workflow():
    """Crear un workflow que demuestra procesamiento paralelo."""
    return WorkflowDefinition(
        id="parallel_processing_demo",
        name="Demostración de Procesamiento Paralelo",
        description="Workflow que ejecuta múltiples análisis independientes en paralelo",
        steps=[
            # Pasos paralelos - no dependen entre sí
            WorkflowStep(
                id="quick_stats",
                name="Estadísticas Rápidas",
                agent_id="pandas_agent",
                task="Generar estadísticas básicas rápidamente",
                inputs={
                    "file_path": "{{workflow_input.sales_file}}",
                    "analysis_type": "basic"
                },
                dependencies=[]
            ),
            
            WorkflowStep(
                id="text_summary",
                name="Resumen de Texto",
                agent_id="sophisticated_agent",
                task="Crear resumen de documentos",
                inputs={
                    "documents_path": "{{workflow_input.docs_file}}",
                    "summary_length": "short"
                },
                dependencies=[]
            ),
            
            WorkflowStep(
                id="quick_qa",
                name="Q&A Rápido",
                agent_id="qa_agent",
                task="Responder pregunta simple",
                inputs={
                    "question": "¿Cuál es el contexto general de los datos?",
                    "context": "análisis de ventas y documentos corporativos"
                },
                dependencies=[]
            ),
            
            # Paso final que combina todos los resultados paralelos
            WorkflowStep(
                id="parallel_summary",
                name="Resumen de Procesamiento Paralelo",
                agent_id="langchain_agent",
                task="Consolidar resultados de procesamiento paralelo",
                inputs={
                    "stats": "{{quick_stats.result}}",
                    "summary": "{{text_summary.result}}",
                    "qa": "{{quick_qa.result}}"
                },
                dependencies=["quick_stats", "text_summary", "quick_qa"]
            )
        ],
        metadata={
            "version": "1.0",
            "category": "parallel_demo",
            "estimated_duration": "2-3 minutes",
            "complexity": "medium"
        }
    )


async def demonstrate_workflow_execution(orchestrator, workflow, inputs, workflow_name):
    """Ejecutar y monitorear un workflow."""
    print(f"\n🚀 Iniciando workflow: {workflow_name}")
    print(f"   Descripción: {workflow.description}")
    print(f"   Pasos: {len(workflow.steps)}")
    
    start_time = datetime.now()
    
    # Registrar y ejecutar workflow
    orchestrator.register_workflow(workflow)
    execution = await orchestrator.execute_workflow(workflow.id, inputs)
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    # Mostrar resultados
    print(f"\n📊 Resultados del workflow '{workflow_name}':")
    print(f"   Estado: {execution.status.value}")
    print(f"   Tiempo de ejecución: {execution_time:.2f} segundos")
    print(f"   Pasos completados: {len([r for r in execution.step_results.values() if r.status.name == 'COMPLETED'])}/{len(workflow.steps)}")
    
    if execution.status == WorkflowStatus.COMPLETED:
        print("\n✅ Workflow completado exitosamente!")
        
        # Mostrar resumen de cada paso
        for step_id, result in execution.step_results.items():
            step = next(s for s in workflow.steps if s.id == step_id)
            print(f"\n   📋 {step.name}:")
            print(f"      Agente: {step.agent_id}")
            print(f"      Estado: {result.status.name}")
            if result.agent_response:
                content_preview = result.agent_response.content[:100] + "..." if len(result.agent_response.content) > 100 else result.agent_response.content
                print(f"      Resultado: {content_preview}")
    else:
        print(f"\n❌ Workflow falló: {execution.error}")
    
    return execution


async def demonstrate_system_monitoring(orchestrator):
    """Demostrar capacidades de monitoreo del sistema."""
    print("\n📈 Métricas del Sistema:")
    
    metrics = orchestrator.get_system_metrics()
    
    print(f"   Workflows ejecutados: {metrics.get('total_workflows_executed', 0)}")
    print(f"   Ejecuciones activas: {metrics.get('active_executions', 0)}")
    print(f"   Tiempo promedio: {metrics.get('average_execution_time', 0):.2f}s")
    print(f"   Carga del sistema: {metrics.get('system_load', 0):.2f}")
    
    print("\n🤖 Métricas por Agente:")
    for agent_id in orchestrator.available_agents.keys():
        agent_metrics = orchestrator.get_agent_metrics(agent_id)
        if agent_metrics.total_executions > 0:
            print(f"   {agent_id}:")
            print(f"      Ejecuciones: {agent_metrics.total_executions}")
            print(f"      Tasa de éxito: {agent_metrics.success_rate:.1%}")
            print(f"      Tiempo promedio: {agent_metrics.average_execution_time:.2f}s")
            print(f"      Carga actual: {agent_metrics.current_load}")


async def demonstrate_workflow_hooks(orchestrator):
    """Demostrar el sistema de hooks."""
    print("\n🔗 Configurando hooks de monitoreo...")
    
    events_log = []
    
    def log_workflow_started(workflow_id, execution_id):
        events_log.append(f"🚀 Workflow iniciado: {workflow_id} ({execution_id[:8]})")
    
    def log_step_completed(workflow_id, execution_id, step_id, result):
        status = "✅" if result.status.name == "COMPLETED" else "❌"
        events_log.append(f"{status} Paso completado: {step_id} en {workflow_id}")
    
    def log_workflow_completed(workflow_id, execution_id, result):
        status = "🎉" if result.status == WorkflowStatus.COMPLETED else "💥"
        events_log.append(f"{status} Workflow finalizado: {workflow_id}")
    
    # Registrar hooks
    orchestrator.add_hook("workflow_started", log_workflow_started)
    orchestrator.add_hook("step_completed", log_step_completed)
    orchestrator.add_hook("workflow_completed", log_workflow_completed)
    
    print("   Hooks registrados para monitoreo en tiempo real")
    
    return events_log


async def main():
    """Función principal de demostración."""
    print("=" * 60)
    print("🎯 DEMOSTRACIÓN AVANZADA DEL ORCHESTRATOR")
    print("=" * 60)
    
    try:
        # Preparar datos
        print("\n1️⃣ Preparando datos de demostración...")
        sales_file, docs_file = await create_sample_data()
        print(f"   ✅ Datos creados: {sales_file}, {docs_file}")
        
        # Configurar orquestrador
        print("\n2️⃣ Configurando orquestrador...")
        orchestrator = await setup_orchestrator()
        
        # Configurar hooks
        events_log = await demonstrate_workflow_hooks(orchestrator)
        
        # Crear workflows
        print("\n3️⃣ Creando workflows complejos...")
        bi_workflow = create_business_intelligence_workflow()
        parallel_workflow = create_parallel_processing_workflow()
        
        workflow_inputs = {
            "sales_file": sales_file,
            "docs_file": docs_file
        }
        
        # Demostrar workflow de procesamiento paralelo
        print("\n4️⃣ Ejecutando workflow de procesamiento paralelo...")
        await demonstrate_workflow_execution(
            orchestrator, 
            parallel_workflow, 
            workflow_inputs,
            "Procesamiento Paralelo"
        )
        
        # Demostrar workflow complejo de BI
        print("\n5️⃣ Ejecutando workflow completo de Business Intelligence...")
        await demonstrate_workflow_execution(
            orchestrator,
            bi_workflow,
            workflow_inputs, 
            "Business Intelligence Completo"
        )
        
        # Mostrar métricas del sistema
        print("\n6️⃣ Mostrando métricas del sistema...")
        await demonstrate_system_monitoring(orchestrator)
        
        # Mostrar eventos capturados por hooks
        print("\n7️⃣ Eventos capturados por hooks:")
        for event in events_log:
            print(f"   {event}")
        
        # Demostrar workflows predefinidos
        print("\n8️⃣ Ejecutando workflow predefinido...")
        predefined_execution = await orchestrator.execute_workflow(
            "data_analysis_complete",
            {"dataset": sales_file, "analysis_type": "comprehensive"}
        )
        
        print(f"   Workflow predefinido: {predefined_execution.status.value}")
        
        print("\n" + "=" * 60)
        print("🎉 DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        
        # Resumen final
        print(f"\n📊 Resumen de la demostración:")
        print(f"   • Workflows ejecutados: 3")
        print(f"   • Agentes utilizados: 5") 
        print(f"   • Capacidades demostradas:")
        print(f"     - Orquestación multi-agente")
        print(f"     - Procesamiento paralelo")
        print(f"     - Gestión de dependencias")
        print(f"     - Monitoreo en tiempo real")
        print(f"     - Métricas de rendimiento")
        print(f"     - Sistema de hooks")
        print(f"     - Workflows predefinidos")
        
    except Exception as e:
        print(f"\n💥 Error durante la demostración: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ejecutar demostración
    asyncio.run(main())
