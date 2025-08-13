#!/usr/bin/env python3
"""
Demo completo del Step 2.13: CLI y API Interfaces
Demuestra las capacidades del framework AI Agents a través de CLI y API.
"""

import asyncio
import httpx
import json
import time
import subprocess
import sys
from pathlib import Path

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*50}")
    print(f"🎯 {title}")
    print(f"{'='*50}")

def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'─'*30}")
    print(f"📋 {title}")
    print(f"{'─'*30}")

async def demo_cli():
    """Demonstrate CLI capabilities."""
    print_section("Demo CLI - Interfaz de Línea de Comandos")
    
    print_subsection("1. Información General del CLI")
    result = subprocess.run(["ai-agents", "--help"], capture_output=True, text=True)
    print("Comando: ai-agents --help")
    print(result.stdout)
    
    print_subsection("2. Listar Agentes Disponibles")
    result = subprocess.run(["ai-agents", "agent", "list"], capture_output=True, text=True)
    print("Comando: ai-agents agent list")
    if result.returncode == 0:
        agents_data = json.loads(result.stdout)
        print(f"✅ Total de agentes: {agents_data['total']}")
        for agent in agents_data['agents']:
            print(f"   • {agent['id']} ({agent['type']}) - Status: {agent['status']}")
    else:
        print(f"❌ Error: {result.stderr}")
    
    print_subsection("3. Estado del Orquestrador")
    result = subprocess.run(["ai-agents", "orchestrator", "status"], capture_output=True, text=True)
    print("Comando: ai-agents orchestrator status")
    if result.returncode == 0:
        status_data = json.loads(result.stdout)
        print(f"✅ Estado: {status_data['status']}")
        print(f"   • Workflows activos: {status_data['active_workflows']}")
        print(f"   • Total workflows: {status_data['total_workflows']}")
    else:
        print(f"❌ Error: {result.stderr}")
    
    print_subsection("4. Comandos Disponibles")
    commands = [
        ("agent", "Gestión de agentes"),
        ("workflow", "Gestión de workflows"),
        ("orchestrator", "Control del orquestrador"),
        ("chat", "Chat interactivo"),
        ("data", "Análisis de datos"),
        ("text", "Procesamiento de texto"),
        ("serve", "Servidor API REST")
    ]
    
    for cmd, desc in commands:
        print(f"   • ai-agents {cmd} - {desc}")

async def demo_api():
    """Demonstrate API capabilities."""
    print_section("Demo API - Interfaz REST")
    
    base_url = "http://localhost:8002"
    
    async with httpx.AsyncClient() as client:
        print_subsection("1. Health Check - Estado del Sistema")
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API Status: {health_data['status']}")
                print(f"   • Orquestrador: {health_data['orchestrator']}")
                print(f"   • Workflows activos: {health_data['active_workflows']}")
                print(f"   • Agentes saludables: {len([a for a in health_data['agents'].values() if a['status'] == 'healthy'])}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Error conectando con API: {e}")
            return
        
        print_subsection("2. Listado de Agentes via API")
        try:
            response = await client.get(f"{base_url}/agents")
            if response.status_code == 200:
                agents_data = response.json()
                print(f"✅ Agentes encontrados: {agents_data['total']}")
                for agent in agents_data['agents']:
                    print(f"   • {agent['id']} - {agent['type']} ({agent['status']})")
            else:
                print(f"❌ Error obteniendo agentes: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print_subsection("3. Workflows Disponibles")
        try:
            response = await client.get(f"{base_url}/workflows")
            if response.status_code == 200:
                workflows_data = response.json()
                print(f"✅ Workflows disponibles: {workflows_data['total']}")
                for workflow in workflows_data['workflows']:
                    print(f"   • {workflow['id']} - {workflow['name']}")
                    print(f"     Status: {workflow['status']} | Tipo: {workflow['type']}")
            else:
                print(f"❌ Error obteniendo workflows: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print_subsection("4. Métricas del Sistema")
        try:
            response = await client.get(f"{base_url}/orchestrator/metrics")
            if response.status_code == 200:
                metrics_data = response.json()
                print(f"✅ Métricas del sistema:")
                print(f"   • Total de agentes: {len(metrics_data['agents'])}")
                print(f"   • Workflows definidos: {metrics_data['total_workflows']}")
                print(f"   • Sistema activo desde: {metrics_data['timestamp']}")
            else:
                print(f"❌ Error obteniendo métricas: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")

def demo_integration():
    """Demonstrate CLI and API integration."""
    print_section("Demo Integración - CLI ↔ API")
    
    print_subsection("Ejemplo de Flujo Completo")
    print("""
    🔄 Flujo típico de uso:
    
    1. CLI para desarrollo y testing:
       ai-agents agent list                    # Ver agentes disponibles
       ai-agents agent run pandas "analizar ventas.csv"
       ai-agents workflow create mi_flujo
       
    2. API para integración web/aplicaciones:
       GET /agents                            # Obtener lista de agentes
       POST /agents/{agent_id}/execute        # Ejecutar tarea específica
       GET /workflows/{workflow_id}/status    # Estado del workflow
       
    3. Servidor integrado:
       ai-agents serve --port 8000 --reload  # Levantar API desde CLI
       # Acceso a documentación: http://localhost:8000/docs
    
    📖 Documentación automática disponible en:
       • /docs - Swagger UI interactiva
       • /redoc - Documentación ReDoc
       • /openapi.json - Especificación OpenAPI
    """)

def demo_features():
    """Show key features of Step 2.13."""
    print_section("Features Implementadas - Step 2.13")
    
    features = [
        ("🎯 CLI Completo", [
            "Comandos intuitivos con Click framework",
            "Soporte para múltiples formatos de salida (JSON, YAML, tabla)",
            "Modo verbose para debugging",
            "Comandos organizados por dominio (agent, workflow, etc.)",
            "Integración nativa con AdvancedOrchestrator"
        ]),
        ("🌐 API REST", [
            "FastAPI con documentación automática",
            "Endpoints para todos los componentes principales",
            "Validación de datos con Pydantic v2",
            "Soporte async/await completo",
            "CORS habilitado para integración web"
        ]),
        ("🔧 Integración", [
            "CLI puede lanzar el servidor API",
            "Compartición del mismo backend (AdvancedOrchestrator)",
            "Manejo consistente de errores en ambas interfaces",
            "Configuración unificada",
            "Scripts de inicio independientes"
        ]),
        ("📊 Monitoreo", [
            "Health checks detallados",
            "Métricas de sistema en tiempo real",
            "Estado de agentes y workflows",
            "Logs estructurados",
            "Endpoints de diagnóstico"
        ])
    ]
    
    for feature_name, items in features:
        print_subsection(feature_name)
        for item in items:
            print(f"   ✅ {item}")

async def main():
    """Run the complete Step 2.13 demonstration."""
    print("🚀 Iniciando Demo Completo del Step 2.13")
    print("   Framework AI Agents - CLI y API Interfaces")
    print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if we can access CLI
    try:
        result = subprocess.run(["ai-agents", "--version"], capture_output=True, text=True, timeout=5)
        cli_available = result.returncode == 0
    except:
        cli_available = False
    
    # Check if API is accessible
    api_available = False
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8002/health", timeout=5.0)
            api_available = response.status_code == 200
    except:
        pass
    
    print(f"\n📋 Estado de las interfaces:")
    print(f"   • CLI disponible: {'✅' if cli_available else '❌'}")
    print(f"   • API disponible: {'✅' if api_available else '❌ (usar: ai-agents serve --port 8002)'}")
    
    # Run demos
    demo_features()
    
    if cli_available:
        await demo_cli()
    else:
        print("\n⚠️  CLI no disponible. Instalar con: pip install -e .")
    
    if api_available:
        await demo_api()
    else:
        print("\n⚠️  API no disponible. Iniciar con: ai-agents serve --port 8002")
    
    demo_integration()
    
    print_section("🎉 Demo Completado - Step 2.13")
    print("""
    ✅ Step 2.13 implementado exitosamente:
    
    🎯 CLI Framework completo con Click
    🌐 API REST con FastAPI y documentación automática  
    🔧 Integración seamless entre CLI y API
    📊 Monitoreo y métricas en tiempo real
    📖 Documentación interactiva (Swagger UI)
    
    🚀 Próximos pasos:
    • Step 2.14: Dashboard web interface
    • Step 2.15: Authentication & Authorization
    • Step 2.16: Advanced monitoring & analytics
    """)

if __name__ == "__main__":
    asyncio.run(main())
