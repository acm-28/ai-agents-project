"""
CLI principal para el framework AI Agents.
Interfaz de línea de comandos para gestionar agentes y workflows.
"""

import click
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from ai_agents.agents import (
    PandasAgent,
    SophisticatedAgent,
    MemoryQAAgent,
    LangChainChatAgent,
    LLMChatAgent,
    AgentOrchestrator,
    AdvancedOrchestrator
)
from ai_agents.agents.orchestration.advanced_orchestrator import (
    WorkflowDefinition,
    WorkflowStep,
    WorkflowStatus
)
from ai_agents.config.settings import settings


class CLIContext:
    """Contexto compartido para comandos CLI."""
    
    def __init__(self):
        self.orchestrator: Optional[AdvancedOrchestrator] = None
        self.verbose: bool = False
        self.output_format: str = "json"
    
    async def get_orchestrator(self) -> AdvancedOrchestrator:
        """Obtener instancia del orquestrador."""
        if self.orchestrator is None:
            if self.verbose:
                click.echo("🔧 Inicializando orquestrador avanzado...")
            
            # Crear agentes especializados
            pandas_agent = PandasAgent()
            sophisticated_agent = SophisticatedAgent()
            qa_agent = MemoryQAAgent()
            langchain_agent = LangChainChatAgent()
            llm_agent = LLMChatAgent()
            
            # Crear orquestrador
            self.orchestrator = AdvancedOrchestrator(agent_id="cli_orchestrator")
            
            # Configurar agentes
            self.orchestrator.specialized_agents = {
                "pandas_agent": pandas_agent,
                "sophisticated_agent": sophisticated_agent,
                "qa_agent": qa_agent,
                "langchain_agent": langchain_agent,
                "llm_agent": llm_agent
            }
            
            # Inicializar
            await self.orchestrator.initialize()
            
            if self.verbose:
                click.echo("✅ Orquestrador inicializado correctamente")
        
        return self.orchestrator
    
    def format_output(self, data: Any) -> str:
        """Formatear salida según el formato configurado."""
        if self.output_format == "json":
            return json.dumps(data, indent=2, default=str, ensure_ascii=False)
        elif self.output_format == "yaml":
            try:
                import yaml
                return yaml.dump(data, default_flow_style=False, allow_unicode=True)
            except ImportError:
                click.echo("⚠️  YAML no disponible, usando JSON", err=True)
                return json.dumps(data, indent=2, default=str, ensure_ascii=False)
        elif self.output_format == "table":
            return self._format_table(data)
        else:
            return str(data)
    
    def _format_table(self, data: Any) -> str:
        """Formatear datos como tabla."""
        if isinstance(data, dict):
            rows = []
            for key, value in data.items():
                rows.append(f"{key}: {value}")
            return "\n".join(rows)
        elif isinstance(data, list):
            if not data:
                return "No data"
            if isinstance(data[0], dict):
                headers = data[0].keys()
                rows = ["\t".join(headers)]
                for item in data:
                    rows.append("\t".join(str(item.get(h, "")) for h in headers))
                return "\n".join(rows)
        return str(data)


# Instancia global del contexto
cli_context = CLIContext()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Modo verbose')
@click.option('--format', '-f', 'output_format', 
              type=click.Choice(['json', 'yaml', 'table']), 
              default='json', help='Formato de salida')
@click.pass_context
def main(ctx, verbose, output_format):
    """
    🤖 AI Agents Framework CLI
    
    Interfaz de línea de comandos para gestionar agentes y workflows.
    """
    ctx.ensure_object(dict)
    cli_context.verbose = verbose
    cli_context.output_format = output_format


@main.group()
def agent():
    """🤖 Comandos para gestión de agentes."""
    pass


@main.group() 
def workflow():
    """📋 Comandos para gestión de workflows."""
    pass


@main.group()
def orchestrator():
    """🎭 Comandos para gestión del orquestrador."""
    pass


@agent.command()
@click.argument('agent_type', type=click.Choice([
    'pandas', 'sophisticated', 'qa', 'langchain', 'llm'
]))
@click.argument('task')
@click.option('--context', '-c', help='Contexto adicional para la tarea')
@click.option('--file', '-f', type=click.Path(exists=True), help='Archivo de entrada')
def run(agent_type, task, context, file):
    """🚀 Ejecutar una tarea con un agente específico."""
    
    async def _run():
        orchestrator = await cli_context.get_orchestrator()
        
        if cli_context.verbose:
            click.echo(f"🎯 Ejecutando tarea con agente: {agent_type}")
            click.echo(f"📝 Tarea: {task}")
        
        # Preparar entrada
        task_input = {"task": task}
        if context:
            task_input["context"] = context
        if file:
            task_input["file_path"] = str(file)
        
        # Mapear tipo de agente
        agent_map = {
            'pandas': 'pandas_agent',
            'sophisticated': 'sophisticated_agent', 
            'qa': 'qa_agent',
            'langchain': 'langchain_agent',
            'llm': 'llm_agent'
        }
        
        agent_id = agent_map[agent_type]
        agent = orchestrator.specialized_agents[agent_id]
        
        try:
            # Ejecutar tarea
            result = await agent.process_request(task_input)
            
            # Formatear resultado
            output = {
                "agent": agent_type,
                "task": task,
                "success": result.success,
                "result": result.content,
                "metadata": result.metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            if result.error:
                output["error"] = result.error
            
            click.echo(cli_context.format_output(output))
            
        except Exception as e:
            error_output = {
                "agent": agent_type,
                "task": task,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            click.echo(cli_context.format_output(error_output))
            sys.exit(1)
    
    asyncio.run(_run())


@agent.command()
def list():
    """📋 Listar agentes disponibles."""
    
    async def _list():
        orchestrator = await cli_context.get_orchestrator()
        
        agents_info = []
        for agent_id, agent in orchestrator.specialized_agents.items():
            agents_info.append({
                "id": agent_id,
                "type": agent.__class__.__name__,
                "status": "ready",
                "description": getattr(agent, '__doc__', '').split('\n')[0] if hasattr(agent, '__doc__') else ""
            })
        
        output = {
            "agents": agents_info,
            "total": len(agents_info),
            "timestamp": datetime.now().isoformat()
        }
        
        click.echo(cli_context.format_output(output))
    
    asyncio.run(_list())


@workflow.command()
@click.argument('workflow_id')
@click.option('--input-file', '-i', type=click.Path(exists=True), 
              help='Archivo JSON con parámetros de entrada')
@click.option('--param', '-p', multiple=True, 
              help='Parámetro en formato key=value')
def execute(workflow_id, input_file, param):
    """🚀 Ejecutar un workflow."""
    
    async def _execute():
        orchestrator = await cli_context.get_orchestrator()
        
        # Preparar parámetros
        params = {}
        
        # Cargar desde archivo
        if input_file:
            with open(input_file, 'r', encoding='utf-8') as f:
                params.update(json.load(f))
        
        # Agregar parámetros de línea de comandos
        for p in param:
            if '=' in p:
                key, value = p.split('=', 1)
                params[key] = value
            else:
                click.echo(f"❌ Formato inválido para parámetro: {p}", err=True)
                sys.exit(1)
        
        if cli_context.verbose:
            click.echo(f"🎯 Ejecutando workflow: {workflow_id}")
            click.echo(f"📋 Parámetros: {params}")
        
        try:
            # Ejecutar workflow
            execution = await orchestrator.execute_workflow(workflow_id, params)
            
            # Formatear resultado
            output = {
                "workflow_id": workflow_id,
                "execution_id": execution.execution_id,
                "status": execution.status.value,
                "progress": execution.progress,
                "start_time": execution.start_time,
                "end_time": execution.end_time,
                "results": {k: v.content for k, v in execution.results.items()},
                "errors": execution.errors,
                "timestamp": datetime.now().isoformat()
            }
            
            click.echo(cli_context.format_output(output))
            
            # Código de salida basado en estado
            if execution.status == WorkflowStatus.FAILED:
                sys.exit(1)
            
        except Exception as e:
            error_output = {
                "workflow_id": workflow_id,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            click.echo(cli_context.format_output(error_output))
            sys.exit(1)
    
    asyncio.run(_execute())


@workflow.command()
def list():
    """📋 Listar workflows disponibles."""
    
    async def _list():
        orchestrator = await cli_context.get_orchestrator()
        
        workflows_info = []
        for workflow_id, workflow_def in orchestrator.workflow_definitions.items():
            workflows_info.append({
                "id": workflow_id,
                "name": workflow_def.name,
                "description": workflow_def.description,
                "steps": len(workflow_def.steps),
                "max_parallel": workflow_def.max_parallel,
                "timeout_minutes": workflow_def.timeout_minutes
            })
        
        output = {
            "workflows": workflows_info,
            "total": len(workflows_info),
            "timestamp": datetime.now().isoformat()
        }
        
        click.echo(cli_context.format_output(output))
    
    asyncio.run(_list())


@workflow.command()
@click.argument('workflow_id')
def describe(workflow_id):
    """📄 Describir un workflow específico."""
    
    async def _describe():
        orchestrator = await cli_context.get_orchestrator()
        
        if workflow_id not in orchestrator.workflow_definitions:
            click.echo(f"❌ Workflow '{workflow_id}' no encontrado", err=True)
            sys.exit(1)
        
        workflow_def = orchestrator.workflow_definitions[workflow_id]
        
        steps_info = []
        for step in workflow_def.steps:
            steps_info.append({
                "id": step.step_id,
                "agent_type": step.agent_type,
                "task_config": step.task_config,
                "dependencies": step.dependencies,
                "max_retries": step.max_retries,
                "timeout_seconds": step.timeout_seconds
            })
        
        output = {
            "workflow_id": workflow_id,
            "name": workflow_def.name,
            "description": workflow_def.description,
            "steps": steps_info,
            "max_parallel": workflow_def.max_parallel,
            "timeout_minutes": workflow_def.timeout_minutes,
            "metadata": workflow_def.metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        click.echo(cli_context.format_output(output))
    
    asyncio.run(_describe())


@orchestrator.command()
def status():
    """📊 Estado del orquestrador."""
    
    async def _status():
        orchestrator = await cli_context.get_orchestrator()
        
        # Métricas del sistema
        system_metrics = orchestrator.system_metrics
        
        # Métricas de agentes
        agent_metrics = {}
        for agent_id in orchestrator.specialized_agents.keys():
            metrics = orchestrator.agent_metrics.get(agent_id)
            if metrics:
                agent_metrics[agent_id] = {
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "error_rate": metrics.error_rate,
                    "average_response_time": metrics.average_response_time,
                    "current_load": metrics.current_load,
                    "availability": metrics.availability
                }
        
        # Workflows activos
        active_workflows = {}
        for execution_id, execution in orchestrator.active_workflows.items():
            active_workflows[execution_id] = {
                "workflow_id": execution.workflow_def.workflow_id,
                "status": execution.status.value,
                "progress": execution.progress,
                "start_time": execution.start_time
            }
        
        output = {
            "orchestrator_status": "running",
            "system_metrics": system_metrics,
            "agent_metrics": agent_metrics,
            "active_workflows": active_workflows,
            "configuration": {
                "max_concurrent_workflows": orchestrator.max_concurrent_workflows,
                "load_balancing_enabled": orchestrator.load_balancing_enabled,
                "auto_scaling_enabled": orchestrator.auto_scaling_enabled
            },
            "timestamp": datetime.now().isoformat()
        }
        
        click.echo(cli_context.format_output(output))
    
    asyncio.run(_status())


@orchestrator.command()
def metrics():
    """📈 Métricas detalladas del sistema."""
    
    async def _metrics():
        orchestrator = await cli_context.get_orchestrator()
        
        # Métricas completas
        output = {
            "system_metrics": orchestrator.system_metrics,
            "agent_metrics": {
                agent_id: {
                    "agent_name": metrics.agent_name,
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "error_rate": metrics.error_rate,
                    "average_response_time": metrics.average_response_time,
                    "last_request_time": metrics.last_request_time,
                    "availability": metrics.availability,
                    "current_load": metrics.current_load,
                    "max_concurrent": metrics.max_concurrent
                }
                for agent_id, metrics in orchestrator.agent_metrics.items()
            },
            "workflow_history": [
                {
                    "execution_id": exec.execution_id,
                    "workflow_id": exec.workflow_def.workflow_id,
                    "status": exec.status.value,
                    "start_time": exec.start_time,
                    "end_time": exec.end_time,
                    "progress": exec.progress
                }
                for exec in orchestrator.workflow_history[-10:]  # Últimos 10
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        click.echo(cli_context.format_output(output))
    
    asyncio.run(_metrics())


@main.command()
@click.option('--config-file', '-c', type=click.Path(), 
              help='Archivo de configuración')
def config(config_file):
    """⚙️  Mostrar configuración actual."""
    
    config_data = {
        "ai_agents": {
            "openai_api_key": "***" if settings.openai_api_key else None,
            "default_model": settings.default_model,
            "log_level": settings.log_level,
            "memory_backend": settings.memory_backend,
            "data_dir": str(settings.data_dir),
            "cache_dir": str(settings.cache_dir)
        },
        "cli": {
            "verbose": cli_context.verbose,
            "output_format": cli_context.output_format
        },
        "timestamp": datetime.now().isoformat()
    }
    
    click.echo(cli_context.format_output(config_data))


@main.command()
@click.option('--host', '-h', default='0.0.0.0', help='Host para el servidor API')
@click.option('--port', '-p', default=8000, help='Puerto para el servidor API')
@click.option('--reload', is_flag=True, help='Modo de recarga automática')
def serve(host, port, reload):
    """🌐 Iniciar servidor API REST."""
    try:
        # Importar dinámicamente para evitar errores si FastAPI no está instalado
        from ai_agents.api.main import run_api
        
        if cli_context.verbose:
            click.echo(f"🌐 Iniciando servidor API en http://{host}:{port}")
            click.echo(f"📚 Documentación disponible en http://{host}:{port}/docs")
        
        run_api(host=host, port=port, reload=reload)
        
    except ImportError:
        click.echo("❌ FastAPI no está instalado. Instala las dependencias de API:", err=True)
        click.echo("   pip install fastapi uvicorn", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error al iniciar servidor API: {e}", err=True)
        sys.exit(1)


# Registrar comandos especializados
from .commands import register_commands
register_commands(main)


if __name__ == "__main__":
    main()
