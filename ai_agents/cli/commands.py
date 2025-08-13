"""
Comandos especializados del CLI.
"""

import click
import asyncio
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .main import cli_context


@click.group()
def data():
    """📊 Comandos para análisis de datos."""
    pass


@click.group()
def text():
    """📝 Comandos para procesamiento de texto."""
    pass


@click.group()
def chat():
    """💬 Comandos para chat interactivo."""
    pass


@data.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--operations', '-o', multiple=True,
              type=click.Choice(['describe', 'info', 'head', 'tail', 'correlations']),
              default=['describe'], help='Operaciones a realizar')
@click.option('--output', '-out', type=click.Path(), help='Archivo de salida')
def analyze(file_path, operations, output):
    """🔍 Analizar archivo de datos."""
    
    async def _analyze():
        orchestrator = await cli_context.get_orchestrator()
        pandas_agent = orchestrator.specialized_agents['pandas_agent']
        
        if cli_context.verbose:
            click.echo(f"📊 Analizando archivo: {file_path}")
            click.echo(f"🔧 Operaciones: {', '.join(operations)}")
        
        try:
            # Ejecutar análisis
            task_input = {
                "file_path": str(file_path),
                "operations": list(operations)
            }
            
            result = await pandas_agent.process_request(task_input)
            
            output_data = {
                "file_path": str(file_path),
                "operations": list(operations),
                "success": result.success,
                "analysis": result.content,
                "metadata": result.metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            if result.error:
                output_data["error"] = result.error
            
            # Guardar en archivo si se especifica
            if output:
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, default=str, ensure_ascii=False)
                click.echo(f"💾 Resultados guardados en: {output}")
            
            click.echo(cli_context.format_output(output_data))
            
        except Exception as e:
            error_output = {
                "file_path": str(file_path),
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            click.echo(cli_context.format_output(error_output))
            raise click.ClickException(f"Error en análisis: {e}")
    
    asyncio.run(_analyze())


@data.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--columns', '-c', multiple=True, help='Columnas a procesar')
@click.option('--operation', '-op', 
              type=click.Choice(['clean', 'normalize', 'deduplicate', 'fill_na']),
              default='clean', help='Operación de limpieza')
@click.option('--output', '-out', type=click.Path(), help='Archivo de salida')
def clean(file_path, columns, operation, output):
    """🧹 Limpiar datos."""
    
    async def _clean():
        orchestrator = await cli_context.get_orchestrator()
        pandas_agent = orchestrator.specialized_agents['pandas_agent']
        
        if cli_context.verbose:
            click.echo(f"🧹 Limpiando archivo: {file_path}")
            click.echo(f"🔧 Operación: {operation}")
        
        try:
            task_input = {
                "file_path": str(file_path),
                "operation": operation,
                "columns": list(columns) if columns else None,
                "output_path": str(output) if output else None
            }
            
            result = await pandas_agent.process_request(task_input)
            
            output_data = {
                "file_path": str(file_path),
                "operation": operation,
                "success": result.success,
                "result": result.content,
                "metadata": result.metadata,
                "output_file": str(output) if output else None,
                "timestamp": datetime.now().isoformat()
            }
            
            if result.error:
                output_data["error"] = result.error
            
            click.echo(cli_context.format_output(output_data))
            
        except Exception as e:
            error_output = {
                "file_path": str(file_path),
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            click.echo(cli_context.format_output(error_output))
            raise click.ClickException(f"Error en limpieza: {e}")
    
    asyncio.run(_clean())


@text.command()
@click.argument('text_input')
@click.option('--operation', '-op',
              type=click.Choice(['summarize', 'sentiment', 'keywords', 'entities']),
              default='summarize', help='Operación de procesamiento')
@click.option('--language', '-l', default='es', help='Idioma del texto')
def process(text_input, operation, language):
    """🔤 Procesar texto."""
    
    async def _process():
        orchestrator = await cli_context.get_orchestrator()
        sophisticated_agent = orchestrator.specialized_agents['sophisticated_agent']
        
        if cli_context.verbose:
            click.echo(f"🔤 Procesando texto con operación: {operation}")
        
        try:
            task_input = {
                "text": text_input,
                "operation": operation,
                "language": language
            }
            
            result = await sophisticated_agent.process_request(task_input)
            
            output_data = {
                "text_preview": text_input[:100] + "..." if len(text_input) > 100 else text_input,
                "operation": operation,
                "language": language,
                "success": result.success,
                "result": result.content,
                "metadata": result.metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            if result.error:
                output_data["error"] = result.error
            
            click.echo(cli_context.format_output(output_data))
            
        except Exception as e:
            error_output = {
                "operation": operation,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            click.echo(cli_context.format_output(error_output))
            raise click.ClickException(f"Error en procesamiento: {e}")
    
    asyncio.run(_process())


@text.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--chunk-size', '-cs', default=1000, help='Tamaño de chunks')
@click.option('--overlap', '-ov', default=200, help='Solapamiento entre chunks')
@click.option('--output', '-out', type=click.Path(), help='Archivo de salida')
def analyze_document(file_path, chunk_size, overlap, output):
    """📄 Analizar documento completo."""
    
    async def _analyze_document():
        orchestrator = await cli_context.get_orchestrator()
        sophisticated_agent = orchestrator.specialized_agents['sophisticated_agent']
        
        if cli_context.verbose:
            click.echo(f"📄 Analizando documento: {file_path}")
        
        try:
            # Leer archivo
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                raise click.ClickException(f"Formato de archivo no soportado: {file_path_obj.suffix}")
            
            task_input = {
                "document_content": content,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "analysis_type": "comprehensive"
            }
            
            result = await sophisticated_agent.process_request(task_input)
            
            output_data = {
                "file_path": str(file_path),
                "document_size": len(content),
                "chunk_size": chunk_size,
                "overlap": overlap,
                "success": result.success,
                "analysis": result.content,
                "metadata": result.metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            if result.error:
                output_data["error"] = result.error
            
            # Guardar en archivo si se especifica
            if output:
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, default=str, ensure_ascii=False)
                click.echo(f"💾 Análisis guardado en: {output}")
            
            click.echo(cli_context.format_output(output_data))
            
        except Exception as e:
            error_output = {
                "file_path": str(file_path),
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            click.echo(cli_context.format_output(error_output))
            raise click.ClickException(f"Error en análisis de documento: {e}")
    
    asyncio.run(_analyze_document())


@chat.command()
@click.option('--agent', '-a',
              type=click.Choice(['qa', 'langchain', 'llm']),
              default='qa', help='Agente para el chat')
@click.option('--context', '-c', help='Contexto inicial')
@click.option('--memory/--no-memory', default=True, help='Usar memoria conversacional')
def interactive(agent, context, memory):
    """💬 Chat interactivo con agentes."""
    
    async def _interactive():
        orchestrator = await cli_context.get_orchestrator()
        
        agent_map = {
            'qa': 'qa_agent',
            'langchain': 'langchain_agent',
            'llm': 'llm_agent'
        }
        
        selected_agent = orchestrator.specialized_agents[agent_map[agent]]
        
        click.echo(f"💬 Iniciando chat interactivo con {agent}")
        click.echo("📝 Escribe 'quit', 'exit' o 'salir' para terminar")
        
        if context:
            click.echo(f"🎯 Contexto inicial: {context}")
        
        conversation_history = []
        if context:
            conversation_history.append({"role": "system", "content": context})
        
        while True:
            try:
                # Obtener entrada del usuario
                user_input = click.prompt("👤", type=str)
                
                # Comandos de salida
                if user_input.lower() in ['quit', 'exit', 'salir']:
                    click.echo("👋 ¡Hasta luego!")
                    break
                
                # Agregar a historial
                if memory:
                    conversation_history.append({"role": "user", "content": user_input})
                
                # Preparar entrada para el agente
                task_input = {
                    "query": user_input,
                    "conversation_history": conversation_history if memory else []
                }
                
                # Ejecutar consulta
                result = await selected_agent.process_request(task_input)
                
                if result.success:
                    click.echo(f"🤖 {result.content}")
                    
                    # Agregar respuesta al historial
                    if memory:
                        conversation_history.append({"role": "assistant", "content": result.content})
                else:
                    click.echo(f"❌ Error: {result.error}")
                
            except KeyboardInterrupt:
                click.echo("\n👋 Chat interrumpido por el usuario")
                break
            except Exception as e:
                click.echo(f"❌ Error: {e}")
    
    asyncio.run(_interactive())


@chat.command()
@click.argument('question')
@click.option('--context', '-c', help='Contexto para la pregunta')
@click.option('--agent', '-a',
              type=click.Choice(['qa', 'langchain', 'llm']),
              default='qa', help='Agente para responder')
def ask(question, context, agent):
    """❓ Hacer una pregunta específica."""
    
    async def _ask():
        orchestrator = await cli_context.get_orchestrator()
        
        agent_map = {
            'qa': 'qa_agent',
            'langchain': 'langchain_agent', 
            'llm': 'llm_agent'
        }
        
        selected_agent = orchestrator.specialized_agents[agent_map[agent]]
        
        if cli_context.verbose:
            click.echo(f"❓ Preguntando a {agent}: {question}")
        
        try:
            task_input = {
                "query": question,
                "context": context
            }
            
            result = await selected_agent.process_request(task_input)
            
            output_data = {
                "question": question,
                "context": context,
                "agent": agent,
                "success": result.success,
                "answer": result.content,
                "metadata": result.metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            if result.error:
                output_data["error"] = result.error
            
            click.echo(cli_context.format_output(output_data))
            
        except Exception as e:
            error_output = {
                "question": question,
                "agent": agent,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            click.echo(cli_context.format_output(error_output))
            raise click.ClickException(f"Error en consulta: {e}")
    
    asyncio.run(_ask())


# Registrar comandos en el CLI principal
def register_commands(main_cli):
    """Registrar comandos especializados."""
    main_cli.add_command(data)
    main_cli.add_command(text)
    main_cli.add_command(chat)
