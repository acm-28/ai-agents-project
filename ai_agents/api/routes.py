"""
Rutas adicionales para la API REST.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from .models import *
from .main import get_orchestrator

# Logger
logger = logging.getLogger(__name__)

# Router para chat
chat_router = APIRouter(prefix="/chat", tags=["Chat"])

# Router para análisis de datos
data_router = APIRouter(prefix="/data", tags=["Data Analysis"])

# Router para procesamiento de texto
text_router = APIRouter(prefix="/text", tags=["Text Processing"])


# ===============================
# RUTAS DE CHAT
# ===============================

@chat_router.post("/message", response_model=ChatResponse)
async def send_chat_message(request: ChatRequest):
    """Enviar mensaje de chat a un agente."""
    orch = await get_orchestrator()
    
    # Mapear tipo de agente
    agent_map = {
        'qa': 'qa_agent',
        'langchain': 'langchain_agent',
        'llm': 'llm_agent'
    }
    
    if request.agent_type not in agent_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent type: {request.agent_type}"
        )
    
    agent = orch.specialized_agents[agent_map[request.agent_type]]
    
    try:
        # Preparar entrada
        task_input = {
            "query": request.message,
            "context": request.context,
            "conversation_history": [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
        }
        
        if request.options:
            task_input.update(request.options)
        
        # Ejecutar consulta
        result = await agent.process_request(task_input)
        
        return ChatResponse(
            message=result.content,
            agent_type=request.agent_type,
            success=result.success,
            metadata=result.metadata,
            error=result.error,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error in chat with agent {request.agent_type}: {e}")
        return ChatResponse(
            message="",
            agent_type=request.agent_type,
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )


@chat_router.get("/agents")
async def list_chat_agents():
    """Listar agentes disponibles para chat."""
    return {
        "agents": [
            {
                "type": "qa",
                "name": "Memory Q&A Agent",
                "description": "Agente con memoria conversacional para preguntas y respuestas"
            },
            {
                "type": "langchain",
                "name": "LangChain Chat Agent", 
                "description": "Agente basado en LangChain para conversaciones"
            },
            {
                "type": "llm",
                "name": "LLM Chat Agent",
                "description": "Agente de chat directo con modelo de lenguaje"
            }
        ],
        "timestamp": datetime.now()
    }


# ===============================
# RUTAS DE ANÁLISIS DE DATOS
# ===============================

@data_router.post("/analyze", response_model=DataAnalysisResponse)
async def analyze_data(request: DataAnalysisRequest):
    """Analizar datos usando el PandasAgent."""
    orch = await get_orchestrator()
    pandas_agent = orch.specialized_agents['pandas_agent']
    
    try:
        # Preparar entrada
        task_input = {
            "operations": request.operations
        }
        
        if request.file_path:
            task_input["file_path"] = request.file_path
        elif request.data_url:
            task_input["data_url"] = request.data_url
        elif request.data_content:
            task_input["data_content"] = request.data_content
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide file_path, data_url, or data_content"
            )
        
        if request.options:
            task_input.update(request.options)
        
        # Ejecutar análisis
        result = await pandas_agent.process_request(task_input)
        
        return DataAnalysisResponse(
            file_path=request.file_path,
            operations=request.operations,
            success=result.success,
            results=result.metadata if result.success else None,
            metadata=result.metadata,
            error=result.error,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error in data analysis: {e}")
        return DataAnalysisResponse(
            file_path=request.file_path,
            operations=request.operations,
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )


@data_router.post("/clean")
async def clean_data(
    file_path: str,
    operation: str = "clean",
    columns: Optional[list] = None,
    output_path: Optional[str] = None
):
    """Limpiar datos usando el PandasAgent."""
    orch = await get_orchestrator()
    pandas_agent = orch.specialized_agents['pandas_agent']
    
    try:
        task_input = {
            "file_path": file_path,
            "operation": operation,
            "columns": columns,
            "output_path": output_path
        }
        
        result = await pandas_agent.process_request(task_input)
        
        return {
            "file_path": file_path,
            "operation": operation,
            "success": result.success,
            "result": result.content,
            "metadata": result.metadata,
            "error": result.error,
            "output_file": output_path,
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Error in data cleaning: {e}")
        return {
            "file_path": file_path,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now()
        }


@data_router.get("/operations")
async def list_data_operations():
    """Listar operaciones disponibles para análisis de datos."""
    return {
        "analysis_operations": [
            "describe", "info", "head", "tail", "correlations",
            "missing_values", "dtypes", "value_counts"
        ],
        "cleaning_operations": [
            "clean", "normalize", "deduplicate", "fill_na",
            "remove_outliers", "standardize"
        ],
        "transformation_operations": [
            "group_by", "pivot", "melt", "merge", "sort"
        ],
        "timestamp": datetime.now()
    }


# ===============================
# RUTAS DE PROCESAMIENTO DE TEXTO
# ===============================

@text_router.post("/process", response_model=TextProcessingResponse)
async def process_text(request: TextProcessingRequest):
    """Procesar texto usando el SophisticatedAgent."""
    orch = await get_orchestrator()
    sophisticated_agent = orch.specialized_agents['sophisticated_agent']
    
    try:
        # Preparar entrada
        task_input = {
            "text": request.text,
            "operation": request.operation,
            "language": request.language
        }
        
        if request.options:
            task_input.update(request.options)
        
        # Ejecutar procesamiento
        result = await sophisticated_agent.process_request(task_input)
        
        text_preview = request.text[:100] + "..." if len(request.text) > 100 else request.text
        
        return TextProcessingResponse(
            text_preview=text_preview,
            operation=request.operation,
            language=request.language,
            success=result.success,
            result=result.content,
            metadata=result.metadata,
            error=result.error,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error in text processing: {e}")
        return TextProcessingResponse(
            text_preview=request.text[:100] + "..." if len(request.text) > 100 else request.text,
            operation=request.operation,
            language=request.language,
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )


@text_router.post("/analyze-document")
async def analyze_document(
    content: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    analysis_type: str = "comprehensive"
):
    """Analizar documento completo."""
    orch = await get_orchestrator()
    sophisticated_agent = orch.specialized_agents['sophisticated_agent']
    
    try:
        task_input = {
            "document_content": content,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "analysis_type": analysis_type
        }
        
        result = await sophisticated_agent.process_request(task_input)
        
        return {
            "document_size": len(content),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "analysis_type": analysis_type,
            "success": result.success,
            "analysis": result.content,
            "metadata": result.metadata,
            "error": result.error,
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Error in document analysis: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now()
        }


@text_router.get("/operations")
async def list_text_operations():
    """Listar operaciones disponibles para procesamiento de texto."""
    return {
        "processing_operations": [
            "summarize", "sentiment", "keywords", "entities",
            "classification", "translation", "extraction"
        ],
        "analysis_operations": [
            "comprehensive", "structure", "topics", "readability"
        ],
        "supported_languages": [
            "es", "en", "fr", "de", "it", "pt", "auto"
        ],
        "timestamp": datetime.now()
    }


# ===============================
# RUTAS DE UTILIDADES
# ===============================

utilities_router = APIRouter(prefix="/utils", tags=["Utilities"])


@utilities_router.get("/status")
async def system_status():
    """Estado general del sistema."""
    orch = await get_orchestrator()
    
    return {
        "system": "healthy",
        "orchestrator": "running",
        "agents_count": len(orch.specialized_agents),
        "workflows_count": len(orch.workflow_definitions),
        "active_workflows": len(orch.active_workflows),
        "uptime": datetime.now() - orch.system_metrics.get('uptime_start', datetime.now()),
        "timestamp": datetime.now()
    }


@utilities_router.get("/version")
async def get_version():
    """Información de versión."""
    return {
        "framework_version": "1.0.0",
        "api_version": "1.0.0",
        "python_version": "3.13.5",
        "agents": {
            "pandas_agent": "1.0.0",
            "sophisticated_agent": "1.0.0",
            "qa_agent": "1.0.0",
            "langchain_agent": "1.0.0",
            "llm_agent": "1.0.0",
            "orchestrator": "1.0.0",
            "advanced_orchestrator": "1.0.0"
        },
        "timestamp": datetime.now()
    }


# Lista de todos los routers para registrar
all_routers = [
    chat_router,
    data_router,
    text_router,
    utilities_router
]
