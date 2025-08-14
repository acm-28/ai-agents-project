"""
Agente sofisticado con workflows de LangGraph para análisis de texto.

Este agente implementa un pipeline complejo de procesamiento de texto
usando LangGraph StateGraph con múltiples nodos especializados.
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict, Union
from dataclasses import dataclass

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

try:
    from langgraph.graph import StateGraph, START, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    StateGraph = None
    START = END = None

from ...core.base_agent import BaseAgent
from ...core.types import AgentResponse, Message

logger = logging.getLogger(__name__)


class TextAnalysisState(TypedDict):
    """Estado para el workflow de análisis de texto."""
    text: str
    classification: Optional[str]
    entities: Optional[List[str]]
    summary: Optional[str]
    confidence_scores: Optional[Dict[str, float]]
    metadata: Optional[Dict[str, Any]]


@dataclass
class AnalysisResult:
    """Resultado del análisis de texto."""
    classification: str
    entities: List[str]
    summary: str
    confidence_scores: Dict[str, float]
    processing_steps: List[str]


class SophisticatedAgent(BaseAgent):
    """
    Agente sofisticado para análisis complejo de texto usando LangGraph.
    
    Implementa un pipeline de múltiples etapas:
    1. Clasificación de texto
    2. Extracción de entidades
    3. Generación de resumen
    4. Análisis de confianza
    """
    
    def __init__(self, agent_id: str = None, **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        self.llm: Optional[ChatOpenAI] = None
        self.workflow_app = None
        self.last_analysis: Optional[AnalysisResult] = None
        
        # Configuración de prompts
        self.prompts = {
            'classification': PromptTemplate(
                input_variables=['text'],
                template="""Classify the following text into one of these categories:
- News: News articles, press releases, current events
- Blog: Personal opinions, informal writing, blog posts
- Research: Academic papers, scientific reports, formal studies
- Business: Corporate communications, reports, proposals
- Other: Anything that doesn't fit the above categories

Text: {text}

Category (respond with just the category name):"""
            ),
            'entities': PromptTemplate(
                input_variables=['text'],
                template="""Extract all named entities from the following text.
Focus on: Person names, Organizations, Locations, Products, Events.

Text: {text}

Entities (comma-separated list):"""
            ),
            'summary': PromptTemplate(
                input_variables=['text'],
                template="""Summarize the following text in 1-2 concise sentences.
Capture the main point and key information.

Text: {text}

Summary:"""
            )
        }
    
    async def initialize(self) -> None:
        """Inicializa el agente y construye el workflow."""
        # No llamar super().initialize() porque es abstracto
        
        if not HAS_LANGGRAPH:
            logger.warning("LangGraph no está disponible. Funcionalidad limitada.")
            return
        
        # Inicializar LLM
        try:
            from ...config.settings import settings
            self.llm = ChatOpenAI(
                model=settings.default_model,
                temperature=0.1,
                openai_api_key=settings.openai_api_key
            )
            logger.info("LLM inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando LLM: {e}")
            raise
        
        # Construir workflow
        self._build_workflow()
        logger.info("SophisticatedAgent inicializado correctamente")
    
    def _build_workflow(self) -> None:
        """Construye el workflow de LangGraph."""
        if not HAS_LANGGRAPH:
            return
        
        # Crear StateGraph
        workflow = StateGraph(TextAnalysisState)
        
        # Agregar nodos
        workflow.add_node('classification', self._classification_node)
        workflow.add_node('entity_extraction', self._entity_extraction_node)
        workflow.add_node('summarization', self._summarization_node)
        workflow.add_node('confidence_analysis', self._confidence_analysis_node)
        
        # Definir flujo
        workflow.add_edge(START, 'classification')
        workflow.add_edge('classification', 'entity_extraction')
        workflow.add_edge('entity_extraction', 'summarization')
        workflow.add_edge('summarization', 'confidence_analysis')
        workflow.add_edge('confidence_analysis', END)
        
        # Compilar
        self.workflow_app = workflow.compile()
        logger.info("Workflow compilado exitosamente")
    
    async def _classification_node(self, state: TextAnalysisState) -> Dict[str, Any]:
        """Nodo de clasificación de texto."""
        try:
            message = HumanMessage(
                content=self.prompts['classification'].format(text=state['text'])
            )
            response = await self.llm.ainvoke([message])
            classification = response.content.strip()
            
            logger.debug(f"Clasificación: {classification}")
            return {'classification': classification}
            
        except Exception as e:
            logger.error(f"Error en clasificación: {e}")
            return {'classification': 'Error'}
    
    async def _entity_extraction_node(self, state: TextAnalysisState) -> Dict[str, Any]:
        """Nodo de extracción de entidades."""
        try:
            message = HumanMessage(
                content=self.prompts['entities'].format(text=state['text'])
            )
            response = await self.llm.ainvoke([message])
            entities_str = response.content.strip()
            
            # Procesar entidades
            entities = [
                entity.strip() 
                for entity in entities_str.split(',') 
                if entity.strip() and entity.strip().lower() not in ['none', 'n/a', 'no entities']
            ]
            
            logger.debug(f"Entidades extraídas: {len(entities)}")
            return {'entities': entities}
            
        except Exception as e:
            logger.error(f"Error en extracción de entidades: {e}")
            return {'entities': []}
    
    async def _summarization_node(self, state: TextAnalysisState) -> Dict[str, Any]:
        """Nodo de generación de resumen."""
        try:
            message = HumanMessage(
                content=self.prompts['summary'].format(text=state['text'])
            )
            response = await self.llm.ainvoke([message])
            summary = response.content.strip()
            
            logger.debug(f"Resumen generado: {len(summary)} caracteres")
            return {'summary': summary}
            
        except Exception as e:
            logger.error(f"Error en resumen: {e}")
            return {'summary': 'Error generando resumen'}
    
    async def _confidence_analysis_node(self, state: TextAnalysisState) -> Dict[str, Any]:
        """Nodo de análisis de confianza."""
        try:
            # Calcular scores de confianza basados en longitud y completitud
            text_length = len(state.get('text', ''))
            has_classification = bool(state.get('classification'))
            has_entities = bool(state.get('entities'))
            has_summary = bool(state.get('summary'))
            
            confidence_scores = {
                'classification': 0.8 if has_classification and state['classification'] != 'Error' else 0.2,
                'entities': min(0.9, len(state.get('entities', [])) * 0.3 + 0.3) if has_entities else 0.1,
                'summary': 0.8 if has_summary and len(state.get('summary', '')) > 10 else 0.2,
                'overall': 0.7 if text_length > 50 else 0.4
            }
            
            metadata = {
                'text_length': text_length,
                'processing_complete': True,
                'entity_count': len(state.get('entities', [])),
                'summary_length': len(state.get('summary', ''))
            }
            
            return {
                'confidence_scores': confidence_scores,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de confianza: {e}")
            return {
                'confidence_scores': {'overall': 0.1},
                'metadata': {'error': str(e)}
            }
    
    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Procesa una solicitud de análisis de texto.
        
        Args:
            request: Diccionario con 'message' y opcionalmente 'text'
            
        Returns:
            AgentResponse con el resultado del análisis
        """
        message = request.get('message', '')
        text_to_analyze = request.get('text', message)
        
        if not text_to_analyze.strip():
            return AgentResponse(
                content="Por favor proporciona un texto para analizar.",
                metadata={"error": True, "needs_text": True}
            )
        
        if not HAS_LANGGRAPH or not self.workflow_app:
            return await self._fallback_analysis(text_to_analyze)
        
        try:
            # Ejecutar workflow
            state_input = TextAnalysisState(
                text=text_to_analyze,
                classification=None,
                entities=None,
                summary=None,
                confidence_scores=None,
                metadata=None
            )
            
            result = await self.workflow_app.ainvoke(state_input)
            
            # Verificar si hubo errores críticos
            classification = result.get('classification', 'Unknown')
            entities = result.get('entities', [])
            summary = result.get('summary', 'No summary available')
            
            # Si la clasificación es "Error" y no hay entidades ni resumen válido, es un error crítico
            if (classification == 'Error' and 
                len(entities) == 0 and 
                ('error' in summary.lower() or summary == 'Error generando resumen')):
                return AgentResponse(
                    content="Error procesando el texto. El LLM no está disponible o falló.",
                    metadata={"error": True, "llm_failure": True}
                )
            
            # Crear resultado estructurado
            analysis_result = AnalysisResult(
                classification=classification,
                entities=entities,
                summary=summary,
                confidence_scores=result.get('confidence_scores', {}),
                processing_steps=['classification', 'entity_extraction', 'summarization', 'confidence_analysis']
            )
            
            self.last_analysis = analysis_result
            
            # Formatear respuesta
            response_text = self._format_analysis_response(analysis_result)
            
            return AgentResponse(
                content=response_text,
                metadata={
                    "analysis_complete": True,
                    "classification": analysis_result.classification,
                    "entity_count": len(analysis_result.entities),
                    "confidence_scores": analysis_result.confidence_scores,
                    "result": {
                        "classification": analysis_result.classification,
                        "entities": analysis_result.entities,
                        "summary": analysis_result.summary,
                        "confidence_scores": analysis_result.confidence_scores
                    }
                }
            )
            
        except Exception as e:
            error_msg = f"Error en análisis: {str(e)}"
            logger.error(error_msg)
            return AgentResponse(content=error_msg, metadata={"error": True})
    
    async def _fallback_analysis(self, text: str) -> AgentResponse:
        """Análisis de respaldo cuando LangGraph no está disponible."""
        try:
            if not self.llm:
                return AgentResponse(
                    content="LLM no está disponible para análisis.",
                    metadata={"error": True, "missing_llm": True}
                )
            
            # Análisis simple sin workflow
            message = HumanMessage(content=f"""
Analiza el siguiente texto y proporciona:
1. Clasificación (News/Blog/Research/Business/Other)
2. Entidades principales
3. Resumen breve

Texto: {text}

Respuesta estructurada:
""")
            
            response = await self.llm.ainvoke([message])
            
            return AgentResponse(
                content=f"Análisis simple completado:\n\n{response.content}",
                metadata={
                    "analysis_complete": True,
                    "fallback_mode": True,
                    "classification": "Unknown",
                    "entity_count": 0
                }
            )
            
        except Exception as e:
            error_msg = f"Error en análisis de respaldo: {str(e)}"
            logger.error(error_msg)
            return AgentResponse(content=error_msg, metadata={"error": True})
    
    def _format_analysis_response(self, result: AnalysisResult) -> str:
        """Formatea la respuesta del análisis."""
        confidence_str = ", ".join([
            f"{k}: {v:.1%}" for k, v in result.confidence_scores.items()
        ])
        
        entities_str = ", ".join(result.entities) if result.entities else "Ninguna entidad detectada"
        
        return f"""📊 **Análisis de Texto Completado**

🏷️ **Clasificación:** {result.classification}

👥 **Entidades:** {entities_str}

📝 **Resumen:** {result.summary}

📈 **Confianza:** {confidence_str}

✅ **Procesamiento:** {len(result.processing_steps)} etapas completadas"""
    
    def get_capabilities(self) -> List[str]:
        """Obtiene las capacidades del agente."""
        capabilities = [
            "Clasificación de texto",
            "Extracción de entidades", 
            "Resumen automático",
            "Análisis de confianza"
        ]
        
        if HAS_LANGGRAPH:
            capabilities.append("Workflows complejos con LangGraph")
        else:
            capabilities.append("Análisis simple (LangGraph no disponible)")
            
        return capabilities
    
    def get_last_analysis(self) -> Optional[AnalysisResult]:
        """Obtiene el último análisis realizado."""
        return self.last_analysis
    
    async def process_request(self, input_data: Union[str, Dict, Message]) -> AgentResponse:
        """
        Interfaz pública para procesar requests.
        Delega al método process con manejo seguro.
        
        Args:
            input_data: Datos de entrada
            
        Returns:
            AgentResponse con la respuesta del agente
        """
        return await self._safe_process(input_data)
