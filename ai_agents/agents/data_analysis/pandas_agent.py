"""
Agente especializado en análisis de datos con pandas.
"""

import pandas as pd
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from ai_agents.core.base_agent import BaseAgent
from ai_agents.core.types import AgentResponse, Message
from ai_agents.agents.data_analysis.data_models import (
    AnalysisRequest, AnalysisResult, DataAnalysisState, DatasetInfo
)
from ai_agents.agents.data_analysis.workflows import DataAnalysisWorkflow
from ai_agents.agents.data_analysis.processors import CSVProcessor, StatsProcessor, VisualizationProcessor
from ai_agents.agents.data_analysis.tools import DataAnalysisTools

logger = logging.getLogger(__name__)


class PandasAgent(BaseAgent):
    """
    Agente especializado en análisis de datos usando pandas y herramientas estadísticas.
    
    Capacidades:
    - Carga y validación de datos desde archivos CSV
    - Análisis estadístico descriptivo
    - Análisis de correlaciones
    - Análisis de distribuciones
    - Detección de outliers
    - Creación de visualizaciones
    - Generación de reportes
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Inicializar componentes
        self.workflow = DataAnalysisWorkflow()
        self.csv_processor = CSVProcessor()
        self.stats_processor = StatsProcessor()
        self.viz_processor = VisualizationProcessor()
        self.tools = DataAnalysisTools()
        
        # Estado interno
        self.current_dataframe: Optional[pd.DataFrame] = None
        self.current_dataset_info: Optional[DatasetInfo] = None
        self.last_analysis_result: Optional[AnalysisResult] = None
    
    async def initialize(self) -> None:
        """Inicializa el agente."""
        logger.info("Inicializando PandasAgent")
        # No necesita inicialización especial por ahora
        pass
    
    async def process(self, input_data: Union[str, Dict[str, Any]]) -> AgentResponse:
        """
        Procesa una entrada y retorna una respuesta.
        
        Args:
            input_data: Mensaje o diccionario con datos de entrada
            
        Returns:
            Respuesta del agente
        """
        logger.debug(f"process() input_data: {input_data} (tipo: {type(input_data)})")
        
        if isinstance(input_data, str):
            message = input_data
            kwargs = {}
        elif isinstance(input_data, dict):
            message = input_data.get('message', '')
            kwargs = {k: v for k, v in input_data.items() if k != 'message'}
        else:
            message = str(input_data)
            kwargs = {}
        
        logger.debug(f"process() parseado - message: '{message}', kwargs: {kwargs}")
        return await self._safe_process(message, **kwargs)
    
    async def _safe_process(self, message: str, **kwargs) -> AgentResponse:
        """Procesa mensajes de manera segura."""
        try:
            # Determinar tipo de solicitud
            request_type = self._classify_request(message)
            
            if request_type == "load_data":
                return await self._handle_data_loading(message, **kwargs)
            elif request_type == "analyze_data":
                return await self._handle_data_analysis(message, **kwargs)
            elif request_type == "visualize_data":
                return await self._handle_visualization(message, **kwargs)
            elif request_type == "profile_data":
                return await self._handle_profiling(message, **kwargs)
            elif request_type == "export_results":
                return await self._handle_export(message, **kwargs)
            else:
                return await self._handle_general_query(message, **kwargs)
                
        except Exception as e:
            error_msg = f"Error procesando solicitud: {str(e)}"
            logger.error(error_msg)
            return AgentResponse(
                content=error_msg,
                metadata={"error": True, "exception": str(e)}
            )
    
    def _classify_request(self, message: str) -> str:
        """Clasifica el tipo de solicitud basado en el mensaje."""
        message_lower = message.lower()
        
        # Palabras clave para diferentes tipos de solicitudes (orden importa)
        analyze_keywords = ['analizar', 'analyze', 'estadísticas', 'statistics', 'correlación', 'correlation', 'análisis']
        visualize_keywords = ['gráfico', 'plot', 'visualizar', 'visualize', 'chart', 'histogram', 'scatter']
        profile_keywords = ['perfil', 'profile', 'resumen', 'summary', 'overview', 'información']
        export_keywords = ['exportar', 'export', 'guardar', 'save', 'reporte', 'report']
        load_keywords = ['cargar', 'load', 'archivo', 'file', 'csv', 'datos', 'data']
        
        # Verificar en orden de prioridad
        if any(keyword in message_lower for keyword in analyze_keywords):
            return "analyze_data"
        elif any(keyword in message_lower for keyword in visualize_keywords):
            return "visualize_data"
        elif any(keyword in message_lower for keyword in profile_keywords):
            return "profile_data"
        elif any(keyword in message_lower for keyword in export_keywords):
            return "export_results"
        elif any(keyword in message_lower for keyword in load_keywords):
            return "load_data"
        else:
            return "general"
    
    async def _handle_data_loading(self, message: str, **kwargs) -> AgentResponse:
        """Maneja solicitudes de carga de datos."""
        file_path = kwargs.get('file_path')
        sample_data = kwargs.get('sample_data')
        
        if not file_path and not sample_data:
            # Intentar extraer path del mensaje
            words = message.split()
            for word in words:
                if word.endswith('.csv') or '/' in word or '\\' in word:
                    file_path = word
                    break
        
        if not file_path and not sample_data:
            return AgentResponse(
                content="Por favor especifica la ruta del archivo CSV o el tipo de datos de muestra a crear.",
                metadata={"needs_file_path": True}
            )
        
        try:
            if file_path:
                # Cargar desde archivo
                dataframe = await asyncio.to_thread(self.csv_processor.load_from_file, file_path)
                
                self.current_dataframe = dataframe
                self.current_dataset_info = DatasetInfo(
                    name=Path(file_path).name,
                    shape=dataframe.shape,
                    dtypes={col: str(dtype) for col, dtype in dataframe.dtypes.items()},
                    memory_usage=dataframe.memory_usage(deep=True).sum(),
                    null_counts={col: int(count) for col, count in dataframe.isnull().sum().items()},
                    columns=list(dataframe.columns)
                )
                
                response_text = f"""Datos cargados exitosamente:
- Archivo: {file_path}
- Filas: {len(self.current_dataframe)}
- Columnas: {len(self.current_dataframe.columns)}
- Columnas: {', '.join(self.current_dataframe.columns)}
- Tipos de datos: {dict(self.current_dataframe.dtypes)}"""
                
            else:
                # Crear datos de muestra
                result = await asyncio.to_thread(self.csv_processor.create_sample_data, sample_data)
                
                self.current_dataframe = result['dataframe']
                self.current_dataset_info = result['dataset_info']
                
                response_text = f"""Datos de muestra creados:
- Tipo: {sample_data}
- Filas: {len(self.current_dataframe)}
- Columnas: {len(self.current_dataframe.columns)}
- Columnas: {', '.join(self.current_dataframe.columns)}"""
            
            return AgentResponse(
                content=response_text,
                metadata={
                    "data_loaded": True,
                    "rows": len(self.current_dataframe),
                    "columns": len(self.current_dataframe.columns),
                    "dataset_info": self.current_dataset_info.dict() if self.current_dataset_info else None
                }
            )
            
        except Exception as e:
            error_msg = f"Error cargando datos: {str(e)}"
            logger.error(error_msg)
            return AgentResponse(content=error_msg, metadata={"error": True})
    
    async def _handle_data_analysis(self, message: str, **kwargs) -> AgentResponse:
        """Maneja solicitudes de análisis de datos."""
        if self.current_dataframe is None:
            return AgentResponse(
                content="Primero debes cargar datos antes de analizarlos.",
                metadata={"needs_data": True}
            )
        
        try:
            # Realizar análisis estadístico directamente
            analyses = {}
            
            # Análisis descriptivo
            desc_stats = await asyncio.to_thread(
                self.stats_processor.descriptive_stats,
                self.current_dataframe,
                kwargs.get('target_columns')
            )
            analyses['descriptive'] = desc_stats
            
            # Análisis de correlación
            corr_analysis = await asyncio.to_thread(
                self.stats_processor.correlation_analysis,
                self.current_dataframe
            )
            analyses['correlation'] = corr_analysis
            
            # Crear resultado simplificado
            self.last_analysis_result = {
                'summary': 'Análisis completado exitosamente',
                'analyses': analyses,
                'visualizations': {},
                'errors': [],
                'recommendations': []
            }
            
            # Generar respuesta
            response_text = f"Análisis completado exitosamente"
            
            if desc_stats.get('summary'):
                response_text += f"\n\nEstadísticas descriptivas: {desc_stats['summary']}"
            
            if corr_analysis.get('summary'):
                response_text += f"\n\nCorrelaciones: {corr_analysis['summary']}"
            
            return AgentResponse(
                content=response_text,
                metadata={
                    "analysis_completed": True,
                    "result": self.last_analysis_result,
                    "has_visualizations": False,
                    "has_errors": False
                }
            )
            
        except Exception as e:
            error_msg = f"Error en análisis: {str(e)}"
            logger.error(error_msg)
            return AgentResponse(content=error_msg, metadata={"error": True})
    
    async def _handle_visualization(self, message: str, **kwargs) -> AgentResponse:
        """Maneja solicitudes de visualización."""
        if self.current_dataframe is None:
            return AgentResponse(
                content="Primero debes cargar datos antes de crear visualizaciones.",
                metadata={"needs_data": True}
            )
        
        viz_type = kwargs.get('viz_type', 'histogram')
        column = kwargs.get('column')
        
        if not column:
            # Sugerir columna basada en tipo de visualización
            if viz_type == 'histogram':
                numeric_cols = self.current_dataframe.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    column = numeric_cols[0]
            elif viz_type == 'bar_chart':
                categorical_cols = self.current_dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
                if categorical_cols:
                    column = categorical_cols[0]
        
        if not column and viz_type != 'heatmap':
            return AgentResponse(
                content=f"Por favor especifica una columna para el {viz_type}. Columnas disponibles: {', '.join(self.current_dataframe.columns)}",
                metadata={"needs_column": True, "available_columns": list(self.current_dataframe.columns)}
            )
        
        try:
            if viz_type == 'histogram':
                result = await asyncio.to_thread(self.viz_processor.create_histogram, self.current_dataframe, column)
            elif viz_type == 'bar_chart':
                result = await asyncio.to_thread(self.viz_processor.create_bar_chart, self.current_dataframe, column)
            elif viz_type == 'scatter':
                x_col = kwargs.get('x_column', column)
                y_col = kwargs.get('y_column')
                if not y_col:
                    numeric_cols = [c for c in self.current_dataframe.select_dtypes(include=['number']).columns if c != x_col]
                    if numeric_cols:
                        y_col = numeric_cols[0]
                
                if not y_col:
                    return AgentResponse(
                        content="Se necesitan dos columnas numéricas para scatter plot.",
                        metadata={"needs_two_columns": True}
                    )
                
                result = await asyncio.to_thread(self.viz_processor.create_scatter_plot, self.current_dataframe, x_col, y_col)
            elif viz_type == 'heatmap':
                result = await asyncio.to_thread(self.viz_processor.create_correlation_heatmap, self.current_dataframe)
            else:
                return AgentResponse(
                    content=f"Tipo de visualización '{viz_type}' no soportado. Tipos disponibles: histogram, bar_chart, scatter, heatmap",
                    metadata={"unsupported_viz_type": True}
                )
            
            if 'error' in result:
                return AgentResponse(
                    content=f"Error creando visualización: {result['error']}",
                    metadata={"error": True}
                )
            
            return AgentResponse(
                content=f"Visualización {viz_type} creada exitosamente.",
                metadata={
                    "visualization_created": True,
                    "viz_type": viz_type,
                    "image": result.get('image'),
                    "metadata": result.get('metadata')
                }
            )
            
        except Exception as e:
            error_msg = f"Error creando visualización: {str(e)}"
            logger.error(error_msg)
            return AgentResponse(content=error_msg, metadata={"error": True})
    
    async def _handle_profiling(self, message: str, **kwargs) -> AgentResponse:
        """Maneja solicitudes de perfilado de datos."""
        if self.current_dataframe is None:
            return AgentResponse(
                content="Primero debes cargar datos antes de generar un perfil.",
                metadata={"needs_data": True}
            )
        
        try:
            profile = await asyncio.to_thread(self.tools.quick_profile, self.current_dataframe)
            
            if 'error' in profile:
                return AgentResponse(
                    content=f"Error generando perfil: {profile['error']}",
                    metadata={"error": True}
                )
            
            # Generar texto descriptivo
            basic = profile['basic_info']
            types = profile['data_types']
            missing = profile['missing_data']
            
            response_text = f"""Perfil del Dataset:

Información Básica:
- Filas: {basic['rows']:,}
- Columnas: {basic['columns']}
- Tamaño: {basic['size_mb']:.2f} MB

Tipos de Datos:
- Numéricas: {types['numeric']}
- Categóricas: {types['categorical']}
- Fechas: {types['datetime']}
- Booleanas: {types['boolean']}

Datos Faltantes:
- Total valores faltantes: {missing['total_missing']:,}
- Porcentaje faltante: {missing['percentage_missing']:.2f}%"""

            if missing['columns_with_missing']:
                response_text += f"\n- Columnas con faltantes: {', '.join(missing['columns_with_missing'])}"
            
            return AgentResponse(
                content=response_text,
                metadata={
                    "profile_generated": True,
                    "profile": profile
                }
            )
            
        except Exception as e:
            error_msg = f"Error generando perfil: {str(e)}"
            logger.error(error_msg)
            return AgentResponse(content=error_msg, metadata={"error": True})
    
    async def _handle_export(self, message: str, **kwargs) -> AgentResponse:
        """Maneja solicitudes de exportación."""
        if self.last_analysis_result is None:
            return AgentResponse(
                content="No hay resultados de análisis para exportar. Ejecuta un análisis primero.",
                metadata={"needs_analysis": True}
            )
        
        output_path = kwargs.get('output_path', 'analysis_report.json')
        format_type = kwargs.get('format', 'json')
        
        try:
            result = await asyncio.to_thread(
                self.tools.export_analysis,
                self.last_analysis_result.dict(),
                output_path,
                format_type
            )
            
            if 'error' in result:
                return AgentResponse(
                    content=f"Error exportando: {result['error']}",
                    metadata={"error": True}
                )
            
            return AgentResponse(
                content=f"Reporte exportado exitosamente a {result['output_path']} (formato: {result['format']}, tamaño: {result['file_size']} bytes)",
                metadata={
                    "export_completed": True,
                    "export_info": result
                }
            )
            
        except Exception as e:
            error_msg = f"Error exportando: {str(e)}"
            logger.error(error_msg)
            return AgentResponse(content=error_msg, metadata={"error": True})
    
    async def _handle_general_query(self, message: str, **kwargs) -> AgentResponse:
        """Maneja consultas generales."""
        response_text = """Soy un agente especializado en análisis de datos. Puedo ayudarte con:

1. **Cargar datos**: Especifica la ruta de un archivo CSV o pide datos de muestra
2. **Analizar datos**: Realizo análisis estadísticos descriptivos y de correlación
3. **Crear visualizaciones**: Genero histogramas, gráficos de barras, scatter plots y mapas de calor
4. **Perfilar datos**: Proporciono un resumen completo del dataset
5. **Exportar resultados**: Guardo análisis en formato JSON, CSV o HTML

Ejemplos de uso:
- "Cargar datos de /ruta/archivo.csv"
- "Crear datos de muestra de ventas"
- "Analizar datos"
- "Crear histograma de columna edad"
- "Generar perfil de datos"
- "Exportar reporte a informe.html"
"""
        
        metadata = {"help_provided": True}
        
        if self.current_dataframe is not None:
            response_text += f"\n\nDatos actuales cargados: {len(self.current_dataframe)} filas, {len(self.current_dataframe.columns)} columnas"
            metadata["has_data"] = True
        
        return AgentResponse(content=response_text, metadata=metadata)
    
    async def process_request(self, input_data: Union[str, Dict, Message]) -> AgentResponse:
        """
        Interfaz pública para procesar requests.
        Delega al método process con manejo seguro.
        
        Args:
            input_data: Datos de entrada
            
        Returns:
            AgentResponse con la respuesta del agente
        """
        # Debug logging para diagnosticar el problema
        logger.debug(f"process_request recibió input_data: {input_data} (tipo: {type(input_data)})")
        return await self.process(input_data)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Retorna las capacidades del agente."""
        return {
            "agent_type": "data_analysis",
            "supported_formats": ["csv"],
            "analysis_types": ["descriptive", "correlation", "distribution"],
            "visualization_types": ["histogram", "bar_chart", "scatter", "heatmap", "pie_chart", "box_plot"],
            "export_formats": ["json", "csv", "html"],
            "sample_data_types": ["sales", "customers", "financial", "sensors", "survey"],
            "tools": ["quick_profile", "outlier_detection", "column_comparison", "type_suggestions"]
        }
