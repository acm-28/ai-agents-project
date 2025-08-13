"""
Workflows para análisis de datos usando LangGraph.
"""

from typing import Dict, Any, List, Optional, TypedDict
import asyncio
import logging

from langgraph.graph import StateGraph, START, END
from ai_agents.agents.data_analysis.data_models import DataAnalysisState, AnalysisRequest, AnalysisResult
from ai_agents.agents.data_analysis.processors import CSVProcessor, StatsProcessor, VisualizationProcessor

logger = logging.getLogger(__name__)


class DataAnalysisWorkflow:
    """Workflow principal para análisis de datos."""
    
    def __init__(self):
        self.csv_processor = CSVProcessor()
        self.stats_processor = StatsProcessor()
        self.viz_processor = VisualizationProcessor()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Construye el grafo de workflow."""
        workflow = StateGraph(DataAnalysisState)
        
        # Nodos del workflow
        workflow.add_node("load_data", self.load_data_node)
        workflow.add_node("validate_data", self.validate_data_node)
        workflow.add_node("analyze_data", self.analyze_data_node)
        workflow.add_node("create_visualizations", self.create_visualizations_node)
        workflow.add_node("generate_report", self.generate_report_node)
        
        # Definir flujo
        workflow.add_edge(START, "load_data")
        workflow.add_edge("load_data", "validate_data")
        workflow.add_edge("validate_data", "analyze_data")
        workflow.add_edge("analyze_data", "create_visualizations")
        workflow.add_edge("create_visualizations", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    async def load_data_node(self, state: DataAnalysisState) -> DataAnalysisState:
        """Nodo para cargar datos."""
        logger.info("Iniciando carga de datos")
        
        try:
            request = state.request
            
            if request.file_path:
                # Cargar desde archivo
                if request.file_path.endswith('.csv'):
                    result = await asyncio.to_thread(
                        self.csv_processor.load_from_file, 
                        request.file_path
                    )
                    
                    if 'error' in result:
                        state.errors.append(f"Error cargando archivo: {result['error']}")
                        return state
                    
                    state.dataset = result['dataset_info']
                    state.dataframe = result['dataframe']
                    logger.info(f"Datos cargados: {len(result['dataframe'])} filas, {len(result['dataframe'].columns)} columnas")
                    
                else:
                    state.errors.append("Formato de archivo no soportado")
                    
            elif request.sample_data:
                # Crear datos de muestra
                result = await asyncio.to_thread(
                    self.csv_processor.create_sample_data,
                    request.sample_data
                )
                
                state.dataset = result['dataset_info']
                state.dataframe = result['dataframe']
                logger.info(f"Datos de muestra creados: {request.sample_data}")
                
            else:
                state.errors.append("No se especificó fuente de datos")
                
        except Exception as e:
            error_msg = f"Error en carga de datos: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def validate_data_node(self, state: DataAnalysisState) -> DataAnalysisState:
        """Nodo para validar datos."""
        logger.info("Validando datos")
        
        if not state.dataframe or state.dataframe.empty:
            state.errors.append("No hay datos para validar")
            return state
        
        try:
            # Validación básica
            validation_result = await asyncio.to_thread(
                self.csv_processor.validate_data,
                state.dataframe
            )
            
            state.dataset.validation_errors = validation_result.get('errors', [])
            state.dataset.warnings = validation_result.get('warnings', [])
            
            # Actualizar información del dataset
            state.dataset.columns = list(state.dataframe.columns)
            state.dataset.data_types = {col: str(dtype) for col, dtype in state.dataframe.dtypes.items()}
            state.dataset.missing_values = state.dataframe.isnull().sum().to_dict()
            
            logger.info("Validación completada")
            
        except Exception as e:
            error_msg = f"Error en validación: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def analyze_data_node(self, state: DataAnalysisState) -> DataAnalysisState:
        """Nodo para análisis estadístico."""
        logger.info("Realizando análisis estadístico")
        
        if state.dataframe is None or state.dataframe.empty:
            state.errors.append("No hay datos para analizar")
            return state
        
        try:
            analyses = {}
            
            # Análisis descriptivo
            if not state.request.analysis_types or 'descriptive' in state.request.analysis_types:
                desc_stats = await asyncio.to_thread(
                    self.stats_processor.descriptive_stats,
                    state.dataframe,
                    state.request.target_columns
                )
                analyses['descriptive'] = desc_stats
            
            # Análisis de correlación
            if not state.request.analysis_types or 'correlation' in state.request.analysis_types:
                corr_analysis = await asyncio.to_thread(
                    self.stats_processor.correlation_analysis,
                    state.dataframe
                )
                analyses['correlation'] = corr_analysis
            
            # Análisis de distribución para columnas específicas
            if state.request.target_columns:
                distribution_analyses = {}
                for col in state.request.target_columns:
                    if col in state.dataframe.columns:
                        dist_analysis = await asyncio.to_thread(
                            self.stats_processor.distribution_analysis,
                            state.dataframe,
                            col
                        )
                        distribution_analyses[col] = dist_analysis
                
                if distribution_analyses:
                    analyses['distribution'] = distribution_analyses
            
            state.analyses = analyses
            logger.info(f"Análisis completado: {list(analyses.keys())}")
            
        except Exception as e:
            error_msg = f"Error en análisis: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def create_visualizations_node(self, state: DataAnalysisState) -> DataAnalysisState:
        """Nodo para crear visualizaciones."""
        logger.info("Creando visualizaciones")
        
        if state.dataframe is None or state.dataframe.empty:
            state.errors.append("No hay datos para visualizar")
            return state
        
        try:
            visualizations = {}
            
            # Visualizaciones automáticas basadas en tipos de datos
            numeric_cols = state.dataframe.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = state.dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Histogramas para columnas numéricas
            if numeric_cols and (not state.request.visualization_types or 'histogram' in state.request.visualization_types):
                for col in numeric_cols[:3]:  # Máximo 3 histogramas
                    hist_result = await asyncio.to_thread(
                        self.viz_processor.create_histogram,
                        state.dataframe,
                        col
                    )
                    if 'error' not in hist_result:
                        visualizations[f'histogram_{col}'] = hist_result
            
            # Gráficos de barras para columnas categóricas
            if categorical_cols and (not state.request.visualization_types or 'bar_chart' in state.request.visualization_types):
                for col in categorical_cols[:2]:  # Máximo 2 gráficos de barras
                    bar_result = await asyncio.to_thread(
                        self.viz_processor.create_bar_chart,
                        state.dataframe,
                        col
                    )
                    if 'error' not in bar_result:
                        visualizations[f'bar_chart_{col}'] = bar_result
            
            # Mapa de calor de correlaciones
            if len(numeric_cols) >= 2 and (not state.request.visualization_types or 'heatmap' in state.request.visualization_types):
                heatmap_result = await asyncio.to_thread(
                    self.viz_processor.create_correlation_heatmap,
                    state.dataframe
                )
                if 'error' not in heatmap_result:
                    visualizations['correlation_heatmap'] = heatmap_result
            
            # Scatter plot si hay columnas target específicas
            if (state.request.target_columns and len(state.request.target_columns) >= 2 and
                (not state.request.visualization_types or 'scatter' in state.request.visualization_types)):
                
                col1, col2 = state.request.target_columns[:2]
                if (col1 in state.dataframe.columns and col2 in state.dataframe.columns):
                    scatter_result = await asyncio.to_thread(
                        self.viz_processor.create_scatter_plot,
                        state.dataframe,
                        col1,
                        col2
                    )
                    if 'error' not in scatter_result:
                        visualizations[f'scatter_{col1}_{col2}'] = scatter_result
            
            state.visualizations = visualizations
            logger.info(f"Visualizaciones creadas: {list(visualizations.keys())}")
            
        except Exception as e:
            error_msg = f"Error creando visualizaciones: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def generate_report_node(self, state: DataAnalysisState) -> DataAnalysisState:
        """Nodo para generar reporte final."""
        logger.info("Generando reporte final")
        
        try:
            # Crear resultado final
            result = AnalysisResult(
                summary=self._generate_summary(state),
                dataset_info=state.dataset,
                analyses=state.analyses,
                visualizations=state.visualizations,
                recommendations=self._generate_recommendations(state),
                errors=state.errors
            )
            
            state.result = result
            logger.info("Reporte generado exitosamente")
            
        except Exception as e:
            error_msg = f"Error generando reporte: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def _generate_summary(self, state: DataAnalysisState) -> str:
        """Genera resumen del análisis."""
        summary_parts = []
        
        if state.dataset:
            summary_parts.append(f"Dataset: {state.dataset.rows} filas, {state.dataset.columns_count} columnas")
        
        if state.analyses:
            analysis_types = list(state.analyses.keys())
            summary_parts.append(f"Análisis realizados: {', '.join(analysis_types)}")
        
        if state.visualizations:
            viz_count = len(state.visualizations)
            summary_parts.append(f"Visualizaciones creadas: {viz_count}")
        
        if state.errors:
            summary_parts.append(f"Errores encontrados: {len(state.errors)}")
        
        return ". ".join(summary_parts) if summary_parts else "Análisis completado"
    
    def _generate_recommendations(self, state: DataAnalysisState) -> List[str]:
        """Genera recomendaciones basadas en el análisis."""
        recommendations = []
        
        # Recomendaciones basadas en errores de validación
        if state.dataset and state.dataset.validation_errors:
            recommendations.append("Revisar y limpiar los errores de validación en los datos")
        
        # Recomendaciones basadas en valores faltantes
        if state.dataset and state.dataset.missing_values:
            high_missing = [col for col, count in state.dataset.missing_values.items() 
                           if count > state.dataset.rows * 0.1]  # Más del 10% faltantes
            if high_missing:
                recommendations.append(f"Considerar tratamiento de valores faltantes en: {', '.join(high_missing)}")
        
        # Recomendaciones basadas en correlaciones
        if state.analyses and 'correlation' in state.analyses:
            corr_data = state.analyses['correlation']
            if 'top_correlations' in corr_data and corr_data['top_correlations']:
                strong_corr = [c for c in corr_data['top_correlations'] if abs(c['correlation']) > 0.7]
                if strong_corr:
                    recommendations.append("Se encontraron correlaciones fuertes que podrían indicar multicolinealidad")
        
        # Recomendaciones basadas en distribuciones
        if state.analyses and 'distribution' in state.analyses:
            for col, dist_data in state.analyses['distribution'].items():
                if dist_data.get('outliers', {}).get('percentage', 0) > 5:
                    recommendations.append(f"Considerar tratamiento de outliers en {col}")
        
        return recommendations if recommendations else ["Los datos parecen estar en buen estado para análisis"]
    
    async def run_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Ejecuta el workflow completo de análisis.
        
        Args:
            request: Solicitud de análisis
            
        Returns:
            Resultado del análisis
        """
        # Crear estado inicial
        initial_state = DataAnalysisState(request=request)
        
        # Ejecutar workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        # Retornar resultado
        if final_state.result:
            return final_state.result
        else:
            # Crear resultado de error si no hay resultado
            return AnalysisResult(
                summary="Error en el análisis",
                errors=final_state.errors or ["Error desconocido en el workflow"]
            )
