"""
Herramientas especializadas para análisis de datos.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path

from ai_agents.agents.data_analysis.processors import CSVProcessor, StatsProcessor, VisualizationProcessor

logger = logging.getLogger(__name__)


class DataAnalysisTools:
    """Conjunto de herramientas para análisis de datos."""
    
    def __init__(self):
        self.csv_processor = CSVProcessor()
        self.stats_processor = StatsProcessor()
        self.viz_processor = VisualizationProcessor()
    
    def quick_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera un perfil rápido del dataset.
        
        Args:
            df: DataFrame a perfilar
            
        Returns:
            Diccionario con información del perfil
        """
        if df.empty:
            return {'error': 'DataFrame está vacío'}
        
        try:
            profile = {
                'basic_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'size_mb': df.memory_usage(deep=True).sum() / 1024**2,
                    'columns_list': list(df.columns)
                },
                'data_types': {
                    'numeric': len(df.select_dtypes(include=[np.number]).columns),
                    'categorical': len(df.select_dtypes(include=['object', 'category']).columns),
                    'datetime': len(df.select_dtypes(include=['datetime64']).columns),
                    'boolean': len(df.select_dtypes(include=['bool']).columns)
                },
                'missing_data': {
                    'total_missing': df.isnull().sum().sum(),
                    'percentage_missing': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                    'columns_with_missing': df.columns[df.isnull().any()].tolist()
                },
                'duplicates': {
                    'duplicate_rows': df.duplicated().sum(),
                    'percentage_duplicates': (df.duplicated().sum() / len(df)) * 100
                }
            }
            
            # Análisis por columna
            column_analysis = {}
            for col in df.columns:
                col_data = df[col]
                col_info = {
                    'type': str(col_data.dtype),
                    'unique_values': col_data.nunique(),
                    'missing_count': col_data.isnull().sum(),
                    'missing_percentage': (col_data.isnull().sum() / len(df)) * 100
                }
                
                if pd.api.types.is_numeric_dtype(col_data):
                    col_info.update({
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'mean': col_data.mean(),
                        'median': col_data.median(),
                        'std': col_data.std()
                    })
                elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
                    col_info.update({
                        'most_frequent': col_data.mode().iloc[0] if not col_data.mode().empty else None,
                        'top_values': col_data.value_counts().head().to_dict()
                    })
                
                column_analysis[col] = col_info
            
            profile['column_analysis'] = column_analysis
            
            return profile
            
        except Exception as e:
            logger.error(f"Error en quick_profile: {str(e)}")
            return {'error': f'Error generando perfil: {str(e)}'}
    
    def detect_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr') -> Dict[str, Any]:
        """
        Detecta outliers en una columna específica.
        
        Args:
            df: DataFrame
            column: Nombre de la columna
            method: Método de detección ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            Diccionario con información de outliers
        """
        if column not in df.columns:
            return {'error': f'Columna {column} no encontrada'}
        
        col_data = df[column].dropna()
        
        if not pd.api.types.is_numeric_dtype(col_data):
            return {'error': f'Columna {column} no es numérica'}
        
        try:
            outliers_info = {'method': method, 'column': column}
            
            if method == 'iqr':
                q1, q3 = col_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
                outliers = col_data[outliers_mask]
                
                outliers_info.update({
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr
                })
            
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(col_data))
                threshold = 3
                outliers_mask = z_scores > threshold
                outliers = col_data[outliers_mask]
                
                outliers_info.update({
                    'threshold': threshold,
                    'max_zscore': z_scores.max(),
                    'mean_zscore': z_scores.mean()
                })
            
            elif method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers_mask = iso_forest.fit_predict(col_data.values.reshape(-1, 1)) == -1
                outliers = col_data[outliers_mask]
                
                outliers_info.update({
                    'contamination': 0.1,
                    'algorithm': 'Isolation Forest'
                })
            
            else:
                return {'error': f'Método {method} no reconocido'}
            
            outliers_info.update({
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers) / len(col_data)) * 100,
                'outliers_values': outliers.tolist()[:20],  # Máximo 20 valores
                'total_values': len(col_data)
            })
            
            return outliers_info
            
        except Exception as e:
            logger.error(f"Error detectando outliers: {str(e)}")
            return {'error': f'Error detectando outliers: {str(e)}'}
    
    def suggest_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Sugiere mejores tipos de datos para las columnas.
        
        Args:
            df: DataFrame a analizar
            
        Returns:
            Diccionario con sugerencias
        """
        suggestions = {}
        
        try:
            for col in df.columns:
                col_data = df[col]
                current_type = str(col_data.dtype)
                suggestions[col] = {'current_type': current_type, 'suggested_type': current_type, 'reason': ''}
                
                # Si es object, analizar si puede ser numérico
                if col_data.dtype == 'object':
                    # Intentar convertir a numérico
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    if not numeric_data.isnull().all():
                        null_percentage = numeric_data.isnull().sum() / len(col_data) * 100
                        if null_percentage < 10:  # Menos del 10% de errores de conversión
                            if (numeric_data % 1 == 0).all():  # Todos son enteros
                                suggestions[col].update({
                                    'suggested_type': 'int64',
                                    'reason': 'Todos los valores pueden convertirse a enteros'
                                })
                            else:
                                suggestions[col].update({
                                    'suggested_type': 'float64',
                                    'reason': 'Valores pueden convertirse a números decimales'
                                })
                    
                    # Verificar si puede ser fecha
                    try:
                        pd.to_datetime(col_data.dropna().head(100), errors='raise')
                        suggestions[col].update({
                            'suggested_type': 'datetime64[ns]',
                            'reason': 'Valores parecen ser fechas'
                        })
                    except:
                        pass
                    
                    # Verificar si puede ser categórica
                    unique_ratio = col_data.nunique() / len(col_data)
                    if unique_ratio < 0.5 and col_data.nunique() < 50:  # Menos del 50% únicos y menos de 50 categorías
                        suggestions[col].update({
                            'suggested_type': 'category',
                            'reason': f'Solo {col_data.nunique()} valores únicos, eficiente como categórica'
                        })
                
                # Si es numérico, verificar si puede ser entero
                elif pd.api.types.is_float_dtype(col_data):
                    if (col_data.dropna() % 1 == 0).all():
                        suggestions[col].update({
                            'suggested_type': 'int64',
                            'reason': 'Todos los valores son enteros'
                        })
                
                # Si es int64 con pocos valores únicos, podría ser categórica
                elif pd.api.types.is_integer_dtype(col_data):
                    unique_ratio = col_data.nunique() / len(col_data)
                    if unique_ratio < 0.1 and col_data.nunique() < 20:
                        suggestions[col].update({
                            'suggested_type': 'category',
                            'reason': f'Solo {col_data.nunique()} valores únicos, eficiente como categórica'
                        })
            
            return {
                'suggestions': suggestions,
                'summary': self._summarize_type_suggestions(suggestions)
            }
            
        except Exception as e:
            logger.error(f"Error sugiriendo tipos: {str(e)}")
            return {'error': f'Error sugiriendo tipos: {str(e)}'}
    
    def compare_columns(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Compara dos columnas del DataFrame.
        
        Args:
            df: DataFrame
            col1: Primera columna
            col2: Segunda columna
            
        Returns:
            Diccionario con comparación
        """
        if col1 not in df.columns or col2 not in df.columns:
            return {'error': 'Una o ambas columnas no existen'}
        
        try:
            comparison = {
                'columns': [col1, col2],
                'basic_comparison': {},
                'statistical_comparison': {},
                'relationship_analysis': {}
            }
            
            data1, data2 = df[col1], df[col2]
            
            # Comparación básica
            comparison['basic_comparison'] = {
                'types': [str(data1.dtype), str(data2.dtype)],
                'unique_values': [data1.nunique(), data2.nunique()],
                'missing_values': [data1.isnull().sum(), data2.isnull().sum()],
                'missing_percentage': [
                    (data1.isnull().sum() / len(df)) * 100,
                    (data2.isnull().sum() / len(df)) * 100
                ]
            }
            
            # Si ambas son numéricas
            if pd.api.types.is_numeric_dtype(data1) and pd.api.types.is_numeric_dtype(data2):
                comparison['statistical_comparison'] = {
                    'means': [data1.mean(), data2.mean()],
                    'medians': [data1.median(), data2.median()],
                    'stds': [data1.std(), data2.std()],
                    'mins': [data1.min(), data2.min()],
                    'maxs': [data1.max(), data2.max()]
                }
                
                # Análisis de relación
                correlation = data1.corr(data2)
                comparison['relationship_analysis'] = {
                    'correlation': correlation,
                    'correlation_strength': self.stats_processor._interpret_correlation(abs(correlation))
                }
                
                # Test estadístico
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(data1.dropna(), data2.dropna())
                comparison['statistical_test'] = {
                    'test': 't-test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant_difference': p_value < 0.05
                }
            
            # Si ambas son categóricas
            elif (data1.dtype == 'object' or pd.api.types.is_categorical_dtype(data1)) and \
                 (data2.dtype == 'object' or pd.api.types.is_categorical_dtype(data2)):
                
                # Valores en común
                common_values = set(data1.dropna().unique()) & set(data2.dropna().unique())
                comparison['relationship_analysis'] = {
                    'common_values': list(common_values),
                    'common_values_count': len(common_values),
                    'jaccard_similarity': len(common_values) / len(set(data1.dropna().unique()) | set(data2.dropna().unique()))
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparando columnas: {str(e)}")
            return {'error': f'Error comparando columnas: {str(e)}'}
    
    def export_analysis(self, analysis_result: Dict[str, Any], output_path: str, format: str = 'json') -> Dict[str, Any]:
        """
        Exporta resultados de análisis a archivo.
        
        Args:
            analysis_result: Resultado del análisis
            output_path: Ruta de salida
            format: Formato de exportación ('json', 'csv', 'html')
            
        Returns:
            Diccionario con información de exportación
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'json':
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result, f, indent=2, ensure_ascii=False, default=str)
            
            elif format == 'csv':
                # Exportar solo datos tabulares
                if 'analyses' in analysis_result and 'descriptive' in analysis_result['analyses']:
                    desc_stats = analysis_result['analyses']['descriptive']
                    if 'descriptive_statistics' in desc_stats:
                        df_stats = pd.DataFrame(desc_stats['descriptive_statistics']).T
                        df_stats.to_csv(output_path)
                else:
                    return {'error': 'No hay datos tabulares para exportar en CSV'}
            
            elif format == 'html':
                html_content = self._generate_html_report(analysis_result)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            else:
                return {'error': f'Formato {format} no soportado'}
            
            return {
                'success': True,
                'output_path': str(output_path),
                'format': format,
                'file_size': output_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Error exportando análisis: {str(e)}")
            return {'error': f'Error exportando análisis: {str(e)}'}
    
    def _summarize_type_suggestions(self, suggestions: Dict) -> str:
        """Genera resumen de sugerencias de tipos."""
        changes = sum(1 for s in suggestions.values() if s['suggested_type'] != s['current_type'])
        total = len(suggestions)
        return f"{changes} de {total} columnas podrían beneficiarse de cambio de tipo"
    
    def _generate_html_report(self, analysis_result: Dict[str, Any]) -> str:
        """Genera reporte HTML del análisis."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Análisis de Datos</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; }
                .header { color: #2c3e50; border-bottom: 2px solid #3498db; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .error { color: #e74c3c; }
                .success { color: #27ae60; }
            </style>
        </head>
        <body>
            <h1 class="header">Reporte de Análisis de Datos</h1>
        """
        
        # Agregar resumen
        if 'summary' in analysis_result:
            html += f"<div class='section'><h2>Resumen</h2><p>{analysis_result['summary']}</p></div>"
        
        # Agregar información del dataset
        if 'dataset_info' in analysis_result:
            dataset = analysis_result['dataset_info']
            html += f"""
            <div class='section'>
                <h2>Información del Dataset</h2>
                <p>Filas: {getattr(dataset, 'rows', 'N/A')}</p>
                <p>Columnas: {getattr(dataset, 'columns_count', 'N/A')}</p>
            </div>
            """
        
        # Agregar errores si existen
        if 'errors' in analysis_result and analysis_result['errors']:
            html += "<div class='section'><h2>Errores</h2><ul>"
            for error in analysis_result['errors']:
                html += f"<li class='error'>{error}</li>"
            html += "</ul></div>"
        
        html += "</body></html>"
        return html
