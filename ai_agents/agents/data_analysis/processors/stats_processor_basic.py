"""
Procesador básico para análisis estadísticos de datos (sin scipy).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings

logger = logging.getLogger(__name__)


class StatsProcessor:
    """Procesador especializado para análisis estadísticos básicos."""
    
    def __init__(self):
        # Suprimir warnings para análisis limpio
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    def descriptive_stats(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calcula estadísticas descriptivas básicas.
        
        Args:
            df: DataFrame a analizar
            columns: Columnas específicas a analizar (None = todas las numéricas)
            
        Returns:
            Diccionario con estadísticas descriptivas
        """
        if columns is None:
            # Solo columnas numéricas
            numeric_df = df.select_dtypes(include=[np.number])
        else:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'error': 'No hay columnas numéricas para analizar'}
        
        stats_dict = {}
        
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            if len(col_data) == 0:
                continue
                
            stats_dict[col] = {
                'count': len(col_data),
                'mean': col_data.mean(),
                'median': col_data.median(),
                'mode': col_data.mode().iloc[0] if not col_data.mode().empty else None,
                'std': col_data.std(),
                'var': col_data.var(),
                'min': col_data.min(),
                'max': col_data.max(),
                'range': col_data.max() - col_data.min(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75),
                'iqr': col_data.quantile(0.75) - col_data.quantile(0.25),
                'skewness': col_data.skew(),  # pandas tiene skew básico
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
            }
        
        return {
            'descriptive_statistics': stats_dict,
            'summary': self._generate_descriptive_summary(stats_dict)
        }
    
    def correlation_analysis(self, df: pd.DataFrame, method: str = 'pearson') -> Dict[str, Any]:
        """
        Analiza correlaciones entre variables numéricas.
        
        Args:
            df: DataFrame a analizar
            method: Método de correlación ('pearson', 'spearman', 'kendall')
            
        Returns:
            Diccionario con análisis de correlación
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'error': 'Se necesitan al menos 2 columnas numéricas para correlación'}
        
        # Calcular matriz de correlación
        corr_matrix = numeric_df.corr(method=method)
        
        # Encontrar correlaciones más fuertes
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                if not np.isnan(corr_value):
                    correlations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': corr_value,
                        'strength': self._interpret_correlation(abs(corr_value))
                    })
        
        # Ordenar por valor absoluto de correlación
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'top_correlations': correlations[:10],
            'method': method,
            'summary': self._generate_correlation_summary(correlations)
        }
    
    def distribution_analysis(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Analiza la distribución de una variable.
        
        Args:
            df: DataFrame
            column: Nombre de la columna a analizar
            
        Returns:
            Diccionario con análisis de distribución
        """
        if column not in df.columns:
            return {'error': f'Columna {column} no encontrada'}
        
        col_data = df[column].dropna()
        
        if not pd.api.types.is_numeric_dtype(col_data):
            return self._categorical_distribution(col_data, column)
        
        return self._numeric_distribution(col_data, column)
    
    def _numeric_distribution(self, data: pd.Series, column: str) -> Dict[str, Any]:
        """Analiza distribución de variable numérica."""
        
        # Detectar outliers usando IQR
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            'column': column,
            'type': 'numeric',
            'distribution_stats': {
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'skewness': data.skew()
            },
            'outliers': {
                'count': len(outliers),
                'percentage': (len(outliers) / len(data)) * 100,
                'values': outliers.tolist()[:20]  # Máximo 20 outliers
            },
            'distribution_type': self._detect_distribution_type_basic(data)
        }
    
    def _categorical_distribution(self, data: pd.Series, column: str) -> Dict[str, Any]:
        """Analiza distribución de variable categórica."""
        value_counts = data.value_counts()
        
        return {
            'column': column,
            'type': 'categorical',
            'unique_values': len(value_counts),
            'value_counts': value_counts.to_dict(),
            'percentages': (value_counts / len(data) * 100).to_dict(),
            'most_frequent': value_counts.index[0],
            'least_frequent': value_counts.index[-1]
        }
    
    def comparative_analysis(self, df: pd.DataFrame, group_column: str, value_column: str) -> Dict[str, Any]:
        """
        Compara estadísticas entre grupos.
        
        Args:
            df: DataFrame
            group_column: Columna de agrupación
            value_column: Columna de valores a comparar
            
        Returns:
            Diccionario con análisis comparativo
        """
        if group_column not in df.columns or value_column not in df.columns:
            return {'error': 'Una o ambas columnas no existen'}
        
        # Agrupar y calcular estadísticas
        groups = df.groupby(group_column)[value_column]
        
        group_stats = {}
        
        for name, group in groups:
            group_stats[name] = {
                'count': len(group.dropna()),
                'mean': group.mean(),
                'median': group.median(),
                'std': group.std(),
                'min': group.min(),
                'max': group.max()
            }
        
        return {
            'group_statistics': group_stats,
            'summary': self._generate_comparative_summary(group_stats)
        }
    
    def _interpret_correlation(self, corr_value: float) -> str:
        """Interpreta la fuerza de una correlación."""
        abs_corr = abs(corr_value)
        if abs_corr >= 0.8:
            return 'muy fuerte'
        elif abs_corr >= 0.6:
            return 'fuerte'
        elif abs_corr >= 0.4:
            return 'moderada'
        elif abs_corr >= 0.2:
            return 'débil'
        else:
            return 'muy débil'
    
    def _detect_distribution_type_basic(self, data: pd.Series) -> str:
        """Detecta el tipo de distribución de manera básica."""
        skew = data.skew()
        
        if abs(skew) < 0.5:
            return 'aproximadamente simétrica'
        elif skew > 1:
            return 'sesgada a la derecha'
        elif skew < -1:
            return 'sesgada a la izquierda'
        else:
            return 'ligeramente sesgada'
    
    def _generate_descriptive_summary(self, stats_dict: Dict) -> str:
        """Genera resumen en texto de estadísticas descriptivas."""
        if not stats_dict:
            return "No hay estadísticas disponibles."
        
        summaries = []
        for col, stats in stats_dict.items():
            summary = f"{col}: media={stats['mean']:.2f}, mediana={stats['median']:.2f}, desv.std={stats['std']:.2f}"
            if stats['missing_percentage'] > 0:
                summary += f", valores faltantes={stats['missing_percentage']:.1f}%"
            summaries.append(summary)
        
        return "; ".join(summaries)
    
    def _generate_correlation_summary(self, correlations: List[Dict]) -> str:
        """Genera resumen de correlaciones."""
        if not correlations:
            return "No se encontraron correlaciones significativas."
        
        strong_corr = [c for c in correlations[:5] if abs(c['correlation']) > 0.6]
        if strong_corr:
            top_corr = strong_corr[0]
            return f"Correlación más fuerte: {top_corr['variable1']} y {top_corr['variable2']} (r={top_corr['correlation']:.3f}, {top_corr['strength']})"
        else:
            return "No se encontraron correlaciones fuertes entre las variables."
    
    def _generate_comparative_summary(self, group_stats: Dict) -> str:
        """Genera resumen de análisis comparativo."""
        if not group_stats:
            return "No hay grupos para comparar."
        
        means = {group: stats['mean'] for group, stats in group_stats.items()}
        highest_group = max(means, key=means.get)
        lowest_group = min(means, key=means.get)
        
        summary = f"Grupo con mayor promedio: {highest_group} ({means[highest_group]:.2f}), "
        summary += f"menor promedio: {lowest_group} ({means[lowest_group]:.2f})"
        
        return summary
