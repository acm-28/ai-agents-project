"""
Procesador para análisis estadísticos de datos.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None

logger = logging.getLogger(__name__)


class StatsProcessor:
    """Procesador especializado para análisis estadísticos."""
    
    def __init__(self):
        # Suprimir warnings de scipy para análisis limpio
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        if not HAS_SCIPY:
            logger.warning("scipy no está disponible. Algunas funciones estadísticas estarán limitadas.")
    
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
                'skewness': stats.skew(col_data) if HAS_SCIPY else None,
                'kurtosis': stats.kurtosis(col_data) if HAS_SCIPY else None,
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
        
        # Tests de normalidad
        shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        
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
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            },
            'normality_tests': {
                'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p, 'is_normal': shapiro_p > 0.05},
                'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p, 'is_normal': ks_p > 0.05}
            },
            'outliers': {
                'count': len(outliers),
                'percentage': (len(outliers) / len(data)) * 100,
                'values': outliers.tolist()[:20]  # Máximo 20 outliers
            },
            'distribution_type': self._detect_distribution_type(data)
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
            'least_frequent': value_counts.index[-1],
            'entropy': stats.entropy(value_counts)
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
        group_data = []
        
        for name, group in groups:
            group_data.append(group.dropna())
            group_stats[name] = {
                'count': len(group.dropna()),
                'mean': group.mean(),
                'median': group.median(),
                'std': group.std(),
                'min': group.min(),
                'max': group.max()
            }
        
        # Test estadístico (ANOVA si más de 2 grupos, t-test si 2 grupos)
        statistical_test = {}
        if len(group_data) == 2:
            stat, p_value = stats.ttest_ind(group_data[0], group_data[1])
            statistical_test = {
                'test': 't-test',
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        elif len(group_data) > 2:
            stat, p_value = stats.f_oneway(*group_data)
            statistical_test = {
                'test': 'ANOVA',
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return {
            'group_statistics': group_stats,
            'statistical_test': statistical_test,
            'summary': self._generate_comparative_summary(group_stats, statistical_test)
        }
    
    def trend_analysis(self, df: pd.DataFrame, date_column: str, value_column: str) -> Dict[str, Any]:
        """
        Analiza tendencias temporales.
        
        Args:
            df: DataFrame
            date_column: Columna de fechas
            value_column: Columna de valores
            
        Returns:
            Diccionario con análisis de tendencias
        """
        if date_column not in df.columns or value_column not in df.columns:
            return {'error': 'Una o ambas columnas no existen'}
        
        # Preparar datos
        df_trend = df[[date_column, value_column]].copy()
        df_trend[date_column] = pd.to_datetime(df_trend[date_column])
        df_trend = df_trend.sort_values(date_column).dropna()
        
        if len(df_trend) < 2:
            return {'error': 'Datos insuficientes para análisis de tendencia'}
        
        # Calcular tendencia lineal
        x = np.arange(len(df_trend))
        y = df_trend[value_column].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Detectar estacionalidad básica
        seasonality = self._detect_seasonality(df_trend, date_column, value_column)
        
        return {
            'linear_trend': {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            },
            'seasonality': seasonality,
            'period_analysis': self._analyze_periods(df_trend, date_column, value_column),
            'summary': self._generate_trend_summary(slope, r_value**2, seasonality)
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
    
    def _detect_distribution_type(self, data: pd.Series) -> str:
        """Detecta el tipo de distribución más probable."""
        # Test simple basado en skewness y kurtosis
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        
        if abs(skew) < 0.5 and abs(kurt) < 0.5:
            return 'aproximadamente normal'
        elif skew > 1:
            return 'sesgada a la derecha'
        elif skew < -1:
            return 'sesgada a la izquierda'
        elif kurt > 1:
            return 'leptocúrtica (picos altos)'
        elif kurt < -1:
            return 'platicúrtica (picos bajos)'
        else:
            return 'distribución irregular'
    
    def _detect_seasonality(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Detecta patrones estacionales básicos."""
        df['month'] = df[date_col].dt.month
        df['weekday'] = df[date_col].dt.dayofweek
        
        monthly_var = df.groupby('month')[value_col].var().mean()
        weekly_var = df.groupby('weekday')[value_col].var().mean()
        total_var = df[value_col].var()
        
        return {
            'monthly_seasonality': monthly_var / total_var if total_var > 0 else 0,
            'weekly_seasonality': weekly_var / total_var if total_var > 0 else 0,
            'has_seasonality': (monthly_var / total_var > 0.1) or (weekly_var / total_var > 0.1)
        }
    
    def _analyze_periods(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Analiza diferentes períodos temporales."""
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        
        yearly_stats = df.groupby('year')[value_col].agg(['mean', 'sum', 'count']).to_dict('index')
        monthly_stats = df.groupby('month')[value_col].agg(['mean', 'sum', 'count']).to_dict('index')
        
        return {
            'yearly_statistics': yearly_stats,
            'monthly_statistics': monthly_stats
        }
    
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
    
    def _generate_comparative_summary(self, group_stats: Dict, statistical_test: Dict) -> str:
        """Genera resumen de análisis comparativo."""
        if not group_stats:
            return "No hay grupos para comparar."
        
        means = {group: stats['mean'] for group, stats in group_stats.items()}
        highest_group = max(means, key=means.get)
        lowest_group = min(means, key=means.get)
        
        summary = f"Grupo con mayor promedio: {highest_group} ({means[highest_group]:.2f}), "
        summary += f"menor promedio: {lowest_group} ({means[lowest_group]:.2f})"
        
        if statistical_test:
            if statistical_test['significant']:
                summary += f". Diferencia estadísticamente significativa (p={statistical_test['p_value']:.4f})"
            else:
                summary += f". Sin diferencia estadísticamente significativa (p={statistical_test['p_value']:.4f})"
        
        return summary
    
    def _generate_trend_summary(self, slope: float, r_squared: float, seasonality: Dict) -> str:
        """Genera resumen de análisis de tendencias."""
        trend_dir = "creciente" if slope > 0 else "decreciente" if slope < 0 else "estable"
        strength = "fuerte" if r_squared > 0.7 else "moderada" if r_squared > 0.3 else "débil"
        
        summary = f"Tendencia {trend_dir} con ajuste {strength} (R²={r_squared:.3f})"
        
        if seasonality.get('has_seasonality'):
            summary += ". Se detectaron patrones estacionales"
        
        return summary
