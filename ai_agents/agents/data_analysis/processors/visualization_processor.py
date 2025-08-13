"""
Procesador para visualización de datos.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import base64
import io

# Matplotlib con backend sin GUI
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None

logger = logging.getLogger(__name__)


class VisualizationProcessor:
    """Procesador especializado para crear visualizaciones de datos."""
    
    def __init__(self):
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib/seaborn no están disponibles. Las visualizaciones estarán deshabilitadas.")
            return
        
        # Configurar estilo de seaborn
        sns.set_style("whitegrid")
        plt.style.use('default')
        
        # Configuración por defecto
        self.default_figsize = (10, 6)
        self.default_dpi = 100
        
    def create_histogram(self, df: pd.DataFrame, column: str, bins: int = 30, **kwargs) -> Dict[str, Any]:
        """
        Crea un histograma de una variable numérica.
        
        Args:
            df: DataFrame
            column: Nombre de la columna
            bins: Número de bins
            **kwargs: Argumentos adicionales para personalización
            
        Returns:
            Diccionario con la imagen y metadatos
        """
        if not HAS_MATPLOTLIB:
            return {'error': 'matplotlib no está disponible para crear visualizaciones'}
        
        if column not in df.columns:
            return {'error': f'Columna {column} no encontrada'}
        
        data = df[column].dropna()
        if not pd.api.types.is_numeric_dtype(data):
            return {'error': f'Columna {column} no es numérica'}
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', self.default_figsize))
        
        # Crear histograma
        n, bins_edges, patches = ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        
        # Personalización
        ax.set_title(kwargs.get('title', f'Distribución de {column}'))
        ax.set_xlabel(column)
        ax.set_ylabel('Frecuencia')
        
        # Añadir estadísticas al gráfico
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Media: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Mediana: {median_val:.2f}')
        ax.legend()
        
        plt.tight_layout()
        
        # Convertir a base64
        image_b64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return {
            'type': 'histogram',
            'image': image_b64,
            'metadata': {
                'column': column,
                'bins': bins,
                'count': len(data),
                'mean': mean_val,
                'median': median_val,
                'std': data.std()
            }
        }
    
    def create_scatter_plot(self, df: pd.DataFrame, x_column: str, y_column: str, 
                           hue_column: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Crea un gráfico de dispersión.
        
        Args:
            df: DataFrame
            x_column: Columna para eje X
            y_column: Columna para eje Y
            hue_column: Columna para color (opcional)
            **kwargs: Argumentos adicionales
            
        Returns:
            Diccionario con la imagen y metadatos
        """
        if x_column not in df.columns or y_column not in df.columns:
            return {'error': 'Una o ambas columnas no existen'}
        
        # Filtrar datos válidos
        if hue_column and hue_column in df.columns:
            plot_data = df[[x_column, y_column, hue_column]].dropna()
        else:
            plot_data = df[[x_column, y_column]].dropna()
            hue_column = None
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', self.default_figsize))
        
        # Crear scatter plot
        if hue_column:
            unique_hues = plot_data[hue_column].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_hues)))
            
            for i, hue_val in enumerate(unique_hues):
                subset = plot_data[plot_data[hue_column] == hue_val]
                ax.scatter(subset[x_column], subset[y_column], 
                          label=hue_val, alpha=0.6, color=colors[i])
            ax.legend()
        else:
            ax.scatter(plot_data[x_column], plot_data[y_column], alpha=0.6)
        
        # Personalización
        ax.set_title(kwargs.get('title', f'{y_column} vs {x_column}'))
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        
        # Línea de tendencia si es numérico
        if (pd.api.types.is_numeric_dtype(plot_data[x_column]) and 
            pd.api.types.is_numeric_dtype(plot_data[y_column])):
            z = np.polyfit(plot_data[x_column], plot_data[y_column], 1)
            p = np.poly1d(z)
            ax.plot(plot_data[x_column].sort_values(), 
                   p(plot_data[x_column].sort_values()), 
                   "r--", alpha=0.8, label='Tendencia')
            
            # Calcular correlación
            correlation = plot_data[x_column].corr(plot_data[y_column])
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', 
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        plt.tight_layout()
        
        image_b64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return {
            'type': 'scatter',
            'image': image_b64,
            'metadata': {
                'x_column': x_column,
                'y_column': y_column,
                'hue_column': hue_column,
                'data_points': len(plot_data)
            }
        }
    
    def create_box_plot(self, df: pd.DataFrame, column: str, group_column: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Crea un box plot.
        
        Args:
            df: DataFrame
            column: Columna de valores
            group_column: Columna de agrupación (opcional)
            **kwargs: Argumentos adicionales
            
        Returns:
            Diccionario con la imagen y metadatos
        """
        if column not in df.columns:
            return {'error': f'Columna {column} no encontrada'}
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {'error': f'Columna {column} no es numérica'}
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', self.default_figsize))
        
        if group_column and group_column in df.columns:
            # Box plot agrupado
            plot_data = df[[column, group_column]].dropna()
            groups = [group[column].values for name, group in plot_data.groupby(group_column)]
            labels = [name for name, group in plot_data.groupby(group_column)]
            
            bp = ax.boxplot(groups, labels=labels, patch_artist=True)
            
            # Colores
            colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                
            ax.set_xlabel(group_column)
        else:
            # Box plot simple
            plot_data = df[column].dropna()
            bp = ax.boxplot([plot_data.values], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
        
        ax.set_title(kwargs.get('title', f'Box Plot de {column}'))
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        image_b64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return {
            'type': 'boxplot',
            'image': image_b64,
            'metadata': {
                'column': column,
                'group_column': group_column,
                'groups_count': len(labels) if group_column else 1
            }
        }
    
    def create_correlation_heatmap(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Crea un mapa de calor de correlaciones.
        
        Args:
            df: DataFrame
            **kwargs: Argumentos adicionales
            
        Returns:
            Diccionario con la imagen y metadatos
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'error': 'Se necesitan al menos 2 columnas numéricas'}
        
        # Calcular correlaciones
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
        
        # Crear heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Máscara triangular superior
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title(kwargs.get('title', 'Matriz de Correlación'))
        
        plt.tight_layout()
        
        image_b64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return {
            'type': 'heatmap',
            'image': image_b64,
            'metadata': {
                'variables': list(numeric_df.columns),
                'correlations_count': len(corr_matrix.columns)
            }
        }
    
    def create_time_series(self, df: pd.DataFrame, date_column: str, value_column: str, **kwargs) -> Dict[str, Any]:
        """
        Crea un gráfico de serie temporal.
        
        Args:
            df: DataFrame
            date_column: Columna de fechas
            value_column: Columna de valores
            **kwargs: Argumentos adicionales
            
        Returns:
            Diccionario con la imagen y metadatos
        """
        if date_column not in df.columns or value_column not in df.columns:
            return {'error': 'Una o ambas columnas no existen'}
        
        # Preparar datos
        plot_data = df[[date_column, value_column]].copy()
        plot_data[date_column] = pd.to_datetime(plot_data[date_column])
        plot_data = plot_data.sort_values(date_column).dropna()
        
        if len(plot_data) < 2:
            return {'error': 'Datos insuficientes para gráfico temporal'}
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 6)))
        
        # Crear gráfico de línea
        ax.plot(plot_data[date_column], plot_data[value_column], 
               linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        # Personalización
        ax.set_title(kwargs.get('title', f'Serie Temporal: {value_column}'))
        ax.set_xlabel(date_column)
        ax.set_ylabel(value_column)
        
        # Formatear fechas en eje X
        plt.xticks(rotation=45)
        
        # Agregar línea de tendencia
        if len(plot_data) > 10:
            x_numeric = np.arange(len(plot_data))
            z = np.polyfit(x_numeric, plot_data[value_column], 1)
            p = np.poly1d(z)
            ax.plot(plot_data[date_column], p(x_numeric), 
                   "r--", alpha=0.8, label='Tendencia')
            ax.legend()
        
        plt.tight_layout()
        
        image_b64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return {
            'type': 'timeseries',
            'image': image_b64,
            'metadata': {
                'date_column': date_column,
                'value_column': value_column,
                'data_points': len(plot_data),
                'date_range': {
                    'start': plot_data[date_column].min().strftime('%Y-%m-%d'),
                    'end': plot_data[date_column].max().strftime('%Y-%m-%d')
                }
            }
        }
    
    def create_bar_chart(self, df: pd.DataFrame, column: str, value_column: Optional[str] = None, 
                        top_n: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Crea un gráfico de barras.
        
        Args:
            df: DataFrame
            column: Columna categórica
            value_column: Columna de valores (opcional, usa conteos si no se especifica)
            top_n: Número máximo de categorías a mostrar
            **kwargs: Argumentos adicionales
            
        Returns:
            Diccionario con la imagen y metadatos
        """
        if column not in df.columns:
            return {'error': f'Columna {column} no encontrada'}
        
        if value_column:
            if value_column not in df.columns:
                return {'error': f'Columna {value_column} no encontrada'}
            # Agrupar por categoría y sumar valores
            plot_data = df.groupby(column)[value_column].sum().sort_values(ascending=False).head(top_n)
            ylabel = f'Suma de {value_column}'
        else:
            # Contar frecuencias
            plot_data = df[column].value_counts().head(top_n)
            ylabel = 'Frecuencia'
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', self.default_figsize))
        
        # Crear gráfico de barras
        bars = ax.bar(range(len(plot_data)), plot_data.values, 
                     color=plt.cm.viridis(np.linspace(0, 1, len(plot_data))))
        
        # Personalización
        ax.set_title(kwargs.get('title', f'Distribución de {column}'))
        ax.set_xlabel(column)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
        
        # Añadir valores sobre las barras
        for i, (bar, value) in enumerate(zip(bars, plot_data.values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(plot_data.values)*0.01,
                   f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        image_b64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return {
            'type': 'bar_chart',
            'image': image_b64,
            'metadata': {
                'column': column,
                'value_column': value_column,
                'categories_shown': len(plot_data),
                'total_categories': df[column].nunique()
            }
        }
    
    def create_pie_chart(self, df: pd.DataFrame, column: str, top_n: int = 8, **kwargs) -> Dict[str, Any]:
        """
        Crea un gráfico circular.
        
        Args:
            df: DataFrame
            column: Columna categórica
            top_n: Número máximo de categorías (el resto se agrupa en "Otros")
            **kwargs: Argumentos adicionales
            
        Returns:
            Diccionario con la imagen y metadatos
        """
        if column not in df.columns:
            return {'error': f'Columna {column} no encontrada'}
        
        # Obtener conteos
        value_counts = df[column].value_counts()
        
        if len(value_counts) > top_n:
            # Mantener top_n-1 y agrupar el resto en "Otros"
            top_values = value_counts.head(top_n-1)
            others_sum = value_counts.iloc[top_n-1:].sum()
            plot_data = pd.concat([top_values, pd.Series([others_sum], index=['Otros'])])
        else:
            plot_data = value_counts
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 8)))
        
        # Crear gráfico circular
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
        wedges, texts, autotexts = ax.pie(plot_data.values, labels=plot_data.index,
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        
        # Personalización
        ax.set_title(kwargs.get('title', f'Distribución de {column}'))
        
        # Mejorar legibilidad
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        image_b64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return {
            'type': 'pie_chart',
            'image': image_b64,
            'metadata': {
                'column': column,
                'categories_shown': len(plot_data),
                'total_categories': len(value_counts),
                'has_others_category': len(value_counts) > top_n
            }
        }
    
    def _fig_to_base64(self, fig) -> str:
        """Convierte una figura de matplotlib a string base64."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.default_dpi, bbox_inches='tight')
        buffer.seek(0)
        
        # Convertir a base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        
        return image_base64
    
    def suggest_visualizations(self, df: pd.DataFrame, column: str) -> List[str]:
        """
        Sugiere tipos de visualización apropiados para una columna.
        
        Args:
            df: DataFrame
            column: Nombre de la columna
            
        Returns:
            Lista de tipos de visualización sugeridos
        """
        if column not in df.columns:
            return []
        
        suggestions = []
        data = df[column]
        
        if pd.api.types.is_numeric_dtype(data):
            suggestions.extend(['histogram', 'box_plot'])
            
            # Si hay otra columna numérica, sugerir scatter
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                suggestions.append('scatter_plot')
                
        elif pd.api.types.is_categorical_dtype(data) or data.dtype == 'object':
            unique_values = data.nunique()
            if unique_values <= 20:
                suggestions.extend(['bar_chart', 'pie_chart'])
            else:
                suggestions.append('bar_chart')  # Solo bar chart para muchas categorías
        
        # Si parece ser una fecha
        if data.dtype == 'datetime64[ns]' or 'date' in column.lower():
            suggestions.append('time_series')
        
        return suggestions
