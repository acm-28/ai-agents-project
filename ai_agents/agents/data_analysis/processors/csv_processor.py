"""
Procesador para archivos CSV y datos tabulares.
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import io

from ai_agents.agents.data_analysis.data_models import DatasetInfo, DataSource

logger = logging.getLogger(__name__)


class CSVProcessor:
    """Procesador especializado para archivos CSV y datos tabulares."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.tsv', '.txt']
        self.encoding_options = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    def load_from_file(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Carga un archivo CSV en un DataFrame.
        
        Args:
            file_path: Ruta del archivo CSV
            **kwargs: Argumentos adicionales para pd.read_csv()
            
        Returns:
            DataFrame con los datos cargados
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si hay errores de formato
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Formato no soportado: {file_path.suffix}")
        
        # Intentar diferentes encodings si no se especifica
        encoding = kwargs.get('encoding', None)
        if not encoding:
            df = self._try_different_encodings(file_path, **kwargs)
        else:
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
        
        logger.info(f"Archivo cargado exitosamente: {file_path} ({df.shape[0]} filas, {df.shape[1]} columnas)")
        return df
    
    def _try_different_encodings(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Intenta cargar el archivo con diferentes encodings."""
        for encoding in self.encoding_options:
            try:
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                logger.info(f"Archivo cargado con encoding: {encoding}")
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # Si ningún encoding funciona, usar el default y manejar errores
        return pd.read_csv(file_path, encoding='utf-8', errors='replace', **kwargs)
    
    def load_from_string(self, csv_string: str, **kwargs) -> pd.DataFrame:
        """
        Carga datos CSV desde un string.
        
        Args:
            csv_string: String con formato CSV
            **kwargs: Argumentos adicionales para pd.read_csv()
            
        Returns:
            DataFrame con los datos
        """
        return pd.read_csv(io.StringIO(csv_string), **kwargs)
    
    def save_to_file(self, df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        """
        Guarda un DataFrame en un archivo CSV.
        
        Args:
            df: DataFrame a guardar
            file_path: Ruta donde guardar el archivo
            **kwargs: Argumentos adicionales para df.to_csv()
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configuraciones default para CSV
        csv_kwargs = {
            'index': False,
            'encoding': 'utf-8',
            **kwargs
        }
        
        df.to_csv(file_path, **csv_kwargs)
        logger.info(f"DataFrame guardado en: {file_path}")
    
    def preview_file(self, file_path: Union[str, Path], rows: int = 5) -> Dict[str, Any]:
        """
        Obtiene una vista previa del archivo CSV.
        
        Args:
            file_path: Ruta del archivo
            rows: Número de filas a mostrar
            
        Returns:
            Diccionario con información de vista previa
        """
        file_path = Path(file_path)
        
        try:
            # Leer solo las primeras filas para vista previa
            df_preview = pd.read_csv(file_path, nrows=rows)
            
            # Obtener información básica del archivo completo
            df_info = pd.read_csv(file_path, nrows=0)  # Solo headers
            
            return {
                'file_path': str(file_path),
                'file_size_mb': file_path.stat().st_size / 1024 / 1024,
                'columns': df_info.columns.tolist(),
                'column_count': len(df_info.columns),
                'preview_data': df_preview.to_dict('records'),
                'data_types': df_preview.dtypes.astype(str).to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo vista previa: {e}")
            return {'error': str(e)}
    
    def detect_delimiter(self, file_path: Union[str, Path], sample_size: int = 1024) -> str:
        """
        Detecta automáticamente el delimitador del archivo.
        
        Args:
            file_path: Ruta del archivo
            sample_size: Tamaño de muestra para detección
            
        Returns:
            Delimitador detectado
        """
        import csv
        
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            sample = file.read(sample_size)
            
        sniffer = csv.Sniffer()
        try:
            delimiter = sniffer.sniff(sample).delimiter
            logger.info(f"Delimitador detectado: '{delimiter}'")
            return delimiter
        except:
            logger.warning("No se pudo detectar delimitador, usando coma por defecto")
            return ','
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida la calidad de los datos cargados.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            Diccionario con resultados de validación
        """
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'data_types': df.dtypes.astype(str).to_dict(),
            'numeric_columns': df.select_dtypes(include='number').columns.tolist(),
            'categorical_columns': df.select_dtypes(include='object').columns.tolist(),
            'datetime_columns': df.select_dtypes(include='datetime').columns.tolist()
        }
        
        # Calcular estadísticas adicionales
        validation_results['missing_percentage'] = {
            col: (count / len(df)) * 100 
            for col, count in validation_results['missing_values'].items()
        }
        
        return validation_results
    
    def create_sample_data(self, dataset_type: str = "sales") -> Dict[str, Any]:
        """
        Crea datos de muestra y retorna información estructurada.
        
        Args:
            dataset_type: Tipo de dataset a crear
            
        Returns:
            Diccionario con DataFrame y información del dataset
        """
        try:
            # Crear DataFrame de muestra
            df = self.create_sample_dataset(dataset_type)
            
            # Crear información del dataset
            dataset_info = DatasetInfo(
                name=f"Sample {dataset_type.title()} Dataset",
                shape=(len(df), len(df.columns)),
                columns=list(df.columns),
                dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
                memory_usage=df.memory_usage(deep=True).sum() / 1024**2,  # MB
                null_counts={col: int(df[col].isnull().sum()) for col in df.columns}
            )
            
            return {
                'dataframe': df,
                'dataset_info': dataset_info,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error creando datos de muestra: {str(e)}")
            return {'error': f'Error creando datos de muestra: {str(e)}'}

    def create_sample_dataset(self, dataset_type: str = "sales") -> pd.DataFrame:
        """
        Crea un dataset de ejemplo para pruebas.
        
        Args:
            dataset_type: Tipo de dataset a crear ('sales', 'customers', 'products')
            
        Returns:
            DataFrame con datos de ejemplo
        """
        import numpy as np
        from datetime import datetime, timedelta
        
        np.random.seed(42)
        
        if dataset_type == "sales":
            n_rows = 1000
            start_date = datetime(2022, 1, 1)
            dates = [start_date + timedelta(days=i) for i in range(n_rows)]
            
            data = {
                'Date': dates,
                'Product': np.random.choice(['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Printer'], n_rows),
                'Category': np.random.choice(['Electronics', 'Accessories', 'Hardware'], n_rows),
                'Price': np.random.uniform(50, 2000, n_rows).round(2),
                'Quantity': np.random.randint(1, 10, n_rows),
                'Customer_ID': np.random.randint(1000, 9999, n_rows),
                'Sales_Rep': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana'], n_rows)
            }
            
        elif dataset_type == "customers":
            n_rows = 500
            data = {
                'Customer_ID': range(1000, 1000 + n_rows),
                'Name': [f'Customer_{i}' for i in range(n_rows)],
                'Age': np.random.randint(18, 80, n_rows),
                'City': np.random.choice(['Madrid', 'Barcelona', 'Valencia', 'Sevilla'], n_rows),
                'Segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_rows),
                'Registration_Date': [datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1000)) for _ in range(n_rows)]
            }
            
        else:  # products
            n_rows = 100
            data = {
                'Product_ID': range(1, n_rows + 1),
                'Product_Name': [f'Product_{i}' for i in range(1, n_rows + 1)],
                'Category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_rows),
                'Price': np.random.uniform(10, 500, n_rows).round(2),
                'Stock': np.random.randint(0, 100, n_rows),
                'Supplier': np.random.choice(['Supplier_A', 'Supplier_B', 'Supplier_C'], n_rows)
            }
        
        df = pd.DataFrame(data)
        logger.info(f"Dataset de ejemplo '{dataset_type}' creado: {df.shape}")
        return df
