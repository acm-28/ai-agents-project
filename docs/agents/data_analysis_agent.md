# Agent3: Data Analysis Agent

## Descripción General

El `DataAnalysisAgent` es un agente inteligente especializado en análisis de datos que combina tres tecnologías clave:

1. **Memoria Conversacional**
2. **Estructura LangGraph por Nodos**
3. **Capacidades de Análisis de Datos** (basado en `guia_agent3_data_analysis`)

## Características Principales

### 🧠 Memoria Conversacional
- Mantiene el contexto de conversaciones por sesión
- Permite referencias a análisis anteriores ("como vimos antes", "comparado con el resultado anterior")
- Soporte para múltiples sesiones independientes
- Gestión automática del historial de consultas

### 🔄 Flujo Estructurado con LangGraph
El agente procesa las consultas a través de 5 nodos especializados:

1. **Input Processing**: Analiza la intención de la consulta
2. **Data Preparation**: Valida y prepara el dataset
3. **Analysis Execution**: Ejecuta el análisis usando pandas
4. **Result Formatting**: Formatea la respuesta de manera comprensible
5. **Memory Update**: Actualiza la memoria de la sesión

### 📊 Análisis de Datos Avanzado
- Integración con `langchain_experimental.agents.agent_toolkits.create_pandas_dataframe_agent`
- Soporte para CSV, Excel y DataFrames de pandas
- Análisis estadísticos automáticos
- Respuestas en lenguaje natural

## Instalación y Configuración

### Dependencias
```bash
pip install -r requirements.txt
```

Las dependencias principales incluyen:
- `langchain>=0.1.0`
- `langchain-experimental>=0.1.0`
- `langgraph>=0.1.0`
- `pandas>=1.5.0`
- `numpy>=1.21.0`

### Variables de Entorno
Crear un archivo `.env` con:
```
OPENAI_API_KEY=tu_api_key_aqui
```

## Uso Básico

### Inicialización
```python
from agents.agent3_data_analysis import DataAnalysisAgent

# Crear e inicializar el agente
agent = DataAnalysisAgent()
agent.initialize()
```

### Consultas Simples
```python
# Hacer una consulta básica
response = agent.respond("¿Cuántas filas tiene el dataset?", "mi_sesion")
print(response)
```

### Cargar Dataset Personalizado
```python
import pandas as pd

# Cargar desde archivo
result = agent.load_dataset("mi_archivo.csv")

# O desde DataFrame
df = pd.read_csv("datos.csv")
result = agent.load_dataset(dataframe=df)
```

## Ejemplos de Uso

### Análisis Básico
```python
# Información general
agent.respond("¿Cuáles son las columnas del dataset?", "sesion1")
agent.respond("¿Cuántas filas tiene?", "sesion1")
agent.respond("Dame un resumen estadístico", "sesion1")

# Análisis específico
agent.respond("¿Cuál es el precio promedio?", "sesion1")
agent.respond("¿Qué marca aparece más frecuentemente?", "sesion1")
```

### Memoria Conversacional
```python
# Primera consulta
agent.respond("¿Cuál es el precio promedio de los autos?", "sesion1")

# Consulta que hace referencia a la anterior
agent.respond("¿Y cuál es el precio máximo?", "sesion1")

# Consulta comparativa
agent.respond("Compara esos dos valores", "sesion1")
```

### Múltiples Sesiones
```python
# Sesión para análisis de precios
agent.respond("Analiza los precios", "precios")

# Sesión independiente para marcas
agent.respond("Analiza las marcas", "marcas")

# Las sesiones mantienen contexto independiente
agent.respond("¿Cuál era el promedio?", "precios")  # Se refiere a precios
agent.respond("¿Cuál era la más común?", "marcas")   # Se refiere a marcas
```

## Métodos Disponibles

### Métodos Principales
- `initialize()`: Inicializa el agente
- `respond(message, session_id)`: Procesa una consulta
- `load_dataset(file_path, dataframe)`: Carga un dataset
- `get_dataset_info()`: Información del dataset actual

### Gestión de Memoria
- `get_session_history(session_id)`: Obtiene historial de una sesión
- `clear_session_memory(session_id)`: Limpia memoria de una sesión
- `get_available_sessions()`: Lista sesiones activas

## Tipos de Análisis Soportados

### Descriptivos
- Estadísticas básicas (media, mediana, moda)
- Información de columnas y tipos de datos
- Conteo de valores únicos
- Detección de valores nulos

### Comparativos
- Comparaciones entre grupos
- Correlaciones entre variables
- Análisis por categorías

### Temporales
- Tendencias a lo largo del tiempo
- Análisis por períodos
- Estacionalidad

### Filtrado y Agregación
- Filtros por condiciones específicas
- Agrupaciones por categorías
- Operaciones de suma, promedio, etc.

## Dataset de Ejemplo

El agente incluye un dataset de ejemplo con 1000 registros de ventas de autos:

**Columnas:**
- `Date`: Fecha de venta
- `Make`: Marca del auto
- `Model`: Tipo de modelo
- `Color`: Color del auto
- `Year`: Año del auto
- `Price`: Precio de venta
- `Mileage`: Kilometraje
- `EngineSize`: Tamaño del motor
- `FuelEfficiency`: Eficiencia de combustible
- `SalesPerson`: Vendedor

## Estructura Interna

### Estado de LangGraph
```python
class DataAnalysisState(TypedDict):
    query: str                      # Consulta original
    dataset_info: Dict[str, Any]    # Información del dataset
    analysis_result: str            # Resultado del análisis
    conversation_history: List[str] # Historial de la sesión
    session_id: str                 # ID de sesión
    final_response: str             # Respuesta final
    dataframe: Optional[pd.DataFrame] # DataFrame actual
    pandas_agent: Optional[Any]     # Agente pandas
    error_message: Optional[str]    # Mensajes de error
```

### Flujo de Nodos
1. **Input Processing**: Clasifica la intención (descriptive, comparative, temporal, etc.)
2. **Data Preparation**: Valida dataset y extrae metadatos
3. **Analysis Execution**: Ejecuta análisis con pandas agent
4. **Result Formatting**: Mejora formato de respuesta con LLM
5. **Memory Update**: Guarda en historial de sesión

## Comandos de Demo

### Ejecutar Demo Completo
```bash
cd src
python demo_agent3.py
```

### Ejecutar Tests
```bash
cd src
python tests/test_agent3.py
```

### Ejecutar desde Main
```bash
cd src
python main.py
# Seleccionar opción 4
```

## Casos de Uso Detallados

### 🔍 Ejemplo 1: Análisis Exploratorio Completo

**Escenario:** Analizar ventas de una empresa desde archivo CSV

```python
# Inicialización
agent = DataAnalysisAgent()
await agent.initialize()

# Sesión de análisis exploratorio
session = "analisis_ventas_2024"

# 1. Carga de datos
response1 = await agent.respond(
    "Carga el archivo ventas_2024.csv", 
    session
)
print(response1.content)
# Output: "Datos cargados exitosamente:
#         - Archivo: ventas_2024.csv
#         - Filas: 1,250
#         - Columnas: 8
#         - Columnas: fecha, producto, ventas, region, vendedor, categoria, precio_unitario, descuento"

# 2. Perfil general del dataset
response2 = await agent.respond(
    "Dame un resumen general del dataset",
    session
)
# Output: Información sobre tipos de datos, valores faltantes, estadísticas básicas

# 3. Análisis específico
response3 = await agent.respond(
    "¿Cuál es la región con mayores ventas?",
    session
)

# 4. Análisis temporal
response4 = await agent.respond(
    "Muéstrame las tendencias de ventas por mes",
    session
)

# 5. Análisis comparativo (usando memoria conversacional)
response5 = await agent.respond(
    "Compara las ventas de esa región con las otras",
    session  # El agente recuerda cuál era "esa región"
)
```

### 📊 Ejemplo 2: Análisis de Correlaciones

**Escenario:** Entender relaciones entre variables

```python
session = "correlaciones_productos"

# 1. Cargar datos de muestra
await agent.respond("Crea datos de muestra de productos", session)

# 2. Análisis de correlaciones
response = await agent.respond(
    "Analiza las correlaciones entre precio, stock y ventas",
    session
)

# El sistema ejecuta internamente:
# - stats_processor.correlation_analysis()
# - Genera matriz de correlación
# - Identifica correlaciones significativas
# - Explica las relaciones en lenguaje natural

print(response.content)
# Output: "Análisis de correlaciones completado:
#         
#         Correlaciones significativas encontradas:
#         - Precio vs Stock: correlación negativa (-0.65)
#         - Stock vs Ventas: correlación positiva (0.78)
#         
#         Interpretación:
#         - Los productos más caros tienden a tener menos stock
#         - Mayor stock se asocia con mayores ventas"
```

### 📈 Ejemplo 3: Detección de Anomalías

```python
session = "deteccion_anomalias"

# 1. Análisis de outliers
response = await agent.respond(
    "Detecta valores atípicos en las ventas diarias",
    session
)

# El sistema internamente:
# 1. Aplica método IQR (Interquartile Range)
# 2. Identifica valores fuera de Q1 - 1.5*IQR y Q3 + 1.5*IQR
# 3. Analiza patrones en los outliers
# 4. Sugiere posibles explicaciones

print(response.content)
# Output: "Outliers detectados en ventas diarias:
#         
#         - 5 valores atípicos encontrados
#         - Fechas: 2024-03-15, 2024-07-04, 2024-12-25, 2024-12-31, 2024-01-01
#         - Valores: $45,231 (Máx), $892 (Mín)
#         
#         Posibles explicaciones:
#         - Picos en fechas especiales (Navidad, Año Nuevo)
#         - Caída en día festivo nacional
#         
#         Recomendación: Considerar estacionalidad en análisis futuros"
```

### 🎯 Ejemplo 4: Análisis Guiado por Preguntas

**Flujo conversacional inteligente:**

```python
session = "analisis_guiado"

# El agente mantiene contexto entre preguntas
queries = [
    "¿Cuál es el producto más vendido?",
    "¿En qué regiones se vende más?", 
    "¿Cuál es la estacionalidad de ese producto?",
    "¿Qué factores influyen en sus ventas?",
    "¿Hay diferencias por vendedor?",
    "Basándote en todo esto, ¿qué recomendaciones darías?"
]

for i, query in enumerate(queries, 1):
    print(f"\n--- Pregunta {i} ---")
    response = await agent.respond(query, session)
    print(f"Q: {query}")
    print(f"A: {response.content}")
    
# El agente recuerda:
# - "ese producto" se refiere al más vendido mencionado antes
# - "todo esto" incluye el análisis previo completo
# - Construye recomendaciones basadas en la conversación completa
```

### 🔄 Ejemplo 5: Múltiples Sesiones Independientes

```python
# Sesión 1: Análisis de ventas
await agent.respond("Analiza ventas por trimestre", "ventas_q1_q4")
await agent.respond("¿Cuál fue el mejor trimestre?", "ventas_q1_q4")

# Sesión 2: Análisis de inventario (independiente)
await agent.respond("Analiza niveles de stock", "inventario_2024")
await agent.respond("¿Qué productos están en stock bajo?", "inventario_2024")

# Sesión 3: Análisis de clientes (independiente)
await agent.respond("Segmenta clientes por valor", "segmentacion_clientes")
await agent.respond("¿Cuál es el perfil del cliente premium?", "segmentacion_clientes")

# Las preguntas "¿Cuál fue..." en cada sesión se refieren 
# al contexto específico de esa sesión únicamente
```

### 🛠️ Ejemplo 6: Análisis con Datos Personalizados

**Preparación de datos propios:**

```python
# 1. Estructura recomendada para tu archivo CSV
"""
fecha,cliente,producto,cantidad,precio_total,descuento,vendedor,region
2024-01-15,CLIENTE_001,LAPTOP_GAMING,2,2400.00,5.0,Santiago,Norte
2024-01-15,CLIENTE_002,MOUSE_WIRELESS,1,45.90,0.0,María,Sur
2024-01-16,CLIENTE_003,TECLADO_MECANICO,1,120.50,10.0,Carlos,Este
"""

# 2. Carga y análisis
session = "mi_analisis_personalizado"

# Cargar tus datos
await agent.respond(
    "Carga el archivo mi_dataset_ventas.csv", 
    session
)

# Análisis específicos de tu negocio
await agent.respond(
    "¿Cuál es el ticket promedio por cliente?",
    session
)

await agent.respond(
    "¿Qué vendedor tiene mejor performance?",
    session
)

await agent.respond(
    "¿Hay patrones estacionales en las ventas?",
    session
)

await agent.respond(
    "¿El descuento impacta en el volumen de ventas?",
    session
)
```

### 📋 Ejemplo 7: Reportes Automatizados

```python
session = "reporte_mensual"

# El agente puede generar reportes estructurados
response = await agent.respond(
    """Genera un reporte completo que incluya:
    1. Resumen ejecutivo de ventas
    2. Top 5 productos por ventas
    3. Performance por región
    4. Análisis de tendencias
    5. Recomendaciones de acción""",
    session
)

# El sistema estructura automáticamente la respuesta
print(response.content)
# Output:
# """
# REPORTE MENSUAL DE VENTAS - ENERO 2024
# =====================================
# 
# 1. RESUMEN EJECUTIVO
# - Ventas totales: $2,450,000 (+15% vs mes anterior)
# - Unidades vendidas: 1,250 (+8% vs mes anterior)
# - Ticket promedio: $1,960 (+6% vs mes anterior)
# 
# 2. TOP 5 PRODUCTOS
# 1. Laptop Gaming X1: $850,000 (34.7% del total)
# 2. Monitor 4K Pro: $420,000 (17.1% del total)
# ...
# 
# 3. PERFORMANCE POR REGIÓN
# - Norte: $980,000 (40% del total) - LÍDER
# - Sur: $735,000 (30% del total)
# ...
# 
# 4. TENDENCIAS IDENTIFICADAS
# - Crecimiento sostenido en electrónicos premium
# - Declive en accesorios básicos
# ...
# 
# 5. RECOMENDACIONES
# - Incrementar stock de Laptop Gaming X1 en región Norte
# - Revisar estrategia de pricing en accesorios
# ...
# """
```

### 🎯 Casos de Uso Empresariales

#### **E-commerce:**
```python
# Análisis de comportamiento de compra
"¿Cuál es el patrón de compra por hora del día?"
"¿Qué productos se compran juntos frecuentemente?"
"¿Cuál es la estacionalidad por categoría?"
```

#### **Retail:**
```python
# Análisis de inventario y ventas
"¿Qué productos tienen rotación más lenta?"
"¿Cuál es el punto de reorden óptimo por producto?"
"¿Hay diferencias de ventas por ubicación de tienda?"
```

#### **SaaS/Software:**
```python
# Análisis de métricas de usuario
"¿Cuál es el patrón de retención por cohorte?"
"¿Qué features correlacionan con mayor engagement?"
"¿Cuál es el tiempo promedio hasta conversión?"
```

#### **Finanzas:**
```python
# Análisis de riesgo y performance
"¿Cuál es la distribución de morosidad por segmento?"
"¿Hay estacionalidad en los pagos atrasados?"
"¿Qué factores predicen mejor el riesgo crediticio?"
```

### 📊 Tipos de Análisis Soportados Detallados

#### **Descriptivos Básicos:**
- Medidas de tendencia central (media, mediana, moda)
- Medidas de dispersión (desviación estándar, varianza, rango)
- Distribuciones por percentiles
- Conteos y frecuencias

#### **Comparativos:**
- Comparaciones entre grupos/categorías
- Análisis de diferencias estadísticas
- Rankings y ordenamientos
- Análisis de proporciones

#### **Correlacionales:**
- Matriz de correlaciones completa
- Correlaciones parciales
- Identificación de multicolinealidad
- Análisis de dependencias

#### **Temporales:**
- Análisis de tendencias
- Detección de estacionalidad
- Análisis de crecimiento periodo a periodo
- Identificación de puntos de inflexión

#### **Detección de Anomalías:**
- Outliers univariados (método IQR, Z-score)
- Outliers multivariados
- Análisis de valores atípicos temporales
- Identificación de patrones anómalos

### 1. Análisis Exploratorio de Datos
- Usuarios no técnicos pueden explorar datasets
- Obtener insights rápidos sin programar
- Entender estructura y calidad de los datos

### 2. Reportes Automatizados
- Generar reportes en lenguaje natural
- Comparaciones automáticas entre períodos
- Alertas sobre anomalías en datos

### 3. Análisis Conversacional
- Sesiones de análisis interactivo
- Refinamiento iterativo de consultas
- Exploración guiada por preguntas

### 4. Prototipado Rápido
- Validación rápida de hipótesis
- Análisis ad-hoc durante reuniones
- Exploración de nuevas fuentes de datos

## Arquitectura y Funcionamiento Interno

### 🏗️ Componentes del Sistema

El sistema de análisis de datos está construido con una arquitectura modular que separa responsabilidades:

```
PandasAgent (Coordinador Principal)
├── Processors (Procesadores de Datos)
│   ├── CSVProcessor - Manejo de archivos CSV
│   ├── StatsProcessor - Análisis estadísticos  
│   └── VisualizationProcessor - Generación de gráficos
├── Workflows (Flujos de LangGraph)
│   └── DataAnalysisWorkflow - Orchestación del análisis
├── Tools (Herramientas Especializadas)
│   └── DataAnalysisTools - Utilidades de análisis
└── Models (Estructuras de Datos)
    ├── AnalysisRequest - Solicitudes de análisis
    ├── AnalysisResult - Resultados estructurados
    └── DataAnalysisState - Estado del workflow
```

### 🔄 Flujo Detallado de Procesamiento

#### 1. **Recepción de Input** (`PandasAgent.process()`)
```python
# El agente recibe un mensaje del usuario
input_data = "Analiza las ventas por región"

# Se normaliza la entrada
if isinstance(input_data, str):
    message = input_data
    kwargs = {}
elif isinstance(input_data, dict):
    message = input_data.get('message', '')
    kwargs = {k: v for k, v in input_data.items() if k != 'message'}'}
```

#### 2. **Clasificación de Request** (`_classify_request()`)
```python
def _classify_request(self, message: str) -> str:
    """El sistema analiza palabras clave para determinar la intención"""
    message_lower = message.lower()
    
    # Palabras clave para diferentes tipos de operaciones
    analyze_keywords = ['analizar', 'analyze', 'estadísticas', 'correlación']
    visualize_keywords = ['gráfico', 'plot', 'visualizar', 'chart']
    profile_keywords = ['perfil', 'profile', 'resumen', 'overview']
    load_keywords = ['cargar', 'load', 'archivo', 'csv', 'datos']
    
    # Retorna: "analyze_data", "visualize_data", "profile_data", etc.
```

**Tipos de requests detectados:**
- `load_data` - Cargar/crear datos
- `analyze_data` - Análisis estadístico
- `visualize_data` - Crear gráficos
- `profile_data` - Perfil del dataset
- `export_results` - Exportar resultados
- `general` - Consulta general

#### 3. **Manejo Específico por Tipo**

##### **Carga de Datos** (`_handle_data_loading()`)
```python
# Detecta origen de datos
if file_path:
    # Origen: Archivo CSV
    dataframe = await asyncio.to_thread(
        self.csv_processor.load_from_file, 
        file_path
    )
elif sample_data:
    # Origen: Datos de muestra
    result = await asyncio.to_thread(
        self.csv_processor.create_sample_data, 
        sample_data
    )
```

**Proceso de carga CSV:**
1. **Validación**: Verifica que el archivo existe y tiene formato válido
2. **Encoding Detection**: Intenta diferentes encodings (utf-8, latin1, cp1252, iso-8859-1)
3. **Carga**: Usa `pd.read_csv()` con parámetros optimizados
4. **Metadatos**: Crea `DatasetInfo` con estadísticas del dataset

##### **Análisis de Datos** (`_handle_data_analysis()`)
```python
# Verifica que hay datos cargados
if self.current_dataframe is None:
    return "Primero debes cargar datos antes de analizarlos."

# Ejecuta análisis en paralelo
analyses = {}

# 1. Estadísticas descriptivas
desc_stats = await asyncio.to_thread(
    self.stats_processor.descriptive_stats,
    self.current_dataframe,
    kwargs.get('target_columns')
)

# 2. Análisis de correlación
corr_analysis = await asyncio.to_thread(
    self.stats_processor.correlation_analysis,
    self.current_dataframe
)

# 3. Compila resultados
analyses['descriptive'] = desc_stats
analyses['correlation'] = corr_analysis
```

### � Fuentes de Datos Soportadas

#### **1. Archivos CSV/TSV** (Fuente Principal)
```python
# Formatos soportados
supported_formats = ['.csv', '.tsv', '.txt']

# Encodings manejados automáticamente
encoding_options = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

# Uso
agent.respond("Carga el archivo ventas_2024.csv", "sesion1")
```

**Características:**
- ✅ Detección automática de delimitadores (`,`, `;`, `\t`)
- ✅ Manejo robusto de diferentes encodings
- ✅ Validación de estructura y formato
- ✅ Vista previa antes de carga completa
- ✅ Detección automática de tipos de datos

**Ejemplo de carga:**
```python
# El sistema intenta diferentes configuraciones automáticamente
def _try_different_encodings(self, file_path):
    for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"Archivo cargado con encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            continue
    
    # Fallback con manejo de errores
    return pd.read_csv(file_path, encoding='utf-8', errors='replace')
```

#### **2. Datos de Muestra** (Para Testing y Demos)
```python
# Tipos disponibles
sample_types = ["sales", "customers", "products"]

# Uso
agent.respond("Crea datos de muestra de ventas", "sesion1")
agent.respond("Genera dataset de clientes", "sesion1")
```

**Datasets de muestra incluidos:**

**Sales Dataset (1000 registros):**
```python
{
    'Date': ['2022-01-01', '2022-01-02', ...],
    'Product': ['Laptop', 'Mouse', 'Keyboard', ...],
    'Category': ['Electronics', 'Accessories', ...],
    'Price': [1250.50, 25.99, 89.90, ...],
    'Quantity': [2, 5, 1, ...],
    'Customer_ID': [1001, 1002, 1003, ...],
    'Sales_Rep': ['Alice', 'Bob', 'Charlie', ...]
}
```

**Customers Dataset (500 registros):**
```python
{
    'Customer_ID': [1000, 1001, 1002, ...],
    'Name': ['Customer_1', 'Customer_2', ...],
    'Age': [25, 34, 45, ...],
    'City': ['Madrid', 'Barcelona', 'Valencia', ...],
    'Segment': ['Premium', 'Standard', 'Basic', ...],
    'Registration_Date': ['2020-03-15', '2020-05-22', ...]
}
```

**Products Dataset (100 registros):**
```python
{
    'Product_ID': [1, 2, 3, ...],
    'Product_Name': ['Product_1', 'Product_2', ...],
    'Category': ['Electronics', 'Clothing', 'Books', ...],
    'Price': [199.99, 49.90, 15.50, ...],
    'Stock': [50, 120, 8, ...],
    'Supplier': ['Supplier_A', 'Supplier_B', ...]
}
```

#### **3. Strings CSV** (Datos en Memoria)
```python
# Para datos que vienen como texto
csv_string = """nombre,edad,ciudad
Juan,25,Madrid
María,30,Barcelona"""

df = csv_processor.load_from_string(csv_string)
```

#### **4. Implementación Futura: Base de Datos JSON**

**Estructura propuesta:**
```json
{
  "ventas_2024": [
    {
      "fecha": "2024-01-01",
      "producto": "Laptop Gaming",
      "cantidad": 5,
      "precio": 1200.00,
      "region": "Norte",
      "vendedor": "Santiago"
    }
  ],
  "clientes_activos": [
    {
      "id": 1,
      "nombre": "Juan Pérez",
      "edad": 35,
      "ciudad": "Madrid",
      "segmento": "Premium"
    }
  ],
  "inventario": [
    {
      "producto_id": "LAPTOP001",
      "nombre": "Laptop Gaming X1",
      "categoria": "Electrónicos",
      "precio": 1200.00,
      "stock": 50,
      "proveedor": "TechCorp"
    }
  ]
}
```

**Ventajas de JSON Database:**
- ✅ Control total sobre los datos disponibles
- ✅ No depende de uploads de usuarios
- ✅ Múltiples datasets predefinidos
- ✅ Estructura consistente y validada
- ✅ Fácil mantenimiento y actualización
- ✅ Metadatos incluidos en la misma estructura

**Implementación propuesta:**
```python
class JSONProcessor:
    def __init__(self, json_database_path: str):
        self.database_path = Path(json_database_path)
        self.data_cache = {}
    
    def load_from_json_db(self, dataset_name: Optional[str] = None):
        """Carga dataset específico desde base JSON"""
        with open(self.database_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if dataset_name:
            if dataset_name in json_data:
                return pd.DataFrame(json_data[dataset_name])
            else:
                available = list(json_data.keys())
                raise ValueError(f"Dataset '{dataset_name}' no encontrado. Disponibles: {available}")
        
        # Si no se especifica, mostrar datasets disponibles
        return list(json_data.keys())
    
    def list_available_datasets(self) -> List[str]:
        """Lista todos los datasets disponibles"""
        with open(self.database_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        return list(json_data.keys())
```

### 🔍 Proceso de Detección de Fuente de Datos

```python
def _handle_data_loading(self, message: str, **kwargs):
    """El sistema determina automáticamente la fuente de datos"""
    
    file_path = kwargs.get('file_path')
    sample_data = kwargs.get('sample_data')
    dataset_name = kwargs.get('dataset_name')  # Para JSON DB futuro
    
    # 1. Buscar en kwargs explícitos
    if file_path:
        source = "file"
    elif sample_data:
        source = "sample"
    elif dataset_name:
        source = "json_db"
    
    # 2. Buscar en el mensaje del usuario
    else:
        words = message.split()
        for word in words:
            if word.endswith('.csv') or '/' in word or '\\' in word:
                file_path = word
                source = "file"
                break
            elif word in ['ventas', 'clientes', 'productos']:
                dataset_name = word
                source = "json_db"
                break
        else:
            # 3. Detectar intención de datos de muestra
            if any(keyword in message.lower() for keyword in ['muestra', 'sample', 'demo', 'ejemplo']):
                sample_data = "sales"  # Default
                source = "sample"
    
    # 4. Procesar según la fuente detectada
    if source == "file":
        result = await self.csv_processor.load_from_file(file_path)
    elif source == "sample": 
        result = await self.csv_processor.create_sample_data(sample_data)
    elif source == "json_db":
        result = await self.json_processor.load_from_json_db(dataset_name)
    else:
        return "Por favor especifica la fuente de datos: archivo CSV, datos de muestra, o dataset específico"
```

### 📈 Validación y Calidad de Datos

**Proceso automático de validación:**
```python
def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
    validation_results = {
        # Información básica
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        
        # Tipos de datos
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_columns': df.select_dtypes(include='number').columns.tolist(),
        'categorical_columns': df.select_dtypes(include='object').columns.tolist(),
        'datetime_columns': df.select_dtypes(include='datetime').columns.tolist(),
        
        # Calidad de datos
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'missing_percentage': {
            col: (count / len(df)) * 100 
            for col, count in df.isnull().sum().items()
        }
    }
    
    return validation_results
```

**Alertas de calidad generadas:**
- ⚠️ Columnas con >20% de valores faltantes
- ⚠️ Filas duplicadas encontradas
- ⚠️ Columnas con un solo valor único
- ⚠️ Posibles errores de tipo de dato
- ⚠️ Valores extremos detectados (outliers)


#### **CSVProcessor** - El Corazón de la Carga de Datos

**Responsabilidades:**
- Carga archivos CSV/TSV/TXT
- Detección automática de delimitadores
- Manejo robusto de encodings
- Validación de calidad de datos
- Generación de datos de muestra

**Flujo de carga de archivo:**
```python
def load_from_file(self, file_path):
    # 1. Validación de archivo
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    # 2. Validación de formato
    if file_path.suffix.lower() not in ['.csv', '.tsv', '.txt']:
        raise ValueError(f"Formato no soportado: {file_path.suffix}")
    
    # 3. Detección de encoding
    for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"Archivo cargado con encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            continue
    
    # 4. Fallback con manejo de errores
    return pd.read_csv(file_path, encoding='utf-8', errors='replace')
```

**Datos de muestra disponibles:**
```python
def create_sample_dataset(self, dataset_type="sales"):
    if dataset_type == "sales":
        # Dataset de ventas con 1000 filas
        # Columnas: Date, Product, Category, Price, Quantity, Customer_ID, Sales_Rep
        
    elif dataset_type == "customers":
        # Dataset de clientes con 500 filas
        # Columnas: Customer_ID, Name, Age, City, Segment, Registration_Date
        
    elif dataset_type == "products":
        # Dataset de productos con 100 filas
        # Columnas: Product_ID, Product_Name, Category, Price, Stock, Supplier
```

#### **StatsProcessor** - Motor de Análisis Estadístico

**Capacidades:**
- Estadísticas descriptivas completas
- Análisis de correlaciones
- Detección de outliers
- Análisis de distribuciones
- Comparaciones entre grupos

#### **VisualizationProcessor** - Generación de Gráficos

**Tipos de visualizaciones:**
- Histogramas y distribuciones
- Scatter plots y correlaciones
- Series temporales
- Gráficos de barras y comparaciones

### 🛠️ Herramientas de Análisis (DataAnalysisTools)

#### **Quick Profile** - Perfil Rápido del Dataset
```python
def quick_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
    profile = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'size_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'columns_list': list(df.columns)
        },
        'data_types': {
            'numeric': len(df.select_dtypes(include=[np.number]).columns),
            'categorical': len(df.select_dtypes(include=['object']).columns),
            'datetime': len(df.select_dtypes(include=['datetime64']).columns)
        },
        'missing_data': {
            'total_missing': df.isnull().sum().sum(),
            'percentage_missing': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
    }
    
    # Análisis detallado por columna
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info = {
                'min': df[col].min(),
                'max': df[col].max(), 
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std()
            }
        elif df[col].dtype == 'object':
            col_info = {
                'most_frequent': df[col].mode().iloc[0],
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head().to_dict()
            }
```

### 🔄 Workflows con LangGraph

El sistema usa **LangGraph** para orquestar el flujo de análisis de manera estructurada:

```python
def _build_graph(self) -> StateGraph:
    workflow = StateGraph(DataAnalysisState)
    
    # Nodos del workflow
    workflow.add_node("load_data", self.load_data_node)
    workflow.add_node("validate_data", self.validate_data_node) 
    workflow.add_node("analyze_data", self.analyze_data_node)
    workflow.add_node("create_visualizations", self.create_visualizations_node)
    workflow.add_node("generate_report", self.generate_report_node)
    
    # Flujo secuencial
    workflow.add_edge(START, "load_data")
    workflow.add_edge("load_data", "validate_data")
    workflow.add_edge("validate_data", "analyze_data") 
    workflow.add_edge("analyze_data", "create_visualizations")
    workflow.add_edge("create_visualizations", "generate_report")
    workflow.add_edge("generate_report", END)
```

**Cada nodo tiene responsabilidades específicas:**

1. **load_data_node**: Carga datos desde archivo o crea muestras
2. **validate_data_node**: Verifica calidad y estructura de datos
3. **analyze_data_node**: Ejecuta análisis estadísticos
4. **create_visualizations_node**: Genera gráficos si es necesario
5. **generate_report_node**: Compila resultados en respuesta final

### 🎯 Flujo Completo de una Consulta

**Ejemplo paso a paso:**
```
Usuario: "Analiza las ventas por región en el archivo ventas_2024.csv"

1. RECEPCIÓN (PandasAgent.process)
   ├── Input: "Analiza las ventas por región en el archivo ventas_2024.csv"
   └── Parseado: message="Analiza...", file_path="ventas_2024.csv"

2. CLASIFICACIÓN (_classify_request)
   ├── Detecta palabras: ["analiza", "archivo", "csv"]
   └── Tipo determinado: "analyze_data" 

3. CARGA DE DATOS (_handle_data_loading)
   ├── CSVProcessor.load_from_file("ventas_2024.csv")
   ├── Detección encoding: UTF-8 ✓
   ├── DataFrame creado: 1000 filas, 7 columnas
   └── DatasetInfo generado con metadatos

4. ANÁLISIS (_handle_data_analysis)
   ├── StatsProcessor.descriptive_stats()
   │   ├── Media de ventas por región
   │   ├── Totales por región  
   │   └── Distribución regional
   ├── StatsProcessor.correlation_analysis()
   │   └── Correlaciones entre variables
   └── Resultados compilados

5. RESPUESTA FINAL
   ├── Formateo de resultados en lenguaje natural
   ├── Metadatos para uso posterior
   └── AgentResponse retornado al usuario
```

### 💾 Gestión de Estado y Memoria

```python
class DataAnalysisState(BaseModel):
    # Entrada del usuario  
    query: str = "Consulta original"
    session_id: str = "ID de sesión único"
    
    # Información del dataset
    dataset_info: Dict[str, Any] = "Metadatos del dataset"
    dataframe_id: Optional[str] = "ID del DataFrame cargado"
    
    # Estado del análisis
    analysis_type: Optional[AnalysisType] = "Tipo de análisis detectado"
    analysis_result: str = "Resultado del análisis"
    
    # Contexto conversacional
    conversation_history: List[str] = "Historial de consultas"
    
    # Resultado final
    final_response: str = "Respuesta formateada para el usuario"
    
    # Objetos internos (no serializables)
    dataframe: Optional[pd.DataFrame] = "DataFrame actual en memoria"
    pandas_agent: Optional[Any] = "Agente pandas activo"
```

### 🔧 Extensibilidad del Sistema

El sistema está diseñado para ser fácilmente extensible:

**Para agregar nuevos tipos de datos:**
```python
# 1. Crear nuevo procesador
class JSONProcessor:
    def load_from_json_db(self, dataset_name):
        # Lógica de carga desde JSON
        
# 2. Registrar en PandasAgent
def __init__(self):
    self.json_processor = JSONProcessor()
    
# 3. Modificar clasificación de requests
def _classify_request(self, message):
    if "dataset" in message.lower():
        return "load_json_dataset"
```

**Para agregar nuevos tipos de análisis:**
```python
# 1. Crear método en StatsProcessor
def advanced_ml_analysis(self, df):
    # Análisis de machine learning
    
# 2. Agregar a _handle_data_analysis
ml_analysis = await asyncio.to_thread(
    self.stats_processor.advanced_ml_analysis,
    self.current_dataframe
)
analyses['machine_learning'] = ml_analysis
```

## Limitaciones

- Requiere API key de OpenAI
- Limitado a análisis que pandas puede realizar
- No genera visualizaciones automáticamente
- Memoria de sesión no persiste entre reinicios
- Actualmente solo soporta archivos CSV/TSV
- No incluye análisis de machine learning avanzado
- Visualizaciones requieren configuración adicional

## Extensiones Futuras

- [ ] Soporte para bases de datos SQL
- [ ] Generación automática de gráficos
- [ ] Persistencia de memoria en disco
- [ ] Integración con Jupyter Notebooks
- [ ] Exportación de resultados a PDF/Excel
- [ ] Análisis de texto y NLP
- [ ] Machine Learning básico

## Troubleshooting Detallado

### 🔧 Problemas Comunes y Soluciones

#### **1. Errores de Configuración**

**Error: OPENAI_API_KEY no encontrada**
```bash
# Síntoma
Error: "OpenAI API key not found"

# Diagnóstico
1. Verificar archivo .env existe en directorio raíz
2. Confirmar formato correcto en .env:
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
3. Verificar que no hay espacios extra alrededor del =
4. Confirmar que el archivo .env no tiene extensión .txt oculta

# Solución
1. Crear/editar archivo .env en la raíz del proyecto
2. Agregar la línea: OPENAI_API_KEY=tu_api_key_real
3. Reiniciar la aplicación
4. Verificar carga: print(os.getenv('OPENAI_API_KEY'))
```

**Error: Import pandas could not be resolved**
```bash
# Síntoma
ImportError: No module named 'pandas'

# Diagnóstico
1. Verificar entorno virtual activado
2. Confirmar instalación de dependencias

# Solución
pip install -r requirements.txt
# O específicamente:
pip install pandas numpy langchain langchain-experimental langgraph
```

#### **2. Errores de Carga de Datos**

**Error: Archivo CSV no encontrado**
```python
# Síntoma
FileNotFoundError: Archivo no encontrado: datos.csv

# Diagnóstico
print(os.getcwd())  # Verificar directorio actual
print(os.path.exists("datos.csv"))  # Verificar existencia

# Soluciones
1. Usar ruta absoluta:
   "/Users/usuario/Documents/datos.csv"
2. Verificar ubicación relativa al script
3. Usar Path objects:
   from pathlib import Path
   file_path = Path("data") / "datos.csv"
```

**Error: Encoding de caracteres**
```python
# Síntoma
UnicodeDecodeError: 'utf-8' codec can't decode byte

# El sistema maneja esto automáticamente, pero si persiste:
# Solución manual
df = pd.read_csv("archivo.csv", encoding='latin1')
# O
df = pd.read_csv("archivo.csv", encoding='cp1252')
```

**Error: Formato de datos inválido**
```python
# Síntoma
pandas.errors.EmptyDataError: No columns to parse from file

# Diagnóstico
1. Verificar que el archivo no esté vacío
2. Confirmar que tiene headers
3. Verificar delimitadores correctos

# Solución
# Vista previa del archivo
with open("datos.csv", 'r') as f:
    print(f.read(500))  # Primeros 500 caracteres
```

#### **3. Errores de Análisis**

**Error: "Primero debes cargar datos"**
```python
# Síntoma
"Primero debes cargar datos antes de analizarlos."

# Causa
El agent no tiene un DataFrame cargado en current_dataframe

# Solución
1. Cargar datos explícitamente:
   await agent.respond("Carga datos de muestra", session)
2. Verificar carga exitosa:
   if agent.current_dataframe is not None:
       print(f"Datos cargados: {agent.current_dataframe.shape}")
```

**Error: Columna no encontrada**
```python
# Síntoma
KeyError: 'columna_inexistente'

# Diagnóstico
print(agent.current_dataframe.columns.tolist())

# Solución
1. Verificar nombres exactos de columnas
2. Usar consultas genéricas:
   "¿Qué columnas tiene el dataset?"
3. Revisar case-sensitivity
```

#### **4. Errores de Memoria y Performance**

**Error: Memoria insuficiente**
```python
# Síntoma
MemoryError: Unable to allocate array

# Diagnóstico
print(f"Tamaño del dataset: {agent.current_dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Soluciones
1. Procesar en chunks:
   chunk_size = 10000
   for chunk in pd.read_csv("archivo_grande.csv", chunksize=chunk_size):
       # Procesar chunk
       
2. Optimizar tipos de datos:
   df['columna_int'] = df['columna_int'].astype('int32')  # En lugar de int64
   df['columna_cat'] = df['columna_cat'].astype('category')
   
3. Usar sample para pruebas:
   df_sample = df.sample(n=1000)  # Solo 1000 filas para prueba
```

**Error: Timeouts en análisis**
```python
# Síntoma
asyncio.TimeoutError: Analysis took too long

# Soluciones
1. Usar datasets más pequeños para pruebas
2. Limitar análisis a columnas específicas:
   await agent.respond("Analiza solo las columnas precio y cantidad", session)
3. Dividir análisis complejos en pasos:
   await agent.respond("Dame estadísticas básicas", session)
   await agent.respond("Ahora analiza correlaciones", session)
```

### 🛡️ Mejores Prácticas

#### **1. Preparación de Datos**

**Estructura de archivos CSV recomendada:**
```csv
# ✅ Buena estructura
fecha,producto,ventas,region
2024-01-01,Laptop,1200.50,Norte
2024-01-02,Mouse,25.90,Sur

# ❌ Evitar
# Fecha,Producto con espacios,Ventas ($),Región/Zona
# 01/01/2024,"Laptop Gaming ""Pro""","$1,200.50",Norte/Centro
```

**Convenciones de nombres:**
```python
# ✅ Recomendado
columnas = [
    'fecha',           # Sin espacios
    'producto_id',     # Separadores con underscore
    'precio_unitario', # Descriptivo y claro
    'cantidad',        # Sin abreviaciones confusas
    'descuento_pct'    # Unidades claras
]

# ❌ Evitar
columnas = [
    'Fecha de Venta',  # Espacios
    'Prod ID',         # Espacios y abreviación
    'Precio ($)',      # Caracteres especiales
    'Cant.',           # Abreviación ambigua
    'Desc%'            # Símbolo especial
]
```

**Validación previa:**
```python
# Validar antes de análisis principal
def validate_csv_before_analysis(file_path):
    # 1. Verificar estructura básica
    preview = pd.read_csv(file_path, nrows=5)
    print(f"Columnas: {preview.columns.tolist()}")
    print(f"Tipos: {preview.dtypes.to_dict()}")
    
    # 2. Verificar datos faltantes
    missing_pct = (preview.isnull().sum() / len(preview)) * 100
    print(f"Datos faltantes: {missing_pct.to_dict()}")
    
    # 3. Verificar duplicados en preview
    duplicates = preview.duplicated().sum()
    print(f"Duplicados en muestra: {duplicates}")
    
    return True if duplicates == 0 and missing_pct.max() < 50 else False
```

#### **2. Optimización de Consultas**

**Consultas efectivas:**
```python
# ✅ Específicas y claras
"¿Cuál es el precio promedio de laptops en la región Norte?"
"Compara las ventas de enero vs febrero 2024"
"¿Qué vendedor tiene mayor volumen de ventas?"

# ❌ Ambiguas o muy generales
"Analiza todo"
"¿Qué pasa con los datos?"
"Hazme un análisis completo"
```

**Uso de contexto conversacional:**
```python
# ✅ Aprovechar memoria de sesión
session = "analisis_ventas"
await agent.respond("¿Cuál es el producto más vendido?", session)
await agent.respond("¿En qué región se vende más ese producto?", session)  # "ese producto" se entiende
await agent.respond("¿Cuál es la tendencia de ventas para esa combinación?", session)  # Contexto acumulado

# ❌ Repetir información
await agent.respond("¿Cuál es el producto más vendido?", session)
await agent.respond("¿En qué región se vende más el producto más vendido?", session)  # Redundante
```

#### **3. Gestión de Sesiones**

**Organización por propósito:**
```python
# ✅ Sesiones específicas
sessions = {
    "analisis_ventas_q1": "Análisis del primer trimestre",
    "comparativa_productos": "Comparación entre productos",
    "tendencias_anuales": "Análisis de tendencias año completo",
    "segmentacion_clientes": "Análisis de segmentos de clientes"
}

# ❌ Sesiones mezcladas
session = "analisis_general"  # Todo mezclado, pierde contexto
```

**Limpieza de memoria:**
```python
# Limpiar sesiones cuando no se necesiten
agent.clear_session_memory("analisis_temporal_old")

# Ver sesiones activas
active_sessions = agent.get_available_sessions()
print(f"Sesiones activas: {active_sessions}")
```

#### **4. Monitoreo y Debugging**

**Logging personalizado:**
```python
import logging

# Configurar logging para debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Ver logs del agente
logger = logging.getLogger('ai_agents.agents.data_analysis')
logger.setLevel(logging.DEBUG)
```

**Verificación de estado:**
```python
# Función de diagnóstico personalizada
def debug_agent_state(agent):
    print("=== ESTADO DEL AGENTE ===")
    print(f"DataFrame cargado: {agent.current_dataframe is not None}")
    if agent.current_dataframe is not None:
        print(f"Forma del dataset: {agent.current_dataframe.shape}")
        print(f"Columnas: {agent.current_dataframe.columns.tolist()}")
        print(f"Memoria usada: {agent.current_dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"Dataset info: {agent.current_dataset_info is not None}")
    print(f"Último análisis: {agent.last_analysis_result is not None}")
    
    sessions = agent.get_available_sessions()
    print(f"Sesiones activas: {len(sessions)}")
    for session in sessions:
        history = agent.get_session_history(session)
        print(f"  - {session}: {len(history)} mensajes")

# Uso
debug_agent_state(agent)
```

### 📋 Checklist de Verificación

#### **Antes de usar el agente:**
- [ ] ✅ Variable OPENAI_API_KEY configurada
- [ ] ✅ Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] ✅ Archivo de datos preparado y validado
- [ ] ✅ Nombres de columnas sin espacios ni caracteres especiales
- [ ] ✅ Datos en formato UTF-8 o encoding conocido

#### **Durante el análisis:**
- [ ] ✅ Sesiones organizadas por propósito
- [ ] ✅ Consultas específicas y claras
- [ ] ✅ Aprovechamiento de contexto conversacional
- [ ] ✅ Verificación de resultados intermedios
- [ ] ✅ Monitoreo de memoria y performance

#### **Después del análisis:**
- [ ] ✅ Resultados validados manualmente (sample)
- [ ] ✅ Sesiones innecesarias limpiadas
- [ ] ✅ Insights documentados para uso futuro
- [ ] ✅ Datos sensibles removidos de memoria

### 🆘 Soporte y Recursos Adicionales

**Logs de sistema:**
```bash
# Ubicación de logs
data/logs/YYYY-MM-DD/
├── conversation_logs/
│   ├── session_id_TIMESTAMP.json
│   └── ...
└── system_logs/
    ├── agent_errors.log
    └── performance.log
```

**Comandos de diagnóstico:**
```python
# Test de conectividad con OpenAI
import openai
try:
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    print("✅ Conexión OpenAI OK")
except Exception as e:
    print(f"❌ Error OpenAI: {e}")

# Test de pandas
try:
    import pandas as pd
    df = pd.DataFrame({'test': [1, 2, 3]})
    print("✅ Pandas OK")
except Exception as e:
    print(f"❌ Error Pandas: {e}")
```
