# Agent3: Data Analysis Agent

## Descripción General

El `DataAnalysisAgent` es un agente inteligente especializado en análisis de datos que combina tres tecnologías clave:

1. **Memoria Conversacional** (inspirado en `agent1_context_awareness`)
2. **Estructura LangGraph por Nodos** (inspirado en `agent0_langGraph`) 
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

## Casos de Uso

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

## Limitaciones

- Requiere API key de OpenAI
- Limitado a análisis que pandas puede realizar
- No genera visualizaciones automáticamente
- Memoria de sesión no persiste entre reinicios

## Extensiones Futuras

- [ ] Soporte para bases de datos SQL
- [ ] Generación automática de gráficos
- [ ] Persistencia de memoria en disco
- [ ] Integración con Jupyter Notebooks
- [ ] Exportación de resultados a PDF/Excel
- [ ] Análisis de texto y NLP
- [ ] Machine Learning básico

## Troubleshooting

### Error: OPENAI_API_KEY no encontrada
- Verificar archivo `.env` en el directorio raíz
- Confirmar que la variable está correctamente configurada

### Error: Import pandas could not be resolved
- Instalar dependencias: `pip install -r requirements.txt`
- Verificar que pandas esté instalado correctamente

### Error: No se pudo crear el agente de análisis
- Verificar que el dataset esté cargado correctamente
- Revisar permisos de archivo si se carga desde CSV/Excel

### Respuestas pobres o incorrectas
- Verificar que la consulta esté bien formulada
- Usar contexto más específico en las preguntas
- Revisar que el dataset tenga datos relevantes
