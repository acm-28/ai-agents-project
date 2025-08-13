import os
import pandas as pd
import numpy as np
from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType

from agents.base_agent import BaseAgent


# Definir el estado del agente para LangGraph
class DataAnalysisState(TypedDict):
    query: str
    dataset_info: Dict[str, Any]
    analysis_result: str
    conversation_history: List[str]
    session_id: str
    final_response: str
    dataframe: Optional[pd.DataFrame]
    pandas_agent: Optional[Any]
    error_message: Optional[str]


class DataAnalysisAgent(BaseAgent):
    """
    Agente de análisis de datos que combina:
    - Memoria conversacional (como agent1_context_awareness)
    - Estructura LangGraph por nodos (como agent0_langGraph)
    - Capacidades de análisis de datos con pandas
    """
    
    def __init__(self):
        self.llm = None
        self.app = None
        self.store = {}  # Almacén de memoria por sesión
        self.current_dataframe = None
        self.pandas_agent = None
        self.system_message = """Eres un experto analista de datos. Tu trabajo es ayudar a los usuarios a analizar y entender sus datos de manera clara y útil. 
        Puedes realizar análisis estadísticos, crear resúmenes, identificar patrones y responder preguntas específicas sobre los datos.
        Siempre proporciona respuestas claras y contextualizadas."""

    def initialize(self):
        """Inicializa el agente con OpenAI y construye el workflow de LangGraph."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("No se encontró OPENAI_API_KEY en las variables de entorno.")

        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=api_key
        )

        # Crear dataset de ejemplo si no hay uno cargado
        self._create_sample_dataset()
        self._build_workflow()
        
        print("DataAnalysisAgent con memoria y LangGraph inicializado correctamente.")

    def _create_sample_dataset(self):
        """Crea un dataset de ejemplo basado en la guía (datos de venta de autos)."""
        np.random.seed(42)
        n_rows = 1000

        # Generar fechas
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_rows)]

        # Definir categorías de datos
        makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Kia']
        models = ['Sedan', 'SUV', 'Truck', 'Hatchback', 'Coupe', 'Van']
        colors = ['Red', 'Blue', 'Black', 'White', 'Silver', 'Gray', 'Green']

        # Crear el dataset
        data = {
            'Date': dates,
            'Make': np.random.choice(makes, n_rows),
            'Model': np.random.choice(models, n_rows),
            'Color': np.random.choice(colors, n_rows),
            'Year': np.random.randint(2015, 2023, n_rows),
            'Price': np.random.uniform(20000, 80000, n_rows).round(2),
            'Mileage': np.random.uniform(0, 100000, n_rows).round(0),
            'EngineSize': np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], n_rows),
            'FuelEfficiency': np.random.uniform(20, 40, n_rows).round(1),
            'SalesPerson': np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eva'], n_rows)
        }

        # Crear DataFrame y ordenar por fecha
        self.current_dataframe = pd.DataFrame(data).sort_values('Date')
        
        # Crear el agente pandas
        self.pandas_agent = create_pandas_dataframe_agent(
            self.llm,
            self.current_dataframe,
            verbose=False,
            allow_dangerous_code=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

    def _build_workflow(self):
        """Construye el workflow de LangGraph con nodos especializados."""
        workflow = StateGraph(DataAnalysisState)
        
        # Agregar nodos
        workflow.add_node("input_processing", self._input_processing_node)
        workflow.add_node("data_preparation", self._data_preparation_node)
        workflow.add_node("analysis_execution", self._analysis_execution_node)
        workflow.add_node("result_formatting", self._result_formatting_node)
        workflow.add_node("memory_update", self._memory_update_node)
        
        # Definir el flujo
        workflow.set_entry_point("input_processing")
        workflow.add_edge("input_processing", "data_preparation")
        workflow.add_edge("data_preparation", "analysis_execution")
        workflow.add_edge("analysis_execution", "result_formatting")
        workflow.add_edge("result_formatting", "memory_update")
        workflow.add_edge("memory_update", END)
        
        self.app = workflow.compile()

    def _input_processing_node(self, state: DataAnalysisState):
        """Procesa la consulta inicial y extrae la intención del usuario."""
        query = state['query']
        session_id = state['session_id']
        
        # Obtener historial de la sesión para contexto
        history = self.get_session_history(session_id)
        context_messages = []
        
        if history.messages:
            context_messages = [f"Contexto previo: {msg.content}" for msg in history.messages[-4:]]
        
        # Analizar la intención de la consulta
        intent_prompt = PromptTemplate(
            input_variables=['query', 'context'],
            template="""Analiza la siguiente consulta sobre datos y determina qué tipo de análisis se solicita.
            
Contexto previo: {context}
Consulta: {query}

Clasifica la consulta en una de estas categorías:
- descriptive: Estadísticas descriptivas, resúmenes, información básica
- comparative: Comparaciones entre grupos, correlaciones
- temporal: Análisis de tendencias temporales
- filtering: Filtrado de datos específicos
- aggregation: Agrupaciones y agregaciones
- visualization: Solicitudes de gráficos o visualizaciones
- other: Otros tipos de análisis

Responde solo con la categoría."""
        )
        
        message = HumanMessage(content=intent_prompt.format(
            query=query, 
            context=" | ".join(context_messages) if context_messages else "Sin contexto previo"
        ))
        
        intent = self.llm.invoke([message]).content.strip().lower()
        
        return {
            "query": query,
            "dataset_info": {"intent": intent, "context": context_messages}
        }

    def _data_preparation_node(self, state: DataAnalysisState):
        """Prepara y valida el dataset para el análisis."""
        dataset_info = state["dataset_info"].copy()
        
        if self.current_dataframe is not None:
            # Obtener información básica del dataset
            dataset_info.update({
                "shape": self.current_dataframe.shape,
                "columns": list(self.current_dataframe.columns),
                "dtypes": dict(self.current_dataframe.dtypes.astype(str)),
                "null_counts": dict(self.current_dataframe.isnull().sum()),
                "memory_usage": f"{self.current_dataframe.memory_usage(deep=True).sum() / 1024:.2f} KB"
            })
            
            return {
                "dataset_info": dataset_info,
                "dataframe": self.current_dataframe,
                "pandas_agent": self.pandas_agent
            }
        else:
            return {
                "dataset_info": dataset_info,
                "error_message": "No hay dataset cargado para analizar."
            }

    def _analysis_execution_node(self, state: DataAnalysisState):
        """Ejecuta el análisis usando el pandas agent."""
        query = state["query"]
        pandas_agent = state.get("pandas_agent")
        error_message = state.get("error_message")
        
        if error_message:
            return {"analysis_result": f"Error: {error_message}"}
        
        if not pandas_agent:
            return {"analysis_result": "Error: No se pudo crear el agente de análisis."}
        
        try:
            # Ejecutar la consulta usando el pandas agent
            result = pandas_agent.invoke({"input": query})
            
            # Extraer la respuesta del resultado
            if isinstance(result, dict) and "output" in result:
                analysis_result = result["output"]
            else:
                analysis_result = str(result)
            
            return {"analysis_result": analysis_result}
            
        except Exception as e:
            return {"analysis_result": f"Error al ejecutar el análisis: {str(e)}"}

    def _result_formatting_node(self, state: DataAnalysisState):
        """Formatea el resultado del análisis de manera comprensible."""
        query = state["query"]
        analysis_result = state["analysis_result"]
        dataset_info = state["dataset_info"]
        
        # Crear un prompt para formatear mejor la respuesta
        format_prompt = PromptTemplate(
            input_variables=['query', 'result', 'context'],
            template="""Formatea la siguiente respuesta de análisis de datos de manera clara y profesional.

Pregunta original: {query}
Resultado del análisis: {result}
Contexto del dataset: {context}

Proporciona una respuesta bien estructurada, clara y útil. Si hay números, formatéalos apropiadamente. 
Si es relevante, añade interpretaciones o insights adicionales."""
        )
        
        context_info = f"Dataset con {dataset_info.get('shape', ['?', '?'])[0]} filas y {dataset_info.get('shape', ['?', '?'])[1]} columnas"
        
        message = HumanMessage(content=format_prompt.format(
            query=query,
            result=analysis_result,
            context=context_info
        ))
        
        try:
            formatted_response = self.llm.invoke([message]).content.strip()
            return {"final_response": formatted_response}
        except Exception as e:
            return {"final_response": analysis_result}  # Fallback al resultado original

    def _memory_update_node(self, state: DataAnalysisState):
        """Actualiza la memoria con la consulta y respuesta actual."""
        session_id = state["session_id"]
        query = state["query"]
        final_response = state["final_response"]
        
        # Obtener historial de la sesión
        history = self.get_session_history(session_id)
        
        # Agregar la consulta y respuesta al historial
        history.add_user_message(query)
        history.add_ai_message(final_response)
        
        return {"conversation_history": [f"Q: {query}", f"A: {final_response}"]}

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """Obtiene el historial de una sesión o crea uno nuevo."""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def respond(self, message: str, session_id: str = "default") -> str:
        """Responde a una consulta de análisis de datos."""
        try:
            # Preparar el estado inicial
            initial_state = {
                "query": message,
                "session_id": session_id,
                "dataset_info": {},
                "analysis_result": "",
                "conversation_history": [],
                "final_response": "",
                "dataframe": None,
                "pandas_agent": None,
                "error_message": None
            }
            
            # Ejecutar el workflow
            result = self.app.invoke(initial_state)
            
            return result.get("final_response", "No se pudo procesar la consulta.")
            
        except Exception as e:
            return f"Error al procesar la consulta: {str(e)}"

    def load_dataset(self, file_path: str = None, dataframe: pd.DataFrame = None) -> str:
        """Carga un nuevo dataset desde archivo o DataFrame."""
        try:
            if dataframe is not None:
                self.current_dataframe = dataframe
            elif file_path:
                if file_path.endswith('.csv'):
                    self.current_dataframe = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    self.current_dataframe = pd.read_excel(file_path)
                else:
                    return "Formato de archivo no soportado. Use CSV o Excel."
            else:
                return "Debe proporcionar un archivo o DataFrame."
            
            # Recrear el pandas agent con el nuevo dataset
            self.pandas_agent = create_pandas_dataframe_agent(
                self.llm,
                self.current_dataframe,
                verbose=False,
                allow_dangerous_code=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
            )
            
            return f"Dataset cargado exitosamente. Forma: {self.current_dataframe.shape}"
            
        except Exception as e:
            return f"Error al cargar dataset: {str(e)}"

    def get_dataset_info(self) -> Dict[str, Any]:
        """Obtiene información del dataset actual."""
        if self.current_dataframe is None:
            return {"error": "No hay dataset cargado"}
        
        return {
            "shape": self.current_dataframe.shape,
            "columns": list(self.current_dataframe.columns),
            "dtypes": dict(self.current_dataframe.dtypes.astype(str)),
            "null_counts": dict(self.current_dataframe.isnull().sum()),
            "sample_data": self.current_dataframe.head().to_dict(),
            "memory_usage": f"{self.current_dataframe.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }

    def clear_session_memory(self, session_id: str = "default") -> str:
        """Limpia la memoria de una sesión específica."""
        if session_id in self.store:
            del self.store[session_id]
            return f"Memoria de la sesión '{session_id}' limpiada."
        return f"No se encontró la sesión '{session_id}'."

    def get_available_sessions(self) -> List[str]:
        """Obtiene la lista de sesiones activas."""
        return list(self.store.keys())
