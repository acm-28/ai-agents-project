"""
Tests para el PandasAgent.
"""

import pytest
import pytest_asyncio
import pandas as pd
import asyncio
from pathlib import Path
import tempfile
import os

from ai_agents.agents.data_analysis import PandasAgent
from ai_agents.core.types import AgentResponse


class TestPandasAgent:
    """Test suite para PandasAgent."""
    
    @pytest_asyncio.fixture
    async def agent(self):
        """Fixture del agente."""
        agent = PandasAgent(agent_id="test_pandas_agent_fixed_id")
        await agent.initialize()
        return agent
    
    @pytest.fixture
    def sample_csv(self):
        """Fixture que crea un CSV de muestra."""
        data = {
            'nombre': ['Ana', 'Juan', 'María', 'Carlos', 'Lucía'],
            'edad': [25, 30, 35, 28, 32],
            'salario': [50000, 60000, 70000, 55000, 65000],
            'departamento': ['Ventas', 'IT', 'HR', 'Ventas', 'IT']
        }
        df = pd.DataFrame(data)
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        # Limpiar archivo temporal
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test inicialización del agente."""
        assert agent is not None
        assert agent.current_dataframe is None
        assert agent.current_dataset_info is None
        assert agent.last_analysis_result is None
    
    @pytest.mark.asyncio
    async def test_load_sample_data(self, agent):
        """Test carga de datos de muestra."""
        response = await agent.process({
            "message": "crear datos de muestra", 
            "sample_data": "sales"
        })
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get("data_loaded") is True
        assert agent.current_dataframe is not None
        assert len(agent.current_dataframe) > 0
        assert agent.current_dataset_info is not None
    
    @pytest.mark.asyncio
    async def test_load_csv_file(self, agent, sample_csv):
        """Test carga de archivo CSV."""
        response = await agent.process({
            "message": "cargar datos",
            "file_path": sample_csv
        })
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get("data_loaded") is True
        assert agent.current_dataframe is not None
        assert len(agent.current_dataframe) == 5
        assert 'edad' in agent.current_dataframe.columns
        assert 'salario' in agent.current_dataframe.columns
    
    @pytest.mark.asyncio
    async def test_data_analysis(self, agent):
        """Test análisis de datos."""
        # Primero cargar datos
        await agent.process({"message": "crear datos de muestra", "sample_data": "sales"})
        
        # Realizar análisis
        response = await agent.process("analizar datos")
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get("analysis_completed") is True
        assert agent.last_analysis_result is not None
        assert "Análisis completado" in response.content
    
    @pytest.mark.asyncio
    async def test_data_profiling(self, agent):
        """Test perfilado de datos."""
        # Cargar datos primero
        await agent.process({"message": "crear datos de muestra", "sample_data": "customers"})
        
        # Generar perfil
        response = await agent.process("generar perfil de datos")
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get("profile_generated") is True
        assert "Perfil del Dataset" in response.content
        assert "Información Básica" in response.content
    
    @pytest.mark.asyncio
    async def test_visualization_creation(self, agent):
        """Test creación de visualizaciones."""
        # Cargar datos primero
        await agent.process({"message": "crear datos de muestra", "sample_data": "financial"})
        
        # Crear histograma
        response = await agent.process({"message": "crear histograma", "viz_type": "histogram"})
        
        assert isinstance(response, AgentResponse)
        # Debería crear visualización, indicar que necesita una columna específica, 
        # o indicar que matplotlib no está disponible
        assert (response.metadata.get("visualization_created") is True or 
                response.metadata.get("needs_column") is True or
                response.metadata.get("error") is True and "matplotlib" in response.content)
    
    @pytest.mark.asyncio
    async def test_error_handling_no_data(self, agent):
        """Test manejo de errores cuando no hay datos."""
        # Intentar analizar sin datos cargados
        response = await agent.process("analizar datos")
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get("needs_data") is True
        assert "cargar datos" in response.content.lower()
    
    @pytest.mark.asyncio
    async def test_help_response(self, agent):
        """Test respuesta de ayuda."""
        response = await agent.process("¿qué puedes hacer?")
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get("help_provided") is True
        assert "especializado en análisis de datos" in response.content
        assert "Cargar datos" in response.content
    
    @pytest.mark.asyncio
    async def test_capabilities(self, agent):
        """Test obtención de capacidades."""
        capabilities = agent.get_capabilities()
        
        assert isinstance(capabilities, dict)
        assert capabilities["agent_type"] == "data_analysis"
        assert "csv" in capabilities["supported_formats"]
        assert "descriptive" in capabilities["analysis_types"]
        assert "histogram" in capabilities["visualization_types"]
    
    @pytest.mark.asyncio
    async def test_request_classification(self, agent):
        """Test clasificación de solicitudes."""
        # Test diferentes tipos de solicitudes
        assert agent._classify_request("cargar archivo.csv") == "load_data"
        assert agent._classify_request("analizar estadísticas") == "analyze_data"
        assert agent._classify_request("crear gráfico") == "visualize_data"
        assert agent._classify_request("generar perfil") == "profile_data"
        assert agent._classify_request("exportar reporte") == "export_results"
        assert agent._classify_request("¿qué puedes hacer?") == "general"
    
    @pytest.mark.asyncio
    async def test_workflow_integration(self, agent):
        """Test integración completa del workflow."""
        # 1. Cargar datos
        load_response = await agent.process({"message": "crear datos de muestra", "sample_data": "sales"})
        assert load_response.metadata.get("data_loaded") is True
        
        # 2. Analizar datos
        analysis_response = await agent.process("analizar datos")
        assert analysis_response.metadata.get("analysis_completed") is True
        
        # 3. Generar perfil
        profile_response = await agent.process("perfil de datos")
        assert profile_response.metadata.get("profile_generated") is True
        
        # 4. Verificar que tenemos resultados
        assert agent.current_dataframe is not None
        assert agent.last_analysis_result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, agent):
        """Test operaciones concurrentes."""
        # Cargar datos primero
        await agent.process({"message": "crear datos de muestra", "sample_data": "sales"})
        
        # Ejecutar múltiples operaciones concurrentemente
        tasks = [
            agent.process("perfil de datos"),
            agent.process("analizar datos"),
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # Verificar que todas las respuestas son válidas
        for response in responses:
            assert isinstance(response, AgentResponse)
            assert not response.metadata.get("error", False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
