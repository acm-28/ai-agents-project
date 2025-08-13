"""
Tests para SophisticatedAgent - Agente de workflows complejos con LangGraph.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from ai_agents.agents.workflows.sophisticated_agent import (
    SophisticatedAgent, 
    AnalysisResult,
    TextAnalysisState,
    HAS_LANGGRAPH
)
from ai_agents.core.types import AgentResponse, AgentState


class TestSophisticatedAgent:
    """Test suite para SophisticatedAgent."""

    @pytest_asyncio.fixture
    async def agent(self):
        """Fixture del agente."""
        agent = SophisticatedAgent()
        
        # Mock LLM para evitar llamadas reales a OpenAI
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_llm.ainvoke.return_value = mock_response
        
        with patch('ai_agents.agents.workflows.sophisticated_agent.ChatOpenAI') as mock_openai:
            mock_openai.return_value = mock_llm
            # Usar _safe_initialize en lugar de initialize directamente
            await agent._safe_initialize()
            agent.llm = mock_llm
        
        return agent

    @pytest.fixture
    def sample_texts(self):
        """Fixture con textos de muestra para análisis."""
        return {
            'news': """Breaking News: The tech giant Apple Inc. announced today that they will be releasing a new iPhone model next month. The announcement was made by CEO Tim Cook during a press conference in Cupertino, California. This new model is expected to feature enhanced AI capabilities and improved battery life.""",
            
            'blog': """I've been thinking a lot lately about productivity and how we can optimize our daily routines. In my experience, the key to being productive isn't about working harder, but working smarter. Here are some tips that have worked for me over the years.""",
            
            'research': """This study examines the effects of machine learning algorithms on predictive accuracy in financial markets. We analyzed data from 500 companies over a 10-year period using various statistical models. Our findings suggest that ensemble methods outperform traditional regression models by 15%.""",
            
            'business': """Q3 Financial Report: XYZ Corporation is pleased to announce record-breaking quarterly earnings of $2.4 billion, representing a 12% increase over the same period last year. Revenue growth was driven primarily by our cloud services division and international expansion."""
        }

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test inicialización del agente."""
        assert agent.state == AgentState.READY
        assert agent.is_initialized is True
        assert agent.llm is not None
        
        if HAS_LANGGRAPH:
            assert agent.workflow_app is not None
        
        # Verificar prompts
        assert 'classification' in agent.prompts
        assert 'entities' in agent.prompts
        assert 'summary' in agent.prompts

    @pytest.mark.asyncio
    async def test_text_analysis_news(self, agent, sample_texts):
        """Test análisis de texto tipo noticias."""
        # Mock respuestas del LLM
        agent.llm.ainvoke.side_effect = [
            MagicMock(content="News"),
            MagicMock(content="Apple Inc., Tim Cook, Cupertino, California, iPhone"),
            MagicMock(content="Apple announced a new iPhone model with enhanced AI capabilities and improved battery life.")
        ]
        
        response = await agent.process({
            "message": "analizar texto",
            "text": sample_texts['news']
        })
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get("analysis_complete") is True
        assert "clasificación" in response.content.lower() or "classification" in response.content.lower()
        
        if HAS_LANGGRAPH:
            assert response.metadata.get("classification") == "News"
            assert response.metadata.get("entity_count") >= 0

    @pytest.mark.asyncio 
    async def test_text_analysis_blog(self, agent, sample_texts):
        """Test análisis de texto tipo blog."""
        agent.llm.ainvoke.side_effect = [
            MagicMock(content="Blog"),
            MagicMock(content="No specific entities found"),
            MagicMock(content="Tips for improving productivity by working smarter, not harder.")
        ]
        
        response = await agent.process({
            "message": "analizar",
            "text": sample_texts['blog']
        })
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get("analysis_complete") is True
        
        if HAS_LANGGRAPH:
            assert response.metadata.get("classification") == "Blog"

    @pytest.mark.asyncio
    async def test_text_analysis_research(self, agent, sample_texts):
        """Test análisis de texto de investigación."""
        agent.llm.ainvoke.side_effect = [
            MagicMock(content="Research"),
            MagicMock(content="machine learning, financial markets, ensemble methods"),
            MagicMock(content="Study shows ensemble ML methods outperform traditional regression by 15% in financial predictions.")
        ]
        
        response = await agent.process({
            "message": "analiza este paper",
            "text": sample_texts['research']
        })
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get("analysis_complete") is True
        
        if HAS_LANGGRAPH:
            assert response.metadata.get("classification") == "Research"
            assert response.metadata.get("entity_count") >= 2

    @pytest.mark.asyncio
    async def test_text_analysis_business(self, agent, sample_texts):
        """Test análisis de texto de negocios."""
        agent.llm.ainvoke.side_effect = [
            MagicMock(content="Business"),
            MagicMock(content="XYZ Corporation, Q3, cloud services"),
            MagicMock(content="XYZ Corporation reports record Q3 earnings of $2.4B, up 12% driven by cloud services.")
        ]
        
        response = await agent.process({
            "message": "procesar reporte",
            "text": sample_texts['business']
        })
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get("analysis_complete") is True
        
        if HAS_LANGGRAPH:
            assert response.metadata.get("classification") == "Business"

    @pytest.mark.asyncio
    async def test_empty_text_handling(self, agent):
        """Test manejo de texto vacío."""
        response = await agent.process({"message": ""})
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get("error") is True
        assert response.metadata.get("needs_text") is True
        assert "proporciona un texto" in response.content.lower()

    @pytest.mark.asyncio
    async def test_error_handling_llm_failure(self, agent, sample_texts):
        """Test manejo de errores del LLM."""
        # Simular error en LLM
        agent.llm.ainvoke.side_effect = Exception("LLM Error")
        
        response = await agent.process({
            "message": "analizar",
            "text": sample_texts['news']
        })
        
        assert isinstance(response, AgentResponse)
        assert response.metadata.get("error") is True
        assert "error" in response.content.lower()

    @pytest.mark.asyncio
    async def test_capabilities(self, agent):
        """Test obtención de capacidades."""
        capabilities = agent.get_capabilities()
        
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
        assert any("clasificación" in cap.lower() or "classification" in cap.lower() for cap in capabilities)
        assert any("entidades" in cap.lower() or "entities" in cap.lower() for cap in capabilities)
        assert any("resumen" in cap.lower() or "summary" in cap.lower() for cap in capabilities)

    @pytest.mark.asyncio
    async def test_confidence_analysis(self, agent, sample_texts):
        """Test análisis de confianza."""
        agent.llm.ainvoke.side_effect = [
            MagicMock(content="News"),
            MagicMock(content="Apple Inc., Tim Cook, iPhone"),
            MagicMock(content="Apple announces new iPhone with AI features.")
        ]
        
        response = await agent.process({
            "message": "analizar con confianza",
            "text": sample_texts['news']
        })
        
        assert isinstance(response, AgentResponse)
        
        if HAS_LANGGRAPH:
            confidence_scores = response.metadata.get("confidence_scores", {})
            assert isinstance(confidence_scores, dict)
            assert 'overall' in confidence_scores
            assert all(0 <= score <= 1 for score in confidence_scores.values())

    @pytest.mark.asyncio
    async def test_last_analysis_storage(self, agent, sample_texts):
        """Test almacenamiento del último análisis."""
        agent.llm.ainvoke.side_effect = [
            MagicMock(content="Blog"),
            MagicMock(content="productivity, optimization"),
            MagicMock(content="Tips for better productivity.")
        ]
        
        # Realizar análisis
        await agent.process({
            "message": "analizar",
            "text": sample_texts['blog']
        })
        
        # Verificar almacenamiento
        last_analysis = agent.get_last_analysis()
        
        if HAS_LANGGRAPH:
            assert last_analysis is not None
            assert isinstance(last_analysis, AnalysisResult)
            assert last_analysis.classification == "Blog"
            assert isinstance(last_analysis.entities, list)
            assert isinstance(last_analysis.summary, str)

    @pytest.mark.asyncio
    async def test_fallback_mode_without_langgraph(self, sample_texts):
        """Test modo de respaldo sin LangGraph."""
        # Crear agente sin LangGraph
        agent = SophisticatedAgent()
        
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Fallback analysis result"
        mock_llm.ainvoke.return_value = mock_response
        
        # Simular LangGraph no disponible
        with patch('ai_agents.agents.workflows.sophisticated_agent.HAS_LANGGRAPH', False):
            with patch('ai_agents.agents.workflows.sophisticated_agent.ChatOpenAI') as mock_openai:
                mock_openai.return_value = mock_llm
                await agent.initialize()
                agent.llm = mock_llm
                
                response = await agent.process({
                    "message": "analizar",
                    "text": sample_texts['news']
                })
                
                assert isinstance(response, AgentResponse)
                assert response.metadata.get("analysis_complete") is True
                assert response.metadata.get("fallback_mode") is True

    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, agent, sample_texts):
        """Test análisis concurrente."""
        # Configurar mocks para múltiples respuestas
        agent.llm.ainvoke.side_effect = [
            # Primera análisis
            MagicMock(content="News"),
            MagicMock(content="Apple Inc., Tim Cook"),
            MagicMock(content="Apple announces new iPhone."),
            # Segunda análisis
            MagicMock(content="Blog"),
            MagicMock(content="productivity"),
            MagicMock(content="Tips for productivity.")
        ]
        
        # Ejecutar análisis concurrente
        tasks = [
            agent.process({"message": "analizar", "text": sample_texts['news']}),
            agent.process({"message": "analizar", "text": sample_texts['blog']})
        ]
        
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 2
        assert all(isinstance(r, AgentResponse) for r in responses)
        assert all(r.metadata.get("analysis_complete") for r in responses)

    @pytest.mark.asyncio
    async def test_different_message_formats(self, agent, sample_texts):
        """Test diferentes formatos de mensaje."""
        agent.llm.ainvoke.side_effect = [
            MagicMock(content="News"),
            MagicMock(content="Apple Inc."),
            MagicMock(content="Apple news summary.")
        ]
        
        # Test con texto en el mensaje
        response1 = await agent.process({"message": sample_texts['news']})
        assert isinstance(response1, AgentResponse)
        
        # Test con texto separado
        response2 = await agent.process({
            "message": "analizar esto",
            "text": sample_texts['news']
        })
        assert isinstance(response2, AgentResponse)

    @pytest.mark.asyncio
    async def test_entity_filtering(self, agent):
        """Test filtrado de entidades."""
        # Mock respuesta con entidades inválidas
        agent.llm.ainvoke.side_effect = [
            MagicMock(content="News"),
            MagicMock(content="Apple Inc., none, N/A, Tim Cook, , invalid"),
            MagicMock(content="Test summary.")
        ]
        
        response = await agent.process({
            "message": "analizar",
            "text": "Apple Inc. CEO Tim Cook announced something."
        })
        
        assert isinstance(response, AgentResponse)
        
        if HAS_LANGGRAPH:
            # Verificar que las entidades inválidas fueron filtradas
            result = response.metadata.get("result", {})
            entities = result.get("entities", [])
            assert "none" not in [e.lower() for e in entities]
            assert "n/a" not in [e.lower() for e in entities]
            assert "" not in entities
