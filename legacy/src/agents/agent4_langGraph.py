import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from agents.base_agent import BaseAgent

# Definir la memoria del agente
class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

class SophisticatedAgent(BaseAgent):
    def __init__(self):
        self.llm = None
        self.app = None
    
    def initialize(self):
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError('No se encontró OPENAI_API_KEY')
        
        self.llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, openai_api_key=api_key)
        self._build_workflow()
        print('SophisticatedAgent inicializado correctamente')
    
    def _build_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node('classification_node', self._classification_node)
        workflow.add_node('entity_extraction', self._entity_extraction_node)
        workflow.add_node('summarization', self._summarization_node)
        workflow.set_entry_point('classification_node')
        workflow.add_edge('classification_node', 'entity_extraction')
        workflow.add_edge('entity_extraction', 'summarization')
        workflow.add_edge('summarization', END)
        self.app = workflow.compile()
    
    def _classification_node(self, state: State):
        prompt = PromptTemplate(
            input_variables=['text'],
            template='Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText:{text}\n\nCategory:'
        )
        message = HumanMessage(content=prompt.format(text=state['text']))
        classification = self.llm.invoke([message]).content.strip()
        return {'classification': classification}
    
    def _entity_extraction_node(self, state: State):
        prompt = PromptTemplate(
            input_variables=['text'],
            template='Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:'
        )
        message = HumanMessage(content=prompt.format(text=state['text']))
        entities_str = self.llm.invoke([message]).content.strip()
        entities = [entity.strip() for entity in entities_str.split(', ') if entity.strip()]
        return {'entities': entities}
    
    def _summarization_node(self, state: State):
        prompt = PromptTemplate(
            input_variables=['text'],
            template='Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:'
        )
        message = HumanMessage(content=prompt.format(text=state['text']))
        summary = self.llm.invoke([message]).content.strip()
        return {'summary': summary}
    
    def respond(self, message: str) -> str:
        try:
            state_input = {'text': message}
            result = self.app.invoke(state_input)
            
            response = f'''Análisis completado:

 Clasificación: {result.get('classification', 'N/A')}

 Entidades: {', '.join(result.get('entities', []))}

 Resumen: {result.get('summary', 'N/A')}'''
            
            return response
            
        except Exception as e:
            return f'Error al procesar: {str(e)}'
    
    def analyze_text(self, text: str) -> dict:
        try:
            state_input = {'text': text}
            result = self.app.invoke(state_input)
            return result
        except Exception as e:
            return {'error': str(e)}

def run_sophisticated_agent(text: str) -> dict:
    agent = SophisticatedAgent()
    agent.initialize()
    return agent.analyze_text(text)
