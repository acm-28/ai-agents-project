import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Agregar el directorio src al path para importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.agent3_data_analysis import DataAnalysisAgent


class TestDataAnalysisAgent(unittest.TestCase):
    """Tests para el DataAnalysisAgent."""
    
    def setUp(self):
        """Configuración inicial para cada test."""
        self.agent = DataAnalysisAgent()
        
    def test_initialization(self):
        """Test de inicialización del agente."""
        try:
            self.agent.initialize()
            self.assertIsNotNone(self.agent.llm)
            self.assertIsNotNone(self.agent.app)
            self.assertIsNotNone(self.agent.current_dataframe)
            self.assertIsNotNone(self.agent.pandas_agent)
            print("✓ Test de inicialización pasado")
        except Exception as e:
            self.fail(f"Error en la inicialización: {e}")
    
    def test_sample_dataset_creation(self):
        """Test de creación del dataset de ejemplo."""
        self.agent.initialize()
        
        # Verificar que el dataset tiene la estructura esperada
        df = self.agent.current_dataframe
        self.assertEqual(df.shape[0], 1000)  # 1000 filas
        self.assertEqual(df.shape[1], 10)    # 10 columnas
        
        expected_columns = ['Date', 'Make', 'Model', 'Color', 'Year', 'Price', 
                          'Mileage', 'EngineSize', 'FuelEfficiency', 'SalesPerson']
        self.assertEqual(list(df.columns), expected_columns)
        print("✓ Test de dataset de ejemplo pasado")
    
    def test_basic_queries(self):
        """Test de consultas básicas."""
        self.agent.initialize()
        
        # Test consultas simples
        test_queries = [
            "¿Cuántas filas tiene el dataset?",
            "¿Cuáles son las columnas del dataset?",
            "¿Cuál es el precio promedio de los autos?"
        ]
        
        for query in test_queries:
            response = self.agent.respond(query, "test_session")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            self.assertNotIn("Error", response)
        
        print("✓ Test de consultas básicas pasado")
    
    def test_memory_functionality(self):
        """Test de funcionalidad de memoria."""
        self.agent.initialize()
        
        session_id = "memory_test"
        
        # Primera consulta
        response1 = self.agent.respond("¿Cuántas filas tiene el dataset?", session_id)
        
        # Segunda consulta que debería recordar la anterior
        response2 = self.agent.respond("¿Y cuántas columnas?", session_id)
        
        # Verificar que ambas respuestas son válidas
        self.assertIsInstance(response1, str)
        self.assertIsInstance(response2, str)
        self.assertGreater(len(response1), 0)
        self.assertGreater(len(response2), 0)
        
        # Verificar que la sesión existe en el store
        self.assertIn(session_id, self.agent.store)
        
        print("✓ Test de memoria pasado")
    
    def test_dataset_info(self):
        """Test de información del dataset."""
        self.agent.initialize()
        
        info = self.agent.get_dataset_info()
        
        self.assertIn("shape", info)
        self.assertIn("columns", info)
        self.assertIn("dtypes", info)
        self.assertIn("null_counts", info)
        self.assertIn("sample_data", info)
        
        print("✓ Test de información del dataset pasado")
    
    def test_custom_dataset_loading(self):
        """Test de carga de dataset personalizado."""
        self.agent.initialize()
        
        # Crear un DataFrame de prueba
        test_data = {
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        }
        test_df = pd.DataFrame(test_data)
        
        # Cargar el dataset personalizado
        result = self.agent.load_dataset(dataframe=test_df)
        
        self.assertIn("exitosamente", result)
        self.assertEqual(self.agent.current_dataframe.shape, (5, 3))
        
        print("✓ Test de carga de dataset personalizado pasado")
    
    def test_session_management(self):
        """Test de gestión de sesiones."""
        self.agent.initialize()
        
        # Crear múltiples sesiones
        sessions = ["session1", "session2", "session3"]
        
        for session in sessions:
            self.agent.respond("Test query", session)
        
        # Verificar que las sesiones existen
        available_sessions = self.agent.get_available_sessions()
        for session in sessions:
            self.assertIn(session, available_sessions)
        
        # Limpiar una sesión
        result = self.agent.clear_session_memory("session1")
        self.assertIn("limpiada", result)
        self.assertNotIn("session1", self.agent.get_available_sessions())
        
        print("✓ Test de gestión de sesiones pasado")
    
    def test_error_handling(self):
        """Test de manejo de errores."""
        self.agent.initialize()
        
        # Test con consulta problemática
        response = self.agent.respond("", "test_session")
        self.assertIsInstance(response, str)
        
        # Test con dataset inexistente
        result = self.agent.load_dataset("archivo_inexistente.csv")
        self.assertIn("Error", result)
        
        print("✓ Test de manejo de errores pasado")

def run_tests():
    """Ejecuta todos los tests."""
    print("Iniciando tests del DataAnalysisAgent...")
    print("=" * 50)
    
    # Verificar variables de entorno
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ADVERTENCIA: OPENAI_API_KEY no encontrada. Algunos tests pueden fallar.")
        return
    
    # Crear suite de tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataAnalysisAgent)
    runner = unittest.TextTestRunner(verbosity=0)
    
    # Ejecutar tests
    result = runner.run(suite)
    
    print("=" * 50)
    if result.wasSuccessful():
        print("✅ Todos los tests pasaron exitosamente!")
    else:
        print(f"❌ {len(result.failures)} tests fallaron, {len(result.errors)} errores")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
