#!/usr/bin/env python3
"""
Script simple para iniciar el servidor API del framework AI Agents.
"""

import sys
import os

# Agregar el directorio raíz al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    try:
        import uvicorn
        from ai_agents.api.main import app
        
        print("🚀 Starting AI Agents API Server...")
        print("📖 API Documentation available at: http://localhost:8000/docs")
        print("🔧 Health check at: http://localhost:8000/health")
        
        uvicorn.run(
            "ai_agents.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except ImportError as e:
        print(f"Error importing API dependencies: {e}")
        print("Make sure to install the package with: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting API server: {e}")
        sys.exit(1)
