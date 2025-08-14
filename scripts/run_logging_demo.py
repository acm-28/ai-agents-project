#!/usr/bin/env python3
"""
Script para ejecutar el demo de logging de conversaciones.
"""

import sys
import os
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configurar variables de entorno
os.environ.setdefault("PYTHONPATH", str(project_root))

if __name__ == "__main__":
    from examples.conversation_logging_demo import main
    import asyncio
    
    print("🚀 Iniciando demo de logging de conversaciones...")
    print(f"📁 Proyecto: {project_root}")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrumpido por el usuario.")
    except Exception as e:
        print(f"\n❌ Error ejecutando demo: {e}")
        sys.exit(1)
