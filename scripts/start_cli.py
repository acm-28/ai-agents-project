#!/usr/bin/env python3
"""
Script simple para iniciar el CLI del framework AI Agents.
"""

import sys
import os

# Agregar el directorio raíz al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    try:
        from ai_agents.cli.main import cli
        cli()
    except ImportError as e:
        print(f"Error importing CLI: {e}")
        print("Make sure to install the package with: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"Error running CLI: {e}")
        sys.exit(1)
