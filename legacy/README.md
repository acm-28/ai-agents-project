# 🗄️ Legacy Code

Este directorio contiene código experimental y versiones anteriores del desarrollo del Framework AI Agents.

## ⚠️ Advertencia

El código en este directorio es **LEGACY** y no debe usarse en producción. Se mantiene únicamente para:

- Referencias históricas
- Recuperación de funcionalidades experimentales
- Comparación con versiones actuales
- Debugging de versiones anteriores

## 📁 Contenido

```
legacy/
├── src/                    # Código fuente experimental
│   ├── agents/            # Versiones experimentales de agentes
│   │   ├── agent1_context_awareness.py
│   │   ├── agent2_qa.py
│   │   ├── agent3_data_analysis.py
│   │   ├── agent4_langGraph.py
│   │   ├── base_agent.py
│   │   └── llm_chat_agent.py
│   ├── tests/             # Tests experimentales
│   ├── utils/             # Utilidades experimentales
│   ├── demo_agent3.py     # Demo experimental
│   └── main.py            # Main experimental
└── README.md              # Este archivo
```

## 🔄 Migración Realizada

Los siguientes componentes fueron migrados del código legacy al framework oficial:

### ✅ Agentes Migrados
- **agent1_context_awareness.py** → `ai_agents/agents/workflows/sophisticated_agent.py`
- **agent2_qa.py** → `ai_agents/agents/qa/memory_qa_agent.py`
- **agent3_data_analysis.py** → `ai_agents/agents/data_analysis/pandas_agent.py`
- **llm_chat_agent.py** → `ai_agents/agents/chat/llm_agent.py`

### ✅ Utilidades Migradas
- **helpers.py** → `ai_agents/utils/helpers.py`
- **base_agent.py** → `ai_agents/core/base_agent.py`

### ✅ Tests Migrados
- Tests experimentales → `tests/unit/` (estructura oficial)

## 🚫 No Usar Para

- ❌ Desarrollo nuevo
- ❌ Producción
- ❌ Referencias de API actual
- ❌ Documentación oficial

## ✅ Usar Para

- ✅ Referencias históricas
- ✅ Comparación de implementaciones
- ✅ Recuperación de features experimentales
- ✅ Debugging de versiones anteriores

## 🎯 Framework Oficial

Para desarrollo actual, usar:
```
ai_agents/          # Framework oficial y actualizado
├── agents/        # Agentes oficiales
├── api/           # API REST
├── cli/           # CLI Framework
├── config/        # Configuración
├── core/          # Componentes core
└── utils/         # Utilidades
```

---

**Fecha de Archivado**: 2025-08-12  
**Razón**: Reorganización estructural y migración a framework oficial  
**Estado**: Solo lectura / Referencias históricas
