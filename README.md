# AI Agents Project with LangChain

## Overview
The AI Agents Project is designed to help you learn how to develop artificial intelligence agents from scratch using LangChain. This project provides a structured approach to building and testing AI chat agents that interact with Large Language Models (LLMs) like OpenAI's GPT models.

## Project Structure
```
ai-agents-project
├── src
│   ├── main.py                        # Entry point of the application
│   ├── agents
│   │   ├── base_agent.py              # Abstract base class for AI agents
│   │   ├── langchain_chat_agent.py    # LangChain-based chat agent
│   │   └── llm_chat_agent.py          # Legacy OpenAI direct API agent
│   ├── utils
│   │   └── helpers.py                 # Utility functions for various tasks
│   └── tests
│       └── test_agents.py             # Unit tests for agent classes
├── requirements.txt                   # Project dependencies (LangChain, OpenAI, etc.)
├── .env                              # Environment variables (API keys)
├── .gitignore                        # Files and directories to ignore by Git
└── README.md                         # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd ai-agents-project
   ```
3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Goals
- Understand the fundamentals of AI agent design using LangChain.
- Implement chat agents that interact with Large Language Models.
- Learn how to structure a Python project for AI development.
- Practice using environment variables for secure API key management.
- Write unit tests to ensure the functionality of your agents.

## Features
- **LangChain Integration**: Modern framework for building LLM applications
- **OpenAI GPT Integration**: Chat with GPT-3.5-turbo or GPT-4 models
- **Modular Architecture**: Easy to extend with new agent types
- **Environment Security**: API keys stored securely in `.env` file
- **Interactive Console Chat**: Real-time conversation with the AI agent

## Usage
To run the chat agent, execute the following command:
```
python src/main.py
```

The agent will start and you can begin chatting:
```
¡Hola! Soy tu asistente de chat con LangChain.
Escribe 'salir' para terminar la conversación.

Tú: Hola, ¿cómo estás?
Agente: ¡Hola! Estoy muy bien, gracias por preguntar. ¿En qué puedo ayudarte hoy?

Tú: salir
Agente: ¡Hasta luego! Que tengas un buen día.
```

## Dependencies
- `langchain>=0.1.0` - Framework for developing applications with LLMs
- `langchain-openai>=0.1.0` - OpenAI integration for LangChain
- `python-dotenv` - Load environment variables from .env file

## Learning Path
This project is structured to provide a progressive learning experience:

1. **Basic Understanding**: Start with the `BaseAgent` abstract class to understand agent architecture
2. **LangChain Implementation**: Explore `LangChainChatAgent` to see modern LLM integration
3. **Environment Management**: Learn secure API key handling with `.env` files
4. **Testing**: Write and run tests to ensure your agents work correctly
5. **Extension**: Build upon the foundation to create more sophisticated agents

## Next Steps
Once you're comfortable with the basics, consider exploring:
- Adding conversation memory to remember chat history
- Implementing different agent personalities with system messages
- Creating agents with tools and function calling capabilities
- Building multi-agent systems that can collaborate

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.