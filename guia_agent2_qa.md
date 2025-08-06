# Simple Question-Answering Agent Tutorial

# Overview
This tutorial introduces a basic Question-Answering (QA) agent using LangChain and OpenAI's language model. The agent is designed to understand user queries and provide relevant, concise answers.

# Motivation
In the era of AI-driven interactions, creating a simple QA agent serves as a fundamental stepping stone towards more complex AI systems. This project aims to:
- Demonstrate the basics of AI-driven question-answering
- Introduce key concepts in building AI agents
- Provide a foundation for more advanced agent architectures

# Key Components
- Language Model: Utilizes OpenAI's GPT model for natural language understanding and generation.
- Prompt Template: Defines the structure and context for the agent's responses.
- LLMChain: Combines the language model and prompt template for streamlined processing.

# Method Details
1. Setup and Initialization
- Import necessary libraries (LangChain, dotenv)
- Load environment variables for API key management
- Initialize the OpenAI language model

2. Defining the Prompt Template
- Create a template that instructs the AI on its role and response format
- Use the PromptTemplate class to structure the input

3. Creating the LLMChain
- Combine the language model and prompt template into an LLMChain
- This chain manages the flow from user input to AI response

4. Implementing the Question-Answering Function
- Define a function that takes a user question as input
- Use the LLMChain to process the question and generate an answer

5. User Interaction
- In a Jupyter notebook environment, provide cells for:
- Example usage with a predefined question
- Interactive input for user questions

# Conclusion
This Simple Question-Answering Agent serves as an entry point into the world of AI agents. - By understanding and implementing this basic model, you've laid the groundwork for more sophisticated systems. 

# Future enhancements could include:
- Adding memory to maintain context across multiple questions
- Integrating external knowledge bases for more informed responses
- Implementing more complex decision-making processes

As you progress through more advanced tutorials in this repository, you'll build upon these fundamental concepts to create increasingly capable and intelligent AI agents.

# Import necessary libraries
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)

# Define the prompt template
template = """
You are a helpful AI assistant. Your task is to answer the user's question to the best of your ability.

User's question: {question}

Please provide a clear and concise answer:
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Create the LLMChain
qa_chain = prompt | llm

# Define the get_answer function
def get_answer(question):
    """
    Get an answer to the given question using the QA chain.
    """
    input_variables = {"question": question}
    response = qa_chain.invoke(input_variables).content
    return response

# Example usage
question = "What is the capital of France?"
answer = get_answer(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
Question: What is the capital of France?
Answer: The capital of France is Paris.

# Interactive cell for user questions
user_question = input("Enter your question: ")
user_answer = get_answer(user_question)
print(f"Answer: {user_answer}")
Answer: I'm just a program, so I don't have feelings, but I'm here and ready to help you! How can I assist you today?