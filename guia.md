# AI Agents Project with LangChain

# Setting Up Our Environment
Before we dive into the code, let's set up our development environment properly. This will take just a few minutes and ensure everything runs smoothly.

Create a Virtual Environment First, open your terminal and create a new directory for this project:

bash

mkdir ai_agent_project cd ai_agent_project
Create and activate a virtual environment:

bash

# Windows 
python -m venv agent_env agent_env\Scripts\activate 
# macOS/Linux
python3 -m venv agent_env source agent_env/bin/activate
Install Required Packages With your virtual environment activated, install the necessary packages:

bash

pip install langgraph langchain langchain-openai python-dotenv
Set Up Your OpenAI API Key You'll need an OpenAI API key to use their models. Here's how to get one:

Go to https://platform.openai.com/signup

Create an account or log in

Navigate to the API Keys section

Click "Create new secret key"

Copy your API key

Now create a .env file in your project directory:

bash

# Windows
echo OPENAI_API_KEY=your-api-key-here > .env 
# macOS/Linux
echo "OPENAI_API_KEY=your-api-key-here" > .env
Replace 'your-api-key-here' with your actual OpenAI API key.

Create a Test File Let's make sure everything is working. Create a file named test_setup.py:

python

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# Load environment variables
load_dotenv() 
# Initialize the ChatOpenAI instance 
llm = ChatOpenAI(model="gpt-4o-mini") 
# Test the setup 
response = llm.invoke("Hello! Are you working?") print(response.content)
Run it to verify your setup:

bash
python test_setup.py
If you see a response, congratulations! Your environment is ready to go.

Now when everything is ready, let's get started with building our agent. First, we need to import the tools we'll be using:

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
Each of these imports plays a crucial role in our agent's functionality. The StateGraph class will provide the foundation for our agent's structure, while PromptTemplate and ChatOpenAI give us the tools to interact with AI models effectively.

Designing Our Agent's Memory
Just as human intelligence requires memory, our agent needs a way to keep track of information. We create this using a TypedDict:

class State(TypedDict): text:
str classification: 
str entities: List[str] summary: str
This state design is fascinating because it mirrors how humans process information. When we read a document, we maintain several pieces of information simultaneously: we remember the original text, we understand what kind of document it is, we note important names or concepts, and we form a concise understanding of its main points. Our state structure captures these same elements.

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
We then initialize the llm we want to use (in this case “gpt-4o-mini”, but you can use any llm that you want. if you work with Openai API you’ll need to create a private token on their website that will allow you to use it) with temperature =0. Temperature = 0 in LLMs means the model will always choose the most probable/likely token at each step, making outputs deterministic and consistent. This leads to more focused and precise responses, but potentially less creative ones compared to higher temperature settings which introduce more randomness in token selection.

Creating Our Agent's Core Capabilities
Now we'll create the actual skills our agent will use. Each of these capabilities is implemented as a function that performs a specific type of analysis.

First, let's create our classification capability:

def classification_node(state: State):
    ''' Classify the text into one of the categories: News, Blog, Research, or Other '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText:{text}\n\nCategory:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}
This function acts like a skilled librarian who can quickly determine what kind of document they're looking at. Notice how we use a prompt template to give clear, consistent instructions to our AI model. The function takes in our current state (which includes the text we're analyzing) and returns its classification.

Next, we create our entity extraction capability:

def entity_extraction_node(state: State):
    ''' Extract all the entities (Person, Organization, Location) from the text      '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}
This function is like a careful reader who identifies and remembers all the important names, organizations, and places mentioned in the text. It processes the text and returns a list of these key entities.

Finally, we implement our summarization capability:

def summarization_node(state: State):
    ''' Summarize the text in one short sentence '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}
This function acts like a skilled editor who can distill the essence of a document into a concise summary. It takes our text and creates a brief, informative summary of its main points.

Bringing It All Together
Now comes the most exciting part - connecting these capabilities into a coordinated system:

workflow = StateGraph(State)

# Add nodes to the graph
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

# Add edges to the graph
workflow.set_entry_point("classification_node") # Set the entry point of the graph
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

# Compile the graph
app = workflow.compile()
Congratulations we’ve just built an agent!

A reminder of how it looks:


This is where LangGraph's power shines. We're not just collecting different capabilities - we're creating a coordinated workflow that determines how these capabilities work together. Think of it as creating a production line for information processing, where each step builds on the results of the previous ones.

The structure we've created tells our agent to:

Start by understanding what kind of text it's dealing with

Then identify important entities within that text

Finally, create a summary that captures the main points

End the process once the summary is complete

Seeing Our Agent in Action
Now that we've built our agent, it's time to see how it performs with real-world text. This is where theory meets practice, and where we can truly understand the power of our graph-based approach. Let's test our agent with a concrete example:

sample_text = """
OpenAI has announced the GPT-4 model, which is a large multimodal model that exhibits human-level performance on various professional benchmarks. It is developed to improve the alignment and safety of AI systems.
additionally, the model is designed to be more efficient and scalable than its predecessor, GPT-3. The GPT-4 model is expected to be released in the coming months and will be available to the public for research and development purposes.
"""

state_input = {"text": sample_text}
result = app.invoke(state_input)

print("Classification:", result["classification"])
print("\nEntities:", result["entities"])
print("\nSummary:", result["summary"])
When we run this code, our agent processes the text through each of its capabilities, and we get the following output:

Classification: News

Entities: ['OpenAI', 'GPT-4', 'GPT-3']

Summary: OpenAI's upcoming GPT-4 model is a multimodal AI that aims for human-level performance, improved safety, and greater efficiency compared to GPT-3.

Let's break down what's happening here, as it beautifully demonstrates how our agent coordinates its different capabilities to understand the text comprehensively.

First, our classification node correctly identified this as a news article. This makes sense given the text's announcement-style format and focus on current developments. The agent recognized the hallmarks of news writing - timely information, factual presentation, and focus on a specific development.

Next, the entity extraction capability identified the key players in this story: OpenAI as the organization, and GPT-4 and GPT-3 as the key technical entities being discussed. Notice how it focused on the most relevant entities, filtering out less important details to give us a clear picture of who and what this text is about.

Finally, the summarization capability pulled all this understanding together to create a concise yet comprehensive summary. The summary captures the essential points - the announcement of GPT-4, its key improvements over GPT-3, and its significance. This isn't just a random selection of sentences; it's an intelligent distillation of the most important information.

Understanding the Power of Coordinated Processing
What makes this result particularly impressive isn't just the individual outputs - it's how each step builds on the others to create a complete understanding of the text. The classification provides context that helps frame the entity extraction, and both of these inform the summarization process.

Think about how this mirrors human reading comprehension. When we read a text, we naturally form an understanding of what kind of text it is, note important names and concepts, and form a mental summary - all while maintaining the relationships between these different aspects of understanding.

Practical Applications and Insights
The example we've built demonstrates a fundamental pattern that can be applied to many scenarios. While we used it to analyze a news article about AI, the same structure could be adapted to analyze:

Medical research papers, where understanding the type of study, key medical terms, and core findings is crucial Legal documents, where identifying parties involved, key clauses, and overall implications is essential Financial reports, where understanding the report type, key metrics, and main conclusions drives decision-making