import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Fetch environment variables
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

# Supplement with streamlit secrets if None
if None in [HUGGINGFACE_API_TOKEN, EMBEDDINGS_MODEL, CHAT_MODEL]:
    HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    EMBEDDINGS_MODEL = st.secrets["EMBEDDINGS_MODEL"]
    CHAT_MODEL = st.secrets["CHAT_MODEL"]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    case_citations: Annotated[list[str], "List of found case citations"]
    search_results: Annotated[list[str], "Search results from DuckDuckGo"]

def create_agent_shepard():
    # Initialize LLM
    llm = HuggingFaceEndpoint(
        repo_id=CHAT_MODEL,
        model_kwargs={"huggingface_api_token": HUGGINGFACE_API_TOKEN},
        temperature=0.5,
    )

    # Initialize DuckDuckGo search tool
    search = DuckDuckGoSearchRun()
    search_tool = Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Search for Illinois case law on FindLaw"
    )

    # Create prompts for different agent functions
    citation_prompt = PromptTemplate.from_template("""
        Identify any case law citations in the following text. 
        Return only the citations in a list format.
        Text: {text}
    """)

    search_prompt = PromptTemplate.from_template("""
        Search for Illinois case law related to: {query}
        Only search within caselaw.findlaw.com/court/illinois
        Return the most relevant case citations and brief summaries.
    """)

    # Define agent functions
    def identify_citations(state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        citations = llm.invoke(citation_prompt.format(text=last_message.content))
        state["case_citations"] = citations
        return state

    def search_cases(state: AgentState) -> AgentState:
        query = f"site:caselaw.findlaw.com/court/illinois {state['messages'][-1].content}"
        results = search_tool.run(query)
        state["search_results"] = results
        return state

    def format_response(state: AgentState) -> AgentState:
        response_prompt = """
        Based on the search results and case citations, provide a comprehensive analysis.
        Citations: {citations}
        Search Results: {results}
        """
        response = llm.invoke(response_prompt.format(
            citations=state["case_citations"],
            results=state["search_results"]
        ))
        state["messages"].append(FunctionMessage(content=response, name="shepard"))
        return state

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("identify_citations", identify_citations)
    workflow.add_node("search_cases", search_cases)
    workflow.add_node("format_response", format_response)

    # Add edges
    workflow.add_edge("identify_citations", "search_cases")
    workflow.add_edge("search_cases", "format_response")
    workflow.add_edge("format_response", END)

    # Set entry point
    workflow.set_entry_point("identify_citations")

    # Compile the graph
    chain = workflow.compile()
    
    return chain

def process_query(query: str, chain):
    """Process a user query through the Shepard agent"""
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "case_citations": [],
        "search_results": []
    }
    result = chain.invoke(initial_state)
    return result["messages"][-1].content
