import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
# from dotenv import load_dotenv
import os
import re
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
import time
import logging

# Load environment variables
# load_dotenv()

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

    # Initialize DuckDuckGo search with more specific configuration
    search = DuckDuckGoSearchAPIWrapper(
        region="us-en",   # United States region
        time="",          # No time limit (all time)
        # region="wt-wt",  # Worldwide region
        # time="y",        # Past year
        max_results=5    # Limit results
    )
    
    search_tool = Tool(
        name="DuckDuckGo Search",
        func=lambda q: search.run(q, max_results=5),
        description="Search for Illinois case law on FindLaw",
        handle_tool_error=True
    )

    def search_cases(state: AgentState) -> AgentState:
        try:
            # Extract key legal concepts from the message
            legal_concepts_prompt = """
            Extract the key legal concepts and factual elements from this text:
            {text}
            Focus on: crimes, defenses, circumstances, and legal principles.
            """
            
            key_elements = llm.invoke(legal_concepts_prompt.format(
                text=state['messages'][-1].content
            ))
            
            # Construct targeted search queries
            search_queries = [
                f"site:caselaw.findlaw.com/court/illinois {concept.strip()}" 
                for concept in key_elements.split('\n') if concept.strip()
            ]
            
            all_results = []
            for query in search_queries[:3]:  # Limit to top 3 concepts
                try:
                    results = search_tool.run(query)
                    all_results.append(f"Search for '{query}':\n{results}\n")
                except Exception as e:
                    logging.error(f"Search error for query '{query}': {str(e)}")
                    continue
            
            state["search_results"] = "\n".join(all_results)
            return state
            
        except Exception as e:
            state["search_results"] = (
                "Unable to perform search at this time. "
                "Please try rephrasing your query."
            )
            return state

    def format_response(state: AgentState) -> AgentState:
        try:
            analysis_prompt = """
            Analyze the following Illinois case law search results and provide a detailed analysis.
            
            Context: {context}
            Search Results: {results}
            
            Please provide:
            1. Most relevant cases found (with citations)
            2. Key holdings and principles from these cases
            3. How these cases might apply to the current situation
            4. Any important distinctions or variations in how courts have ruled
            5. Trends in how Illinois courts approach this issue
            
            Format your response to clearly separate these elements.
            """
            
            response = llm.invoke(analysis_prompt.format(
                context=state["messages"][-1].content,
                results=state["search_results"]
            ))
            
            state["messages"].append(FunctionMessage(
                content=f"📚 **Case Law Analysis**\n\n{response}",
                name="shepard"
            ))
            return state
            
        except Exception as e:
            state["messages"].append(FunctionMessage(
                content="Error analyzing case law. Please try again.",
                name="shepard"
            ))
            return state

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search_cases", search_cases)
    workflow.add_node("format_response", format_response)

    # Add edges
    workflow.add_edge("search_cases", "format_response")
    workflow.add_edge("format_response", END)

    # Set entry point
    workflow.set_entry_point("search_cases")

    # Compile the graph
    chain = workflow.compile()
    
    return chain

def process_query(query: str, chain):
    """Process a user query through the Shepard agent"""
    try:
        logging.info(f"Processing query: {query[:200]}...")  # Log first 200 chars
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "case_citations": [],
            "search_results": []
        }
        
        logging.info("Invoking chain...")
        result = chain.invoke(initial_state)
        logging.info("Chain processing complete")
        
        if "messages" in result and len(result["messages"]) > 0:
            return result["messages"][-1].content
        else:
            logging.warning("No response generated by the chain")
            return "No relevant case law found. Please try rephrasing your query."
            
    except Exception as e:
        logging.error(f"Error in process_query: {str(e)}")
        raise
