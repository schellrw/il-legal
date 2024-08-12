__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from dataclasses import dataclass
from typing import Literal
import streamlit as st
from dotenv import load_dotenv
import os
from utils import process
from chat.bot import ChatBot

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["🗣️ Human", "🧑‍⚖️ AI Lawyer"]  #["👤 Human", "🗿 AI Lawyer"]
    message: str


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation" not in st.session_state:
        chatbot = ChatBot()
        st.session_state.conversation = chatbot.retrieval_chain
        st.session_state.chroma_collection = chatbot.chroma_collection
        st.session_state.langchain_chroma = chatbot.langchain_chroma


def on_submit(user_input):
    if user_input:
        response = st.session_state.conversation({
            "question":user_input
        })
        llm_response = response['answer']
        st.session_state.history.append(
            Message("🗣️ Human", user_input)
        )
        st.session_state.history.append(
            Message("🧑‍⚖️ AI Lawyer", llm_response)
        )

        # Optionally, you can print the source documents to see where the information came from
        if 'source_documents' in response:
            print("Source Documents:")
            for doc in response['source_documents']:
                print(f"- {doc.metadata.get('source', 'Unknown source')}: {doc.page_content[:100]}...")

        st.rerun()

initialize_session_state()

st.title("IL-Legal Advisor Chatbot")

st.markdown(
    """
    👋 **Welcome to IL-Legal Advisor!**
    I'm here to assist you with your legal queries within the framework of Illinois criminal law. Whether you're navigating through specific legal issues or seeking general advice, I'm here to help.
    
    📚 **How I Can Assist:**
    
    - Answer questions on various aspects of Illinois criminal law.
    - Guide you through legal processes relevant to Illinois.
    - Provide information on your rights and responsibilities regarding Illinois legal standards.
    
    ⚖️ **Disclaimer:**
    
    While I can provide general information, it may be necessary to consult with a qualified Illinois attorney for advice tailored to your specific situation.
    
    🤖 **Getting Started:**
    
    Feel free to ask any legal question related to Illinois criminal law. I'm here to assist you!
    If you have any documents pertinent to your case to better assist you, please upload them below.
    Let's get started! How may I help you today?
    """
)

chat_placeholder = st.container()

with chat_placeholder:
    for chat in st.session_state.history:
        st.markdown(f"{chat.origin} : {chat.message}")

user_input = st.chat_input("Enter your question here...")

# File upload and processing
uploaded_file = st.file_uploader("Upload your legal document", type="pdf")

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        text = process.extract_text_from_pdf(uploaded_file)
        chunks = process.chunk_text(text)
        st.session_state.user_chunks = chunks
        st.success(f"Uploaded {uploaded_file.name} successfully with {len(chunks)} chunks")

        # Add chunks to Chroma
        ids = [f"doc_{i}" for i in range(len(chunks))]
        metadatas = [{"source": "user_upload"} for _ in chunks] #range(len(chunks))],
        st.session_state.chroma_collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

        # Add chunks to LangChain Chroma wrapper
        st.session_state.langchain_chroma.add_texts(
            texts=chunks,
            metadatas=metadatas
        )

        st.success("Document processed and vectorized successfully!")

    except Exception as e:
        st.error(f"An error occurred while processing {uploaded_file.name}: {str(e)}")


if user_input:
    on_submit(user_input)