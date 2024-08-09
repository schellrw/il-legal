__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from dataclasses import dataclass
from typing import Literal
import streamlit as st
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from pinecone import Pinecone #, ServerlessSpec
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import MergerRetriever
from dotenv import load_dotenv
import os
from utils import process
from langchain_community.vectorstores import Chroma as LangChainChroma
import chromadb
# from chromadb.config import Settings
# from chromadb.utils import embedding_functions

# Load environment variables from the .env file
load_dotenv()

# Fetch environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

# Supplement with streamlit secrets if None
if None in [PINECONE_API_KEY, PINECONE_INDEX, HUGGINGFACE_API_TOKEN, EMBEDDINGS_MODEL, CHAT_MODEL]:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_INDEX = st.secrets["PINECONE_INDEX"]
    HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    EMBEDDINGS_MODEL = st.secrets["EMBEDDINGS_MODEL"]
    CHAT_MODEL = st.secrets["CHAT_MODEL"]

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["🗣️ Human", "🧑‍⚖️ AI Lawyer"]  #["👤 Human", "🗿 AI Lawyer"]
    message: str


def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    return embeddings


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation" not in st.session_state:
        embeddings = download_hugging_face_embeddings()
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        pinecone_docsearch = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX, embedding=embeddings)
        pinecone_retriever = pinecone_docsearch.as_retriever(
            search_kwargs={'filter': {'source': 'user_id'}}
        )
        
        # Initialize Chroma for client uploads
        # chroma_client = chromadb.Client(Settings(
        #     chroma_db_impl="duckdb+parquet",
        #     persist_directory=":memory:",
        #     anonymized_telemetry=False,
        # ))
        chroma_client = chromadb.PersistentClient(path=":memory:")
        chroma_collection = chroma_client.get_or_create_collection(
            name="user_docs",
            # embedding_function=embeddings
        )
            # chroma_db_impl="duckdb+parquet",
            # persist_directory=":memory:",
            # anonymized_telemetry=False,
        # chroma_collection = chroma_client.create_collection(
        #     name="user_docs",
        #     embedding_function=embeddings
        # )
        
        # Create LangChain Chroma wrapper
        langchain_chroma = LangChainChroma(
            client=chroma_client,
            collection_name="user_docs",
            embedding_function=embeddings
        )        
        
        # chroma_retriever = chroma_collection.as_retriever()
        chroma_retriever = langchain_chroma.as_retriever()
        
        # Combine retrievers
        combined_retriever = MergerRetriever(retrievers=[pinecone_retriever, chroma_retriever])

        # Initialize LLM and chain
        llm = HuggingFaceEndpoint(
            repo_id=CHAT_MODEL,
            model_kwargs={"huggingface_api_token":HUGGINGFACE_API_TOKEN},
            temperature=0.5,  ## make st.slider, subsequently
            top_k=10,  ## make st.slider, subsequently
        )

        prompt_template = """
            You are a trained bot to guide people about Illinois Crimnal Law Statutes and the Safe-T Act. You will answer user's query with your knowledge and the context provided. 
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            Do not say thank you and tell you are an AI Assistant and be open about everything.
            Use the following pieces of context to answer the users question.
            Context: {context}
            Question: {question}
            Only return the helpful answer below and nothing else.
            Helpful answer:
            """

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"])
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=ChatMessageHistory(),
            return_messages=True,
        )
        
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=combined_retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            memory= memory
        )

        st.session_state.conversation = retrieval_chain
        st.session_state.chroma_collection = chroma_collection
        st.session_state.langchain_chroma = langchain_chroma


def on_submit(user_input):
    if user_input:
        # print(f"User Input: {user_input}")
        response = st.session_state.conversation({
            "question":user_input
        })
        llm_response = response['answer']
        # print(f"LLM Response: {llm_response}")
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
    - Provide information on your rights and responsibilities as per Illinois legal standards.
    
    ⚖️ **Disclaimer:**
    
    While I can provide general information, it may be necessary to consult with a qualified Illinois attorney for advice tailored to your specific situation.
    
    🤖 **Getting Started:**
    
    Feel free to ask any legal question related to Illinois criminal law. I'm here to assist you!
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