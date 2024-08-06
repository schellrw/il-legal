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
from dotenv import load_dotenv
import os

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
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        docsearch = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX, embedding=embeddings)
        
        repo_id = CHAT_MODEL
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
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
        
        #chain_type_kwargs = {"prompt": PROMPT}
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
            )
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(
                search_kwargs={
                    'filter': {'source': 'user_id'},
                    }),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            memory= memory
            )

        st.session_state.conversation = retrieval_chain

# def on_click_callback():
#     human_prompt = st.session_state.human_prompt
#     st.session_state.human_prompt=""
#     response = st.session_state.conversation(
#         human_prompt
#     )
#     llm_response = response['answer']
#     st.session_state.history.append(
#         Message("🗣️ Human", human_prompt)
#     )
#     st.session_state.history.append(
#         Message("🧑‍⚖️ AI Lawyer", llm_response)
#     )

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

if user_input:
    on_submit(user_input)