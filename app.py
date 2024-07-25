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
# from dotenv import load_dotenv
# import os

# Load environment variables from the .env file
# load_dotenv()

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["👤 Human", "👨🏻‍⚖️ Ai"]
    message: str


def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation" not in st.session_state:
        embeddings = download_hugging_face_embeddings()
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        # pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index("il-legal")
        docsearch = PineconeVectorStore.from_existing_index(index_name="il-legal", embedding=embeddings)
        
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            model_kwargs={"huggingface_api_token":st.secrets["HUGGINGFACEHUB_API_TOKEN"]},
            # model_kwargs={"huggingface_api_token":os.environ["HUGGINGFACEHUB_API_TOKEN"]},
            temperature=0.5,
            top_k=10,
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


def on_click_callback():
    human_prompt = st.session_state.human_prompt
    st.session_state.human_prompt=""
    response = st.session_state.conversation(
        human_prompt
    )
    llm_response = response['answer']
    st.session_state.history.append(
        Message("👤 Human", human_prompt)
    )
    st.session_state.history.append(
        Message("👨🏻‍⚖️ Ai", llm_response)
    )


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
    
    Feel free to ask any legal question related to Illinois law, using keywords like "pre-trial release," "motions," or "procedure." I'm here to assist you!
    Let's get started! How may I help you today?
    """
)

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
# prompt_placeholder = st.container()
# prompt_placeholder = st.chat_input("Chat", key="human_prompt")

with chat_placeholder:
    for chat in st.session_state.history:
        st.markdown(f"{chat.origin} : {chat.message}")

        # st.chat_input(
        # "Chat", 
        # # key="human_prompt",
        # on_submit=on_click_callback
        # )

# with prompt_placeholder:
#     st.chat_input(
#         "Chat", 
#         key="human_prompt",
#         # on_submit=on_click_callback
#         )
#     st.form_submit_button(
#         "Submit",
#         on_click=on_click_callback)

with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,
    )

