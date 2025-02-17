# Importing Necessary Dependencies
import os
from turtle import mode
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

st.title('RAG-Q&A-Chatbot-ALong-Chat-History')

def initialize_contextualize_prompt():
    system_prompt = (
        "Give a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulate a stadalone question which can be understood"
        "without the chat history. Do Not answer the question"
        'just reformulate it if needed and otherwise return it as is.'
    )


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ]
    )

    return contextualize_q_prompt

def initialize_qa_prompt():
    system_prompt = (
        "Give a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulate a stadalone question which can be understood"
        "without the chat history. Do Not answer the question"
        'just reformulate it if needed and otherwise return it as is.'
    )

    template = """Answer the question based only on the following context:

    {context}

    Question: {input}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', template)
        ]
    )

    return qa_prompt

st.session_state.store = {}

def get_session_memory(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()

    else:
        return st.session_state.store[session_id]

chat_session_name = st.text_input('session name')

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if st.button('Load'):
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the PDF using LangChain's PyPDFLoader
        st.session_state.loader = PyPDFLoader("temp.pdf")
        st.session_state.documents = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)

    st.session_state.embedding = OllamaEmbeddings(model='mxbai-embed-large:335m')
    st.session_state.vectordb = Chroma.from_documents(documents=st.session_state.final_documents, embedding = st.session_state.embedding, persist_directory=None)

    st.session_state.llm = ChatOllama(model='llama3.2:3b')
    st.session_state.retriever = st.session_state.vectordb.as_retriever()

    st.session_state.contextualize_q_prompt = initialize_contextualize_prompt()

    st.session_state.history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm,
        st.session_state.retriever,
        st.session_state.contextualize_q_prompt
    )

    st.session_state.qa_prompt = initialize_qa_prompt()
    
    st.session_state.document_chain = create_stuff_documents_chain(st.session_state.llm, st.session_state.qa_prompt)

    st.session_state.rag_chain = create_retrieval_chain(st.session_state.history_aware_retriever, st.session_state.document_chain)

    st.session_state.runnable = runnable = RunnableWithMessageHistory(
        st.session_state.rag_chain,
        get_session_memory,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )

user_input = st.text_input('Enter what you want to ask...')

if user_input:
    if "runnable" in st.session_state and st.session_state.runnable is not None:
        response = st.session_state.runnable.invoke(
            {"input": f"{user_input}"},
            config={'configurable': {'session_id': chat_session_name}}
        )

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            st.write(response['answer'])

        # Show document similarity search results
        with st.expander("Document Similarity Search"):
            if "context" in response:
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            else:
                st.write("No similar documents found.")
    else:
        st.error("Error: Please click 'Load' to initialize the system before asking questions.")