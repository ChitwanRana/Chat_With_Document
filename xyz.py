import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter  # to convert document into chunks
from langchain.chains.combine_documents import create_stuff_documents_chain  # to get relevant data from document
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS  # vector db
from langchain_community.document_loaders import PyPDFLoader  # to read a single PDF
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # embedding technique

from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Streamlit title
st.title("Chat With Document")

# Load the LLM model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

import os

def process_uploaded_file(uploaded_file):
    """
    Process the uploaded PDF file to create embeddings and save them in session state.
    """
    if uploaded_file is not None:
        # Ensure the `./temp` directory exists
        os.makedirs("./temp", exist_ok=True)
        
        # Save uploaded file to the `./temp` directory
        file_path = f"./temp/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load the document and process it
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(models="models/embedding-001")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_document = text_splitter.split_documents(docs)

        # Create a vector store for embeddings
        st.session_state.vectors = FAISS.from_documents(final_document, st.session_state.embeddings)
        st.write("Vector database is ready.")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    if st.button("Process Document"):
        # Process the uploaded file and create vector embeddings
        process_uploaded_file(uploaded_file)

# Input for user prompt
prompt1 = st.text_input("Ask a question about the document")

if prompt1 and "vectors" in st.session_state:
    # Create the retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Process the user query
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time:", time.process_time() - start)

    # Display the response
    st.write(response['answer'])

    # Expandable section for relevant document chunks
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
