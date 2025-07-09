import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
from io import BytesIO

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# App title
st.title("Chat With Document")

# Initialize the ChatGroq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    
# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Define the vector embedding function
def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Directory for storing uploaded files
        save_dir = "uploaded_documents"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Load documents from uploaded files and save them to the server
        all_docs = []
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            
            # Save to the backend directory
            file_path = os.path.join(save_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            
            # Load document using PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
        

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs)
        
        # Create vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button("Process Documents"):
    if uploaded_files:
        vector_embedding(uploaded_files)
        st.success("Vector Store DB is ready.")
    else:
        st.error("Please upload at least one PDF file.")

# Question input and retrieval
prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure response time
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time:", time.process_time() - start)
    
    # Display the response
    st.write(response['answer'])

    # Display document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
