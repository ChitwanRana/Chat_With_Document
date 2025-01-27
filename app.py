import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter # to convert document into chunks
from langchain.chains.combine_documents import create_stuff_documents_chain # to get relevent data from document 
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS  # vector db 
from langchain_community.document_loaders import PyPDFDirectoryLoader    # to read document 
from langchain_google_genai import GoogleGenerativeAIEmbeddings     # embedding technique 

from dotenv import load_dotenv
load_dotenv()  # to load environment variable 

# load groq and google API 

groq_api_key=os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')


st.title("Chat_With_Document")

# load model
llm=ChatGroq(groq_api_key=groq_api_key , model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():       # save entire document into session state so that we can use it anywhere 
     if "vectors" not in st.session_state:
          st.session_state.embeddings=GoogleGenerativeAIEmbeddings()
          st.session_state.loader=PyPDFDirectoryLoader("./PDFs")  #data ingestion 
          st.session_state.docs=st.session_state.loader.load()# document loading 
          st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) 
          st.session_state.final_document=st.session_state.text_splitter.split_documents(st.session_state.docs)
          st.session_state.vectors=FAISS.from_documents(st.session_state.final_document,st.session_state.embeddings)

prompt1 = st.text_input("Give prompt ")

if st.button("create vector embeddings"):
     vector_embedding()
     st.write("vector db is ready ")

import time 

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")    

