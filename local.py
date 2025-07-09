import streamlit as st
import os
import time
from dotenv import load_dotenv
from io import BytesIO
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# ✅ Load env
load_dotenv()

# ✅ Auto-locate latest Falcon snapshot
falcon_base = Path.home() / ".cache" / "huggingface" / "hub" / "models--tiiuae--falcon-rw-1b" / "snapshots"
falcon_snapshots = sorted(falcon_base.iterdir(), key=os.path.getmtime, reverse=True)
falcon_model_path = str(falcon_snapshots[0]) if falcon_snapshots else None

# ✅ Auto-locate latest MiniLM snapshot
minilm_base = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--paraphrase-MiniLM-L3-v2" / "snapshots"
minilm_snapshots = sorted(minilm_base.iterdir(), key=os.path.getmtime, reverse=True)
embedding_model_path = str(minilm_snapshots[0]) if minilm_snapshots else None

# ✅ Load Falcon model
if not falcon_model_path:
    st.error("❌ Falcon model not found.")
    st.stop()

try:
    tokenizer = AutoTokenizer.from_pretrained(falcon_model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        falcon_model_path,
        local_files_only=True,
        device_map="auto",
        offload_folder="cpu_offload"
    )
except Exception as e:
    st.error(f"❌ Failed to load Falcon model/tokenizer: {e}")
    st.stop()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.7, top_p=0.9)
llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

# ✅ Load embeddings
if not embedding_model_path:
    st.error("❌ Embedding model not found.")
    st.stop()

embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    cache_folder=os.path.expanduser("~/.cache/huggingface"),
    model_kwargs={"local_files_only": True},
    encode_kwargs={"normalize_embeddings": True}
)

# ✅ Streamlit App
st.title("📄 Chat With Your Documents (Offline Mode)")

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the provided context.
<context>
{context}
<context>
Question: {input}
""")

def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        save_dir = "uploaded_documents"
        os.makedirs(save_dir, exist_ok=True)

        all_docs = []
        for file in uploaded_files:
            path = os.path.join(save_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.read())
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(all_docs)

        st.session_state.final_documents = docs
        st.session_state.vectors = FAISS.from_documents(docs, embedding_model)

uploaded_files = st.file_uploader("📤 Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button("Process Documents"):
    if uploaded_files:
        vector_embedding(uploaded_files)
        st.success("✅ Document embeddings and vector store ready.")
    else:
        st.warning("⚠️ Please upload at least one PDF file.")

if st.button("Clear Session"):
    st.session_state.pop("vectors", None)
    st.session_state.pop("final_documents", None)
    st.success("🧹 Session cleared.")

query = st.text_input("🔍 Ask a question from the documents")

if query and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    chain = create_retrieval_chain(retriever, document_chain)

    start = time.time()
    response = chain.invoke({"input": query})
    end = time.time()

    st.write(f"⏱ Time taken: {end - start:.2f} seconds")
    st.subheader("💬 Answer")
    st.write(response["answer"])

    with st.expander("📚 Similar Documents"):
        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("------")
