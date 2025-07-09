import streamlit as st
import os
import time
from dotenv import load_dotenv
from io import BytesIO

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# ✅ Load .env
load_dotenv()

# ✅ LOCAL MODEL PATHS
falcon_model_path = r"C:\Users\kushr\.cache\huggingface\hub\models--tiiuae--falcon-rw-1b\snapshots\e4b9872bb803165eb22f0a867d4e6a64d34fce19"
embedding_model_path = r"C:\Users\kushr\.cache\huggingface\hub\models--sentence-transformers--paraphrase-MiniLM-L3-v2\snapshots\4ca70771034acceecb2e72475f72050fcdde4ddc"


# ✅ Load Falcon-RW-1B (causal model)
try:
    tokenizer = AutoTokenizer.from_pretrained(falcon_model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        falcon_model_path,
        local_files_only=True,
        device_map="auto",
        offload_folder="cpu_offload"  # Needed for lower RAM machines
    )
except Exception as e:
    st.error(f"❌ Failed to load Falcon model/tokenizer: {e}")
    st.stop()

# ✅ Text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)

# ✅ LangChain wrapper
llm = HuggingFacePipeline(pipeline=pipe)

# ✅ Load MiniLM embeddings (local)
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    cache_folder=os.path.expanduser("~/.cache/huggingface"),
    model_kwargs={"local_files_only": True},
    encode_kwargs={"normalize_embeddings": True}
)

# ✅ Streamlit Title
st.title("📄 Chat With Your Documents (Offline Mode)")

# ✅ Session state flags
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# ✅ Prompt template
prompt = PromptTemplate.from_template("""
Use the below context to answer the question.
Context:
{context}

Question: {input}
""")

# ✅ Embed PDF documents
def vector_embedding(uploaded_files):
    save_dir = "uploaded_documents"
    os.makedirs(save_dir, exist_ok=True)

    progress_text = st.empty()
    progress_text.write("🔍 Step 1: Saving and loading documents...")
    all_docs = []

    for file in uploaded_files:
        path = os.path.join(save_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.read())
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    progress_text.write("✂️ Step 2: Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(all_docs)

    progress_text.write("🧠 Step 3: Creating embeddings and FAISS vector store...")
    st.session_state.final_documents = docs
    st.session_state.vectors = FAISS.from_documents(docs, embedding_model)

    st.session_state.processing_complete = True
    progress_text.write("✅ All steps completed. Ready to ask questions!")

# ✅ File Upload
uploaded_files = st.file_uploader("📤 Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button("🔄 Process Documents"):
    if uploaded_files:
        vector_embedding(uploaded_files)
        st.success("🎉 Document processing complete. You can now ask questions!")
    else:
        st.warning("⚠️ Please upload at least one PDF file.")

# ✅ Clear session (optional)
if st.button("🧹 Clear Session"):
    st.session_state.pop("vectors", None)
    st.session_state.pop("final_documents", None)
    st.session_state.processing_complete = False
    st.success("Session cleared. Re-upload documents to restart.")

# ✅ Enable question box only if processing complete
if st.session_state.processing_complete:
    st.markdown("---")
    st.subheader("💬 Ask a question from the documents")
    query = st.text_input("🔍 Enter your question")

    if st.button("🧠 Get Answer") and query:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        chain = create_retrieval_chain(retriever, document_chain)

        start = time.time()
        response = chain.invoke({"input": query})
        end = time.time()

        st.write(f"⏱ Time taken: {end - start:.2f} seconds")
        st.subheader("✅ Answer")
        st.write(response["answer"])

        with st.expander("📚 Similar Documents"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("------")
else:
    st.info("📥 Please process your uploaded PDFs first before asking questions.")
