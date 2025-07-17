import streamlit as st
import os
import time
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# ✅ Load environment variables
load_dotenv()

# ✅ Local model paths
LLM_PATH = "./models/flan-t5-base"
EMBEDDING_PATH = "./models/paraphrase-MiniLM-L6-v2"

# ✅ Load FLAN-T5 model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_PATH,
        local_files_only=True,
        torch_dtype="float32"
    )
except Exception as e:
    st.error(f"❌ Failed to load FLAN-T5 model: {e}")
    st.stop()

# ✅ Create generation pipeline
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)

llm = HuggingFacePipeline(pipeline=pipe)

# ✅ Load local embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_PATH,
    model_kwargs={"local_files_only": True},
    encode_kwargs={"normalize_embeddings": True}
)

# ✅ Streamlit UI setup
st.title("📄 Offline PDF QA (FLAN-T5 + MiniLM)")

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# ✅ Prompt template
prompt = PromptTemplate.from_template("""
Answer the following question based on the given context.

Context: {context}

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

    progress_text.write("🧠 Step 3: Creating embeddings and FAISS index...")
    st.session_state.final_documents = docs
    st.session_state.vectors = FAISS.from_documents(docs, embedding_model)

    st.session_state.processing_complete = True
    progress_text.write("✅ Ready to answer your questions!")

# ✅ Upload PDF
uploaded_files = st.file_uploader("📤 Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button("🔄 Process Documents"):
    if uploaded_files:
        vector_embedding(uploaded_files)
        st.success("✅ Documents processed. Ask your questions below!")
    else:
        st.warning("⚠️ Please upload at least one PDF file.")

# ✅ Clear session
if st.button("🧹 Clear Session"):
    st.session_state.pop("vectors", None)
    st.session_state.pop("final_documents", None)
    st.session_state.processing_complete = False
    st.success("Session cleared.")

# ✅ Ask questions
if st.session_state.processing_complete:
    st.markdown("---")
    st.subheader("💬 Ask a question")
    query = st.text_input("🔍 Your question:")

    if st.button("🧠 Get Answer") and query:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})
        chain = create_retrieval_chain(retriever, document_chain)

        start = time.time()
        response = chain.invoke({"input": query})
        end = time.time()

        st.write(f"⏱ Time taken: {end - start:.2f} seconds")
        st.subheader("✅ Answer")
        st.write(response["answer"])

        with st.expander("📚 Retrieved Documents"):
            for doc in response["context"]:
                st.markdown(doc.page_content)
                st.write("------")
else:
    st.info("📥 Please upload and process PDFs to enable question answering.")
