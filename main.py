import os
import time
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_documents'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

load_dotenv()

# Model paths
falcon_model_path = r"C:\Users\kushr\.cache\huggingface\hub\models--tiiuae--falcon-rw-1b\snapshots\e4b9872bb803165eb22f0a867d4e6a64d34fce19"
embedding_model_path = r"C:\Users\kushr\.cache\huggingface\hub\models--sentence-transformers--paraphrase-MiniLM-L3-v2\snapshots\4ca70771034acceecb2e72475f72050fcdde4ddc"

print("🔧 Loading models... (may take a minute)")

# Load Falcon model
tokenizer = AutoTokenizer.from_pretrained(falcon_model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(falcon_model_path, local_files_only=True, device_map="auto", offload_folder="cpu_offload")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,  # ✅ Enables sampling
    pad_token_id=tokenizer.eos_token_id  # ✅ Prevents warning
)

llm = HuggingFacePipeline(pipeline=pipe)

embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    cache_folder=os.path.expanduser("~/.cache/huggingface"),
    model_kwargs={"local_files_only": True},
    encode_kwargs={"normalize_embeddings": True}
)

# Prompt
prompt = PromptTemplate.from_template("""
Use the context below to answer the question.
Context:
{context}

Question: {input}
""")

# Global memory
vector_store = None
documents = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global vector_store, documents

    answer, similar_docs = None, []
    message = ""

    if request.method == 'POST':
        if 'upload' in request.form:
            uploaded_files = request.files.getlist("pdfs")
            all_docs = []
            for file in uploaded_files:
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = splitter.split_documents(all_docs)

            vector_store = FAISS.from_documents(documents, embedding_model)
            message = "✅ Documents processed and embeddings created!"

        elif 'ask' in request.form:
            question = request.form.get("question")
            if question and vector_store:
                try:
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retriever = vector_store.as_retriever()
                    chain = create_retrieval_chain(retriever, document_chain)

                    start = time.time()
                    response = chain.invoke({"input": question})
                    end = time.time()

                    answer = response["answer"]
                    similar_docs = response["context"]
                    message = f"✅ Answered in {end - start:.2f} seconds"
                except Exception as e:
                    message = f"❌ Error: {str(e)}"

    return render_template('index.html', answer=answer, docs=similar_docs, message=message)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
