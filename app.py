from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
import os
from langchain_groq import ChatGroq #type:ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter #type:ignore
from langchain.chains.combine_documents import create_stuff_documents_chain #type:ignore
from langchain_core.prompts import ChatPromptTemplate #type:ignore
from langchain.chains import create_retrieval_chain #type:ignore
from langchain_community.vectorstores import FAISS #type:ignore
from langchain_community.document_loaders import PyPDFLoader #type:ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings #type:ignore
from dotenv import load_dotenv #type:ignore
import time
from werkzeug.utils import secure_filename
import uuid

# Load environment variables
load_dotenv()

# Configuration
UPLOAD_FOLDER = 'uploaded_documents'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.secret_key = os.getenv('SECRET_KEY', str(uuid.uuid4()))

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define prompt template
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

# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Process uploaded files and create vector embeddings
def process_documents(files):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load documents from uploaded files
    all_docs = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load document using PyPDFLoader
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            all_docs.extend(docs)
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(all_docs)
    
    # Create vector store
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    try:
        vectors = process_documents(files)
        # Save vectors to session (Note: this is simplified, you'll need a proper storage solution)
        session['has_vectors'] = True
        # In a real app, you'd save the vectors to disk/database with a session ID
        # For this example, we'll use a global variable (not suitable for production)
        app.config['vectors'] = vectors
        flash('Documents processed successfully!')
    except Exception as e:
        flash(f'Error processing documents: {str(e)}')
    
    return redirect(url_for('index'))

@app.route('/query', methods=['POST'])
def query_documents():
    if not session.get('has_vectors', False):
        return jsonify({'error': 'No documents have been processed yet'})
    
    query = request.form.get('query')
    if not query:
        return jsonify({'error': 'No query provided'})
    
    try:
        # Retrieve vectors
        vectors = app.config.get('vectors')
        
        # Create document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Measure response time
        start = time.process_time()
        response = retrieval_chain.invoke({'input': query})
        response_time = time.process_time() - start
        
        # Extract context snippets
        context_snippets = [doc.page_content for doc in response.get("context", [])]
        
        return jsonify({
            'answer': response['answer'],
            'response_time': response_time,
            'context_snippets': context_snippets
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)