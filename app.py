from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
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
from werkzeug.utils import secure_filename
import uuid
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

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
     You are an AI assistant answering questions based ONLY on the context provided.  
    - If the answer is not in the context, reply: "I could not find this information in the documents."  
    - Be concise and factual.  
    
    Context:
    {context}
    
    Question: {input}
    """
)

# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to ensure event loop exists
def get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Helper function for logging
def flash_and_print(message):
    flash(message)
    print(message)  # Also log to console for debugging

# Process uploaded files and create vector embeddings
def process_documents(files):
    # Ensure event loop exists in this thread
    loop = get_or_create_event_loop()
    
    # Track processing statistics
    total_files = len(files)
    processed_files = 0
    total_pages = 0
    total_chunks = 0
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load documents from uploaded files
    all_docs = []
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                flash_and_print(f"Processing {filename}...")
                
                # Load document using PyPDFLoader
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                
                # Count pages
                file_pages = len(docs)
                total_pages += file_pages
                
                all_docs.extend(docs)
                processed_files += 1
                
                flash_and_print(f"Successfully loaded {filename} ({file_pages} pages)")
            except Exception as e:
                flash_and_print(f"Error processing {file.filename}: {str(e)}")
    
    if not all_docs:
        flash_and_print("No documents were successfully processed!")
        return None
        
    flash_and_print(f"Processed {processed_files}/{total_files} files, {total_pages} total pages")
    
    # Split documents with larger chunk size and overlap
    flash_and_print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Increased from 1000
        chunk_overlap=100,  # Increased from 200
        length_function=len
    )
    final_documents = text_splitter.split_documents(all_docs)
    total_chunks = len(final_documents)
    
    flash_and_print(f"Created {total_chunks} document chunks")
    
    if total_chunks == 0:
        flash_and_print("No document chunks were created!")
        return None
    
    # Create vector store
    flash_and_print("Creating vector embeddings (this may take a while)...")
    vectors = FAISS.from_documents(final_documents, embeddings)
    
    flash_and_print(f"Successfully created vector index from {total_chunks} chunks")
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
        if vectors is not None:
            # Save vectors to session
            session['has_vectors'] = True
            app.config['vectors'] = vectors
            flash('Documents processed successfully!')
        else:
            flash('Failed to create vector embeddings')
    except Exception as e:
        flash(f'Error processing documents: {str(e)}')
    
    return redirect(url_for('index'))

@app.route('/query', methods=['POST'])
def query_documents():
    # Ensure event loop exists in this thread
    loop = get_or_create_event_loop()
    
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
        retriever = vectors.as_retriever(search_kwargs={"k": 6})
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)