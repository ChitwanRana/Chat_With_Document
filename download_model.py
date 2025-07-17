from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import os

# Define local directories
llm_dir = "./models/flan-t5-base"
embedding_dir = "./models/paraphrase-MiniLM-L6-v2"

# Create directories if they don't exist
os.makedirs(llm_dir, exist_ok=True)
os.makedirs(embedding_dir, exist_ok=True)

# ✅ Download and save FLAN-T5 base model
print("📥 Downloading FLAN-T5 base...")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

llm_model.save_pretrained(llm_dir)
llm_tokenizer.save_pretrained(llm_dir)
print("✅ FLAN-T5 base saved to:", llm_dir)

# ✅ Download and save MiniLM embedding model
print("📥 Downloading MiniLM embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
embedding_model.save(embedding_dir)
print("✅ MiniLM embeddings saved to:", embedding_dir)
