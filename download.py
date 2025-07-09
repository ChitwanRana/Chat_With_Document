from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
import os
from dotenv import load_dotenv

# ✅ 1. Load token from .env
load_dotenv()

# # ✅ 2. Set cache directory (optional)
# os.environ["HF_HOME"] = "I:/huggingface_cache"

# ✅ 3. Set the model
model_id = "sentence-transformers/all-MiniLM-L6-v2"

# ✅ 4. Get token from environment
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ✅ 5. Load model with token
model = AutoModel.from_pretrained(model_id, token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

