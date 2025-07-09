from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")
SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
