import os

home = os.path.expanduser("~")
path = os.path.join(home, ".cache", "huggingface", "hub")
print("Default cache path exists:", os.path.exists(path))
