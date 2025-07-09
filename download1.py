from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("HUGGINGFACE_HUB_TOKEN is not set in .env")


os.environ["HF_TOKEN"] = hf_token

os.environ['HF_HOME'] = 'I:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='mistralai/Mistral-7B-Instruct-v0.1',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)