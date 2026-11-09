import os
import requests
from dotenv import load_dotenv
from huggingface_hub import whoami

load_dotenv(override=True)

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN not found in .env")

print("Authenticated account:", whoami(token=token)["name"])

url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction"

response = requests.post(
    url,
    headers={"Authorization": f"Bearer {token}"},
    json={"inputs": ["hello"]},
    timeout=60
)

print("HTTP Status:", response.status_code)

if response.status_code == 200:
    print("SUCCESS: BGE embeddings are working!")
else:
    print("ERROR:", response.text[:1000])