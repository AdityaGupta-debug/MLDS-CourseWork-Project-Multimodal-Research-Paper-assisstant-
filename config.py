
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Retrieval configuration
TOP_K_TEXT = int(os.getenv("TOP_K_TEXT", "5"))
TOP_K_IMAGE = int(os.getenv("TOP_K_IMAGE", "3"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "4"))

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CH  UNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Image configuration
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
MAX_IMAGES_PER_QUESTION = int(os.getenv("MAX_IMAGES_PER_QUESTION", "2"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

# Project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploaded_pdfs")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "extracted_images")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

for directory in [UPLOAD_DIR, IMAGE_DIR, CHROMA_DIR]:
    os.makedirs(directory, exist_ok=True)
