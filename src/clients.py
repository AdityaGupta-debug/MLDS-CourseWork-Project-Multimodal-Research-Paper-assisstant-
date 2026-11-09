
import os
import io
import base64
import time
import torch

from PIL import Image
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from config import MAX_IMAGE_SIZE

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

groq_client = None


# --------------------------------------------------
# GROQ VLM CLIENT
# --------------------------------------------------

def get_groq_client():
    global groq_client

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing from .env")

    if groq_client is None:
        groq_client = Groq(
            api_key=GROQ_API_KEY,
            timeout=90.0,
            max_retries=1
        )

    return groq_client


# --------------------------------------------------
# MINILM TEXT EMBEDDINGS
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)


def embed_texts(texts):
    if not texts:
        return []

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()


# --------------------------------------------------
# CLIP IMAGE EMBEDDINGS
# --------------------------------------------------

clip_model_name = "openai/clip-vit-base-patch32"

clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name)

clip_model.to("cpu")
clip_model.eval()


def embed_clip_image(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    if not isinstance(features, torch.Tensor):
        features = features.pooler_output

    features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().numpy().tolist()


def embed_clip_text(text):
    inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)

    if not isinstance(features, torch.Tensor):
        features = features.pooler_output

    features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().numpy().tolist()


# --------------------------------------------------
# IMAGE LOADING
# --------------------------------------------------

def load_images(image_paths):
    images = []

    for path in image_paths:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
                images.append(image.copy())

        except (OSError, ValueError):
            continue

    return images


def image_to_base64(image):
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=80,
        optimize=True
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# --------------------------------------------------
# GROQ VLM IMAGE ANALYSIS
# --------------------------------------------------

def describe_image_with_vlm(image_path, prompt=None):
    client = get_groq_client()

    prompt = prompt or (
        "Analyze this research paper page. Describe its figures, "
        "charts, diagrams, tables, labels, axes, trends and relationships. "
        "Do not invent unreadable values."
    )

    images = load_images([image_path])

    if not images:
        return "Unable to load the PDF page image."

    image_b64 = image_to_base64(images[0])

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        reasoning_effort="none",
        max_completion_tokens=1200
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("Groq returned an empty visual description.")

    return answer.strip()


# --------------------------------------------------
# GROQ VLM ANSWER GENERATION
# --------------------------------------------------

def generate_answer(question, context, image_paths, history=""):
    client = get_groq_client()

    prompt = (
        "You are an expert multimodal research paper assistant.\n"
        "Answer using the retrieved research paper evidence and images.\n\n"
        "Instructions:\n"
        "1. Give accurate and technically detailed answers.\n"
        "2. Use images to understand figures, charts and diagrams.\n"
        "3. If evidence is insufficient, clearly say so.\n"
        "4. Cite source filenames and page numbers from the context.\n"
        "5. Never fabricate information or citations.\n"
        "6. Use conversation history for follow-up questions.\n"
        "7. Keep the answer clear and concise.\n\n"
        f"Conversation history:\n{history}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        f"Question:\n{question}"
    )

    input_content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    images = load_images(image_paths)

    # Groq Qwen 3.8 27B accepts a maximum of 3 images.
    for image in images[:3]:
        image_b64 = image_to_base64(image)

        input_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            }
        })

    print(f"Images sent to Groq: {len(images[:3])}")

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": input_content
            }
        ],
        reasoning_effort="none",
        max_completion_tokens=1500
    )

    elapsed = time.perf_counter() - start_time

    print(f"Groq generation time: {elapsed:.2f} seconds")

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("Groq returned an empty answer.")

    return answer.strip()
