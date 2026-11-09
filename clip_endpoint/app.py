import base64
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import open_clip

app = FastAPI(title="Hosted CLIP Embedding API")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED, device=device
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model.eval()

class EmbedRequest(BaseModel):
    images: list[str] = []
    texts: list[str] = []

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": device}

@app.post("/embed")
def embed(request: EmbedRequest):
    if not request.images and not request.texts:
        raise HTTPException(status_code=400, detail="Provide images or texts.")
    vectors = []
    with torch.no_grad():
        for encoded in request.images:
            try:
                image_bytes = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                tensor = preprocess(image).unsqueeze(0).to(device)
                vector = model.encode_image(tensor)
                vector = vector / vector.norm(dim=-1, keepdim=True)
                vectors.append(vector[0].cpu().tolist())
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
        for text in request.texts:
            tokens = tokenizer([text]).to(device)
            vector = model.encode_text(tokens)
            vector = vector / vector.norm(dim=-1, keepdim=True)
            vectors.append(vector[0].cpu().tolist())
    return {"embeddings": vectors}
