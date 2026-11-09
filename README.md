# Multimodal Research Paper RAG Assistant (Cloud Model Inference)

## Architecture

- PDF extraction and page rendering: PyMuPDF (runs in the Streamlit app environment)
- Visual descriptions and final answer generation: Qwen2.5-VL hosted inference
- Text embeddings: BAAI/bge-small-en-v1.5 hosted inference
- Native image and image-compatible text embeddings: CLIP ViT-B/32 hosted on a custom Hugging Face Inference Endpoint
- Vector database: ChromaDB with separate `paper_text` and `paper_images` collections
- Retrieval fusion: Reciprocal Rank Fusion (RRF)
- UI: Streamlit

## Important distinction

This project creates **native CLIP image embeddings** for original PDF page images. It also embeds
the user's question with CLIP's text encoder and searches the image collection in CLIP's shared
image-text space. Separately, BGE embeds extracted PDF text, VLM descriptions, and the user's
question for text retrieval. BGE and CLIP vectors are never compared directly; their ranked
results are combined with RRF.

## 1. Setup

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Add your Hugging Face token and deployed CLIP endpoint details to `.env`.

## 2. Deploy CLIP endpoint

Follow `clip_endpoint/README.md`. This is required because the main app calls the hosted CLIP API.
The endpoint is a custom FastAPI container serving CLIP image and text embeddings.

## 3. Run the app

```bash
streamlit run app.py
```

Upload PDFs, click **Process PDFs**, and ask questions.

## 4. What runs where?

- Cloud: Qwen2.5-VL, BGE embedding inference, CLIP embedding inference.
- App environment: Streamlit, PDF parsing/page rendering, and ChromaDB storage.

If you also want the UI and vector database hosted, deploy the Streamlit app to a hosting service
and use persistent storage or a managed vector database. This project keeps those pieces simple
for local development.

## Limitations

- The CLIP endpoint must be deployed and configured before indexing/querying.
- CLIP ViT-B/32 is a general image-text model; it may miss fine-grained chart values, equations,
  small labels, and dense scientific diagrams. Qwen2.5-VL's visual analysis is used alongside it.
- PDF text extraction is not OCR. Scanned PDFs may need OCR.
- Page-level image embeddings are used, not figure-cropped embeddings.
- BM25 is applied to the vector-retrieved text candidate pool in this simple implementation.
- This is a learning/portfolio project, not production infrastructure.
- Hugging Face provider/model availability can change; verify that your selected provider serves
  Qwen2.5-VL and BGE. The custom CLIP endpoint is separately deployed and billed.
- No API credentials are included. Never commit your `.env` file.
