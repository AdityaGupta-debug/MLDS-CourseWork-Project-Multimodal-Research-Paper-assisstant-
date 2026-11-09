# Deploy the CLIP embedding API to the cloud

This folder contains a small FastAPI service that exposes both CLIP image and text embeddings.
Deploy this container as a **custom Hugging Face Inference Endpoint**. The main Streamlit app
will call the endpoint over HTTPS; CLIP weights and inference run in the cloud.

## 1. Build and publish the container

1. Create a Docker Hub or GitHub Container Registry account.
2. Build from this folder:

   ```bash
   docker build -t YOUR_REGISTRY/clip-embedding-api:1.0 .
   docker push YOUR_REGISTRY/clip-embedding-api:1.0
   ```

3. In Hugging Face, create a new Inference Endpoint and select a custom container image.
4. Set the container image to `YOUR_REGISTRY/clip-embedding-api:1.0`.
5. Set the container port to `8080` and health route to `/health`.
6. Choose a GPU instance if you want faster indexing. For a small demo, choose the least expensive
   compatible instance available in your account/region.
7. Set endpoint access to private/authenticated and enable scale-to-zero if available.

Hugging Face custom-container configuration and endpoint pricing/hardware options are documented at:
https://huggingface.co/docs/inference-endpoints/guides/configuration

## 2. Configure the main app

Copy the endpoint URL shown in the Hugging Face dashboard into the root `.env`:

```env
CLIP_ENDPOINT_URL=https://YOUR-ENDPOINT.endpoints.huggingface.cloud
CLIP_ENDPOINT_TOKEN=hf_your_token
```

Use the token permitted to call the endpoint. Keep it private; never commit `.env`.

## 3. Test the endpoint

Open `https://YOUR-ENDPOINT.endpoints.huggingface.cloud/health` or send an authenticated POST
to `/embed` with JSON containing either `images` (base64-encoded image strings), `texts`, or both.

## Cost note

A dedicated endpoint is billed for its provisioned compute while running. Configure scale-to-zero
or pause it when not using it, and check the current hourly price before deploying.
A scale-to-zero endpoint may have a cold-start delay on the first request.
