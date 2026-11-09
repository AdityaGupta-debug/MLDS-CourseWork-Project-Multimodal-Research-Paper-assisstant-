from src.clients import get_gemini_client, GEMINI_MODEL

client = get_gemini_client()

response = client.interactions.create(
    model=GEMINI_MODEL,
    input="Explain what a RAG system is in two sentences."
)

print(response.output_text)