
import time

from src.clients import generate_answer
from src.retriever import retrieve

from config import (
    TOP_K_TEXT,
    TOP_K_IMAGE,
    TOP_K_FINAL,
    MAX_IMAGES_PER_QUESTION,
    MAX_CONTEXT_CHARS
)


def answer_question(question, history=""):
    question_clean = question.strip().lower()

    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you?"
    }

    if question_clean in greetings:
        return (
            "Hello! I am your Multimodal Research Paper Assistant. "
            "Ask me anything about your uploaded research papers.",
            []
        )

    # STEP 1: RETRIEVAL
    retrieval_start = time.perf_counter()

    results = retrieve(
        question,
        TOP_K_TEXT,
        TOP_K_IMAGE
    )

    retrieval_time = time.perf_counter() - retrieval_start

    print(f"\nRetrieval time: {retrieval_time:.2f}s")

    if not results:
        return (
            "I couldn't find relevant information. "
            "Upload and index a research paper first.",
            []
        )

    # STEP 2: SELECT RESULTS
    selected = results[:TOP_K_FINAL]

    context_parts = []
    image_paths = []
    seen_documents = set()

    for item in selected:
        metadata = item["metadata"]

        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "?")
        kind = item["kind"]
        document = item["document"]

        unique_key = (source, page, kind, document)

        if unique_key in seen_documents:
            continue

        seen_documents.add(unique_key)

        if kind == "text":
            context_parts.append(
                f"[Source: {source}, page: {page}]\n{document}"
            )

        elif kind == "image":
            context_parts.append(
                f"[Visual source: {source}, page: {page}]\n"
                f"The original page image is attached for visual analysis."
            )

            image_path = metadata.get("image_path")

            if image_path and image_path not in image_paths:
                image_paths.append(image_path)

    # STEP 3: BUILD CONTEXT
    context = "\n\n".join(context_parts)
    context = context[:MAX_CONTEXT_CHARS]

    image_paths = image_paths[:MAX_IMAGES_PER_QUESTION]

    print(f"Context characters: {len(context)}")
    print(f"Images sent to Gemini: {len(image_paths)}")

    # STEP 4: GENERATE ANSWER
    generation_start = time.perf_counter()

    answer = generate_answer(
        question,
        context,
        image_paths,
        history
    )

    generation_time = time.perf_counter() - generation_start

    print(f"Gemini generation time: {generation_time:.2f}s")

    print(
        f"Total pipeline time: "
        f"{retrieval_time + generation_time:.2f}s\n"
    )

    return answer, selected
