
import os
import streamlit as st

from config import (
    UPLOAD_DIR,
    IMAGE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from src.pdf_processor import process_pdf
from src.chunker import make_chunks
from src.clients import embed_texts, embed_clip_image

from src.vector_store import (
    add_text_records,
    add_image_records,
    reset_collections
)

from src.rag_pipeline import answer_question


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Multimodal Research Paper RAG",
    page_icon="📚",
    layout="wide"
)

st.title("Multimodal Research Paper RAG Assistant")

st.caption(
    "Local MiniLM text embeddings + CLIP image embeddings + "
    "Gemini 3.8 Flash visual reasoning."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# --------------------------------------------------
# PDF INDEXING FUNCTION
# --------------------------------------------------

def index_pdf(pdf_path):
    pages = process_pdf(pdf_path)

    text_records = []
    image_records = []

    filename = os.path.basename(pdf_path)

    # Collect all text chunks first
    all_chunks = []

    for page in pages:
        chunks = make_chunks(
            page["text"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )

        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "page": page["page"],
                "chunk_index": chunk_index
            })

    # Generate MiniLM embeddings in one batch
    if all_chunks:
        texts = [item["text"] for item in all_chunks]
        embeddings = embed_texts(texts)

        for item, embedding in zip(all_chunks, embeddings):
            record_id = (
                f"{filename}_page_{item['page']}_"
                f"chunk_{item['chunk_index']}"
            )

            text_records.append({
                "id": record_id,
                "document": item["text"],
                "embedding": embedding,
                "metadata": {
                    "source": filename,
                    "page": item["page"],
                    "chunk_index": item["chunk_index"]
                }
            })

    # Generate CLIP embeddings for page images
    for page in pages:
        image_path = page["image_path"]

        if not image_path or not os.path.exists(image_path):
            continue

        image_embedding = embed_clip_image(image_path)

        record_id = f"{filename}_page_{page['page']}_image"

        image_records.append({
            "id": record_id,
            "document": (
                f"Page {page['page']} of {filename}"
            ),
            "embedding": image_embedding,
            "metadata": {
                "source": filename,
                "page": page["page"],
                "image_path": image_path
            }
        })

    # Store records
    if text_records:
        add_text_records(text_records)

    if image_records:
        add_image_records(image_records)

    return len(pages), len(text_records), len(image_records)


# --------------------------------------------------
# SIDEBAR: PDF UPLOAD
# --------------------------------------------------

with st.sidebar:
    st.header("Index research papers")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    replace_index = st.checkbox(
        "Replace existing index"
    )

    if st.button(
        "Process PDFs",
        disabled=not uploaded_files,
        use_container_width=True
    ):

        if replace_index:
            with st.spinner("Resetting existing index..."):
                reset_collections()
                st.session_state.indexed_files = []
                st.session_state.messages = []

                for filename in os.listdir(IMAGE_DIR):
                    path = os.path.join(IMAGE_DIR, filename)

                    if os.path.isfile(path):
                        os.remove(path)

        total_pages = 0
        total_text = 0
        total_images = 0

        progress = st.progress(0)

        status = st.empty()

        for index, uploaded_file in enumerate(uploaded_files):
            filename = os.path.basename(uploaded_file.name)

            pdf_path = os.path.join(UPLOAD_DIR, filename)

            status.write(f"Processing {filename}...")

            with open(pdf_path, "wb") as file:
                file.write(uploaded_file.getvalue())

            try:
                with st.spinner(f"Indexing {filename}..."):
                    pages, text_count, image_count = index_pdf(pdf_path)

                total_pages += pages
                total_text += text_count
                total_images += image_count

                st.session_state.indexed_files.append(filename)

            except Exception as error:
                st.error(f"Failed to process {filename}: {error}")

            progress.progress((index + 1) / len(uploaded_files))

        status.empty()

        st.success("PDF processing completed.")

        st.write(f"Pages processed: {total_pages}")
        st.write(f"Text chunks stored: {total_text}")
        st.write(f"Image embeddings stored: {total_images}")

    st.divider()

    st.subheader("Indexed documents")

    if st.session_state.indexed_files:
        for filename in st.session_state.indexed_files:
            st.write(f"📄 {filename}")
    else:
        st.caption("No documents indexed in this session.")


# --------------------------------------------------
# CHAT HISTORY DISPLAY
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    metadata = source["metadata"]

                    st.write(
                        f"**{metadata.get('source', 'Unknown')}** "
                        f"— Page {metadata.get('page', '?')} "
                        f"({source['kind']})"
                    )


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask something about your research papers..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Build conversation history
    history = ""

    for message in st.session_state.messages[:-1]:
        history += (
            f"{message['role'].capitalize()}: "
            f"{message['content']}\n"
        )

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Retrieving text and images, then generating an answer..."
            ):
                answer, sources = answer_question(
                    question,
                    history
                )

            st.markdown(answer)

            if sources:
                with st.expander("View retrieved sources"):
                    for source in sources:
                        metadata = source["metadata"]

                        st.write(
                            f"**{metadata.get('source', 'Unknown')}** "
                            f"— Page {metadata.get('page', '?')} "
                            f"({source['kind']})"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as error:
            st.error(f"Question answering failed: {error}")
