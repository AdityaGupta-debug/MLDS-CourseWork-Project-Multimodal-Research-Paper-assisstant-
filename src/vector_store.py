
import chromadb

from config import CHROMA_DIR


# Initialize persistent ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DIR)


# --------------------------------------------------
# COLLECTIONS
# --------------------------------------------------

def get_text_collection():
    return client.get_or_create_collection(
        name="paper_text",
        metadata={"hnsw:space": "cosine"}
    )


def get_image_collection():
    return client.get_or_create_collection(
        name="paper_images",
        metadata={"hnsw:space": "cosine"}
    )


# --------------------------------------------------
# ADD TEXT RECORDS
# --------------------------------------------------

def add_text_records(records):
    if not records:
        return

    collection = get_text_collection()

    collection.upsert(
        ids=[record["id"] for record in records],
        documents=[record["document"] for record in records],
        embeddings=[record["embedding"] for record in records],
        metadatas=[record["metadata"] for record in records]
    )


# --------------------------------------------------
# ADD IMAGE RECORDS
# --------------------------------------------------

def add_image_records(records):
    if not records:
        return

    collection = get_image_collection()

    collection.upsert(
        ids=[record["id"] for record in records],
        documents=[record["document"] for record in records],
        embeddings=[record["embedding"] for record in records],
        metadatas=[record["metadata"] for record in records]
    )


# --------------------------------------------------
# RESET COLLECTIONS
# --------------------------------------------------

def reset_collections():
    global client

    try:
        client.delete_collection("paper_text")
    except Exception:
        pass

    try:
        client.delete_collection("paper_images")
    except Exception:
        pass

    get_text_collection()
    get_image_collection()


# --------------------------------------------------
# COLLECTION COUNTS
# --------------------------------------------------

def get_collection_counts():
    return {
        "text_records": get_text_collection().count(),
        "image_records": get_image_collection().count()
    }
