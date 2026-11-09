
import math
import re
from collections import Counter

from src.clients import embed_texts, embed_clip_text
from src.vector_store import (
    get_text_collection,
    get_image_collection
)


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------
# BM25
# --------------------------------------------------

def bm25_search(query, documents, top_k):
    if not documents:
        return []

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    tokenized_docs = [
        tokenize(document["document"])
        for document in documents
    ]

    total_docs = len(tokenized_docs)

    doc_lengths = [len(doc) for doc in tokenized_docs]

    avg_length = sum(doc_lengths) / max(total_docs, 1)

    document_frequency = Counter()

    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))

    k1 = 1.5
    b = 0.75

    scores = []

    for index, tokens in enumerate(tokenized_docs):
        frequencies = Counter(tokens)
        doc_length = doc_lengths[index]

        score = 0.0

        for term in query_tokens:
            frequency = frequencies.get(term, 0)

            if frequency == 0:
                continue

            df = document_frequency[term]

            idf = math.log(
                1 + (total_docs - df + 0.5) / (df + 0.5)
            )

            denominator = frequency + k1 * (
                1 - b + b * doc_length / max(avg_length, 1)
            )

            score += idf * (
                frequency * (k1 + 1) / denominator
            )

        scores.append((index, score))

    scores.sort(key=lambda item: item[1], reverse=True)

    return [
        documents[index]
        for index, score in scores[:top_k]
        if score > 0
    ]


# --------------------------------------------------
# CHROMA RESULTS CONVERSION
# --------------------------------------------------

def convert_chroma_results(results, kind):
    converted = []

    if not results or not results.get("ids"):
        return converted

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    for record_id, document, metadata in zip(
        ids,
        documents,
        metadatas
    ):
        converted.append({
            "id": record_id,
            "document": document or "",
            "metadata": metadata or {},
            "kind": kind
        })

    return converted


# --------------------------------------------------
# SEMANTIC TEXT SEARCH
# --------------------------------------------------

def text_search(query, top_k):
    collection = get_text_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas"]
    )

    return convert_chroma_results(results, "text")


# --------------------------------------------------
# CLIP IMAGE SEARCH
# --------------------------------------------------

def image_search(query, top_k):
    collection = get_image_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_clip_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas"]
    )

    return convert_chroma_results(results, "image")


# --------------------------------------------------
# RECIPROCAL RANK FUSION
# --------------------------------------------------

def reciprocal_rank_fusion(result_lists, k=60):
    scores = {}
    records = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            record_id = item["id"]

            scores[record_id] = (
                scores.get(record_id, 0) + 1 / (k + rank)
            )

            records[record_id] = item

    ranked_ids = sorted(
        scores,
        key=lambda record_id: scores[record_id],
        reverse=True
    )

    return [records[record_id] for record_id in ranked_ids]


# --------------------------------------------------
# COMPLETE HYBRID RETRIEVAL
# --------------------------------------------------

def retrieve(query, top_k_text=5, top_k_image=3):
    text_results = text_search(query, top_k_text)

    image_results = image_search(query, top_k_image)

    # BM25 searches the stored text corpus.
    collection = get_text_collection()

    bm25_results = []

    if collection.count() > 0:
        stored = collection.get(
            include=["documents", "metadatas"]
        )

        documents = []

        for record_id, document, metadata in zip(
            stored["ids"],
            stored["documents"],
            stored["metadatas"]
        ):
            documents.append({
                "id": record_id,
                "document": document or "",
                "metadata": metadata or {},
                "kind": "text"
            })

        bm25_results = bm25_search(
            query,
            documents,
            top_k_text
        )

    # Combine semantic, lexical and visual results.
    fused_results = reciprocal_rank_fusion([
        text_results,
        bm25_results,
        image_results
    ])

    return fused_results
