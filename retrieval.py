import numpy as np
import pymupdf

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


def load_pdf(path: str):
    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)

            if text.strip():
                pages.append({
                    "page": page_number,
                    "text": text,
                })

    return pages


def create_chunks(pages, chunk_size=120, overlap=30):
    chunks = []

    step = chunk_size - overlap

    for page in pages:
        words = page["text"].split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if len(chunk_words) < 20:
                continue

            chunks.append({
                "page": page["page"],
                "text": " ".join(chunk_words),
            })

    return chunks


def build_index(
    pdf_path: str,
    chunk_size=120,
    overlap=30,
):
    pages = load_pdf(pdf_path)

    chunks = create_chunks(
        pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode_document(
        texts,
        normalize_embeddings=True,
    )

    return model, chunks, embeddings


def retrieve(
    query,
    model,
    embeddings,
    chunks,
    top_k=3,
):
    query_embedding = model.encode_query(
        query,
        normalize_embeddings=True,
    )

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "text": chunks[index]["text"],
            "page": chunks[index]["page"],
            "score": float(scores[index]),
        })

    return results
