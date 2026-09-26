import numpy as np
import pymupdf

from sentence_transformers import SentenceTransformer


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


pages = load_pdf("data/ust_manual.pdf")
chunks = create_chunks(pages)

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")


model = SentenceTransformer(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode_document(
    texts,
    normalize_embeddings=True,
)

print(f"Embeddings shape: {embeddings.shape}")


query = input("\nAsk a question: ")

query_embedding = model.encode_query(
    query,
    normalize_embeddings=True,
)

scores = embeddings @ query_embedding

TOP_K = 3
top_indices = np.argsort(scores)[::-1][:TOP_K]

for rank, index in enumerate(top_indices, start=1):
    chunk = chunks[index]

    print(f"\n--- Result {rank} ---")
    print(f"Score: {scores[index]:.3f}")
    print(f"Page: {chunk['page']}")
    print(chunk["text"])
