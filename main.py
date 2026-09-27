import numpy as np
import os
import pymupdf

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

TOP_K = 3

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

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

def retrieve(query, model, embeddings, chunks, top_k=TOP_K):
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

def build_context(results):
    parts = []

    for i, result in enumerate(results, start=1):
        parts.append(
            f"[Source {i}, page {result['page']}]\n"
            f"{result['text']}"
        )

    return "\n\n".join(parts)

def generate_answer(query, context):
    prompt = f"""
        You are a document question-answering assistant.

        Answer the question using ONLY the information from the provided context.

        Rules:
        - Do not use outside knowledge.
        - If the context does not contain enough information, say:
        "The answer cannot be determined from the provided documents."
        - Cite your sources using the format [Source N, page X].
        - Keep the answer concise and factual.

        Context:

        {context}

        Question:
        {query}
        """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )

    return response.text

def has_relevant_context(results, threshold=0.30):
    if not results:
        return False

    return results[0]["score"] >= threshold


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

results = retrieve(query, model, embeddings, chunks)

print("\nRetrieved chunks:")

for rank, result in enumerate(results, start=1):
    print(f"\n--- Result {rank} ---")
    print(f"Score: {result['score']:.3f}")
    print(f"Page: {result['page']}")
    print(result["text"])


if not has_relevant_context(results):
    print("\n=== Answer ===")
    print(
        "The answer cannot be determined "
        "from the provided documents."
    )
else:
    context = build_context(results)

    answer = generate_answer(
        query,
        context,
    )

    print("\n=== Answer ===")
    print(answer)
