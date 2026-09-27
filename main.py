import os

from dotenv import load_dotenv
from google import genai

from retrieval import build_index, retrieve


PDF_PATH = "data/ust_manual.pdf"
RELEVANCE_THRESHOLD = 0.30
TOP_K = 3


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


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


def has_relevant_context(
    results,
    threshold=RELEVANCE_THRESHOLD,
):
    if not results:
        return False

    return results[0]["score"] >= threshold


def main():
    print("Building index...")

    model, chunks, embeddings = build_index(
        PDF_PATH,
        chunk_size=120,
        overlap=30,
    )

    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")

    query = input("\nAsk a question: ")

    results = retrieve(
        query=query,
        model=model,
        embeddings=embeddings,
        chunks=chunks,
        top_k=TOP_K,
    )

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
        return

    context = build_context(results)

    answer = generate_answer(
        query=query,
        context=context,
    )

    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()