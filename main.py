import os

from dotenv import load_dotenv
from google import genai

from config import TOP_K
from retrieval import load_embedding_model
from vector_store import search


RELEVANCE_THRESHOLD = 0.30


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


def answer_question(question):
    question = question.strip()

    if not question:
        raise ValueError("Please enter a question.")

    model = load_embedding_model()

    query_embedding = model.encode_query(
        question,
        normalize_embeddings=True,
    )

    results = search(
        query_embedding,
        top_k=TOP_K,
    )

    if not has_relevant_context(results):
        return {
            "answer": (
                "The answer cannot be determined "
                "from the provided documents."
            ),
            "sources": [],
            "status": "insufficient_context",
        }

    context = build_context(results)

    answer = generate_answer(
        query=question,
        context=context,
    )

    sources = []

    for number, result in enumerate(results, start=1):
        sources.append({
            "source_id": number,
            "document": result["document"],
            "page": result["page"],
            "text": result["text"],
            "score": result["score"],
        })

    return {
        "answer": answer,
        "sources": sources,
        "status": "answered",
    }


def main():
    print("Loading embedding model...")
    load_embedding_model()

    question = input("\nAsk a question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    result = answer_question(question)

    print(f"\nStatus: {result['status']}")

    print("\n=== Answer ===")
    print(result["answer"])

    print("\n=== Sources ===")

    for source in result["sources"]:
        print(
            f"\n[Source {source['source_id']}, "
            f"page {source['page']}]"
        )
        print(f"Document: {source['document']}")
        print(f"Score: {source['score']:.3f}")
        print(source["text"])


if __name__ == "__main__":
    main()