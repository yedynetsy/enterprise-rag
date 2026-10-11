from retrieval import build_index, retrieve
from config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)
from vector_store import (
    create_collection,
    search,
    upload_chunks,
    validate_index_data,
)


def main():
    model, chunks, embeddings = build_index(
        PDF_PATH,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    validate_index_data(chunks, embeddings)

    create_collection(recreate=True)

    upload_chunks(chunks, embeddings)

    query = "What is spill protection?"

    query_embedding = model.encode_query(
        query,
        normalize_embeddings=True,
    )

    results = search(
        query_embedding,
        top_k=TOP_K,
    )

    print(f"\nQuery: {query}")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Score: {result['score']:.3f}")
        print(f"Page: {result['page']}")
        print(result["text"])


    baseline_results = retrieve(
        query=query,
        model=model,
        embeddings=embeddings,
        chunks=chunks,
        top_k=TOP_K,
    )

    print("\n=== NumPy vs Qdrant ===")

    if len(baseline_results) != len(results):
        print(
            f"Different result counts: "
            f"NumPy={len(baseline_results)}, "
            f"Qdrant={len(results)}"
        )

    for rank, (baseline, qdrant) in enumerate(
        zip(baseline_results, results),
        start=1,
    ):
        same_chunk = (
            baseline["page"] == qdrant["page"]
            and baseline["text"] == qdrant["text"]
        )

        score_difference = abs(
            baseline["score"] - qdrant["score"]
        )

        print(f"\nRank {rank}")
        print(
            f"NumPy:  page={baseline['page']}, "
            f"score={baseline['score']:.6f}"
        )
        print(
            f"Qdrant: page={qdrant['page']}, "
            f"score={qdrant['score']:.6f}"
        )
        print(f"Same chunk: {same_chunk}")
        print(f"Score difference: {score_difference:.8f}")


if __name__ == "__main__":
    main()
