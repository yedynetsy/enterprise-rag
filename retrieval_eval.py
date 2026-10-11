from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from retrieval import build_index, retrieve
from vector_store import (
    create_collection,
    upload_chunks,
    validate_index_data,
    search,
)


QUERIES = [
    {
        "query": "What is spill protection?",
        "expected_pages": [33, 34],
    },
    {
        "query": "How should spill buckets be maintained?",
        "expected_pages": [35],
    },
    {
        "query": "What are spill buckets used for?",
        "expected_pages": [34, 35],
    },
    {
        "query": "What is overfill protection?",
        "expected_pages": [33, 37],
    },
    {
        "query": "What are the three common types of overfill protection devices?",
        "expected_pages": [37],
    },
    {
        "query": "At what tank capacity should an automatic shutoff device stop fuel flow?",
        "expected_pages": [38, 39],
    },
    {
        "query": "When should an electronic overfill alarm activate?",
        "expected_pages": [40, 41],
    },
    {
        "query": "How does statistical inventory reconciliation work?",
        "expected_pages": [16],
    },
    {
        "query": "How often should SIR data be supplied?",
        "expected_pages": [16],
    },
    {
        "query": "What is continuous in-tank leak detection?",
        "expected_pages": [17],
    },
    {
        "query": "How does vapor monitoring detect leaks?",
        "expected_pages": [18],
    },
    {
        "query": "How does groundwater monitoring detect leaks?",
        "expected_pages": [19],
    },
    {
        "query": "How often must tanks and piping be checked for leaks?",
        "expected_pages": [9],
    },
    {
        "query": "What records must be kept for an automatic tank gauging system?",
        "expected_pages": [14],
    },
    {
        "query": "What immediate actions should be taken when a release occurs?",
        "expected_pages": [29, 32],
    },
    {
        "query": "What materials can be used to absorb a fuel spill?",
        "expected_pages": [30],
    },
    {
        "query": "When must a petroleum spill or overfill be reported to authorities?",
        "expected_pages": [30],
    },
    {
        "query": "How often must pressurized piping undergo line tightness testing?",
        "expected_pages": [27],
    },
    {
        "query": "How often must suction piping undergo line tightness testing?",
        "expected_pages": [27],
    },
    {
        "query": "How often must cathodic protection systems be tested?",
        "expected_pages": [48],
    },
    {
        "query": "How often should an impressed current rectifier be inspected?",
        "expected_pages": [49],
    },
    {
        "query": "How often must internally lined tanks be inspected?",
        "expected_pages": [50],
    },
    {
        "query": "What should be checked every 30 days during a walkthrough inspection?",
        "expected_pages": [55, 57],
    },
    {
        "query": "What should be checked annually during a walkthrough inspection?",
        "expected_pages": [55, 56, 57],
    },
    {
        "query": "How much fuel should be ordered to avoid overfilling a tank?",
        "expected_pages": [38, 44],
    },
]


def calculate_metrics(results_by_query, top_k):
    hit_at_1 = 0
    hit_at_k = 0
    reciprocal_rank_sum = 0.0

    for item, results in zip(QUERIES, results_by_query):
        expected_pages = item["expected_pages"]
        pages = [result["page"] for result in results[:top_k]]

        first_relevant_rank = None

        for rank, page in enumerate(pages, start=1):
            if page in expected_pages:
                first_relevant_rank = rank
                break

        if first_relevant_rank == 1:
            hit_at_1 += 1

        if first_relevant_rank is not None:
            hit_at_k += 1
            reciprocal_rank_sum += 1 / first_relevant_rank

    total = len(QUERIES)

    return {
        "hit@1": hit_at_1 / total,
        "hit@k": hit_at_k / total,
        "mrr@k": reciprocal_rank_sum / total,
    }


def main():
    model, chunks, embeddings = build_index(
        PDF_PATH,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    validate_index_data(chunks, embeddings)
    create_collection(recreate=True)
    upload_chunks(chunks, embeddings)

    numpy_results = []
    qdrant_results = []
    matching_queries = 0

    for item in QUERIES:
        query = item["query"]

        baseline = retrieve(
            query=query,
            model=model,
            embeddings=embeddings,
            chunks=chunks,
            top_k=TOP_K,
        )

        query_embedding = model.encode_query(
            query,
            normalize_embeddings=True,
        )

        database = search(
            query_embedding,
            top_k=TOP_K,
        )

        numpy_results.append(baseline)
        qdrant_results.append(database)

        baseline_chunks = [
            (result["page"], result["text"])
            for result in baseline
        ]
        database_chunks = [
            (result["page"], result["text"])
            for result in database
        ]

        same_results = baseline_chunks == database_chunks
        matching_queries += int(same_results)

        print(f"\nQuery: {query}")
        print(f"Expected pages: {item['expected_pages']}")
        print(f"NumPy pages:  {[r['page'] for r in baseline]}")
        print(f"Qdrant pages: {[r['page'] for r in database]}")
        print(f"Same chunks and order: {same_results}")

    print(
        f"\n=== METRICS: chunk={CHUNK_SIZE}, "
        f"overlap={CHUNK_OVERLAP}, top_k={TOP_K} ==="
    )

    for name, results in (
        ("NumPy", numpy_results),
        ("Qdrant", qdrant_results),
    ):
        metrics = calculate_metrics(results, TOP_K)

        print(
            f"{name:6} | "
            f"Hit@1={metrics['hit@1']:.3f} | "
            f"Hit@{TOP_K}={metrics['hit@k']:.3f} | "
            f"MRR@{TOP_K}={metrics['mrr@k']:.3f}"
        )

    print(
        f"\nIdentical rankings: "
        f"{matching_queries}/{len(QUERIES)}"
    )


if __name__ == "__main__":
    main()
