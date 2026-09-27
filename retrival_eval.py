from retrieval import build_index, retrieve


QUERIES = [
    {
        "query": "What is spill protection?",
        "expected_pages": [33, 34],
    },
    {
        "query": "How should spill buckets be maintained?",
        "expected_pages": [34, 35, 36],
    },
    {
        "query": "What is overfill protection?",
        "expected_pages": [33, 37, 38],
    },
    {
        "query": "How does statistical inventory reconciliation work?",
        "expected_pages": [16],
    },
    {
        "query": "What are spill buckets used for?",
        "expected_pages": [34],
    },
    {
        "query": "What materials can be used to absorb a fuel spill?",
        "expected_pages": [30],
    },
    {
        "query": "What is release detection?",
        "expected_pages": [11],
    },
    {
        "query": "What should be done after a suspected release?",
        "expected_pages": [29, 30],
    },
    {
        "query": "What records should UST operators keep?",
        "expected_pages": [14],
    },
    {
        "query": "How often should SIR data be supplied?",
        "expected_pages": [16],
    },
]


def evaluate_configuration(
    pdf_path,
    chunk_size,
    overlap,
    top_k=3,
):
    model, chunks, embeddings = build_index(
        pdf_path,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    hit_at_1 = 0
    hit_at_k = 0
    reciprocal_ranks = []

    print(
        f"\n=== chunk_size={chunk_size}, "
        f"overlap={overlap} ===\n"
    )

    for item in QUERIES:
        results = retrieve(
            item["query"],
            model,
            embeddings,
            chunks,
            top_k=top_k,
        )

        expected_pages = item["expected_pages"]
        retrieved_pages = [
            result["page"]
            for result in results
        ]

        # Hit@1
        if retrieved_pages[0] in expected_pages:
            hit_at_1 += 1

        # Hit@K
        if any(
            page in expected_pages
            for page in retrieved_pages
        ):
            hit_at_k += 1

        # Reciprocal Rank
        reciprocal_rank = 0.0

        for rank, page in enumerate(
            retrieved_pages,
            start=1,
        ):
            if page in expected_pages:
                reciprocal_rank = 1 / rank
                break

        reciprocal_ranks.append(
            reciprocal_rank
        )

        print(
            f"{item['query']}\n"
            f"Expected: {expected_pages}\n"
            f"Retrieved: {retrieved_pages}\n"
            f"RR: {reciprocal_rank:.3f}\n"
        )

    total = len(QUERIES)

    hit_1_score = hit_at_1 / total
    hit_k_score = hit_at_k / total
    mrr = sum(reciprocal_ranks) / total

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "hit@1": hit_1_score,
        f"hit@{top_k}": hit_k_score,
        "mrr": mrr,
    }


CONFIGS = [
    {
        "chunk_size": 60,
        "overlap": 15,
    },
    {
        "chunk_size": 120,
        "overlap": 30,
    },
    {
        "chunk_size": 240,
        "overlap": 60,
    },
]


all_results = []

for config in CONFIGS:
    result = evaluate_configuration(
        pdf_path="data/ust_manual.pdf",
        chunk_size=config["chunk_size"],
        overlap=config["overlap"],
        top_k=3,
    )

    all_results.append(result)


print("\n=== FINAL RESULTS ===\n")

for result in all_results:
    print(
        f"chunk={result['chunk_size']:3} | "
        f"overlap={result['overlap']:3} | "
        f"Hit@1={result['hit@1']:.2f} | "
        f"Hit@3={result['hit@3']:.2f} | "
        f"MRR={result['mrr']:.2f}"
    )
