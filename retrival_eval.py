from retrieval import build_index, retrieve


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

    failures = []

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

        if retrieved_pages[0] in expected_pages:
            hit_at_1 += 1

        found_in_top_k = any(
            page in expected_pages
            for page in retrieved_pages
        )

        if found_in_top_k:
            hit_at_k += 1

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

        if reciprocal_rank < 1:
            failures.append({
                "query": item["query"],
                "expected": expected_pages,
                "retrieved": retrieved_pages,
                "rr": reciprocal_rank,
            })

    total = len(QUERIES)

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "hit@1": hit_at_1 / total,
        f"hit@{top_k}": hit_at_k / total,
        "mrr": sum(reciprocal_ranks) / total,
        "failures": failures,
    }


CONFIGS = [
    {
        "chunk_size": 90,
        "overlap": 20,
    },
    {
        "chunk_size": 120,
        "overlap": 30,
    },
    {
        "chunk_size": 150,
        "overlap": 40,
    },
    {
        "chunk_size": 180,
        "overlap": 45,
    },
]


all_results = []

for config in CONFIGS:
    print(
        f"\nEvaluating "
        f"chunk_size={config['chunk_size']}, "
        f"overlap={config['overlap']}..."
    )

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
        f"overlap={result['overlap']:2} | "
        f"Hit@1={result['hit@1']:.3f} | "
        f"Hit@3={result['hit@3']:.3f} | "
        f"MRR={result['mrr']:.3f}"
    )


print("\n=== FAILURES ===")

for result in all_results:
    print(
        f"\n--- chunk={result['chunk_size']} "
        f"overlap={result['overlap']} ---"
    )

    if not result["failures"]:
        print("No failures.")
        continue

    for failure in result["failures"]:
        print()
        print(f"Query:     {failure['query']}")
        print(f"Expected:  {failure['expected']}")
        print(f"Retrieved: {failure['retrieved']}")
        print(f"RR:        {failure['rr']:.3f}")