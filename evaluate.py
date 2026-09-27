import numpy as np

from retrieval import build_index, retrieve


QUERIES = [
    # Relevant
    {
        "query": "What is spill protection?",
        "relevant": True,
    },
    {
        "query": "What are spill buckets used for?",
        "relevant": True,
    },
    {
        "query": "What is overfill protection?",
        "relevant": True,
    },
    {
        "query": "What should be done after a suspected release?",
        "relevant": True,
    },
    {
        "query": "What records should UST operators keep?",
        "relevant": True,
    },
    {
        "query": "How does statistical inventory reconciliation work?",
        "relevant": True,
    },
    {
        "query": "How often should SIR data be supplied?",
        "relevant": True,
    },
    {
        "query": "What materials can be used to absorb a fuel spill?",
        "relevant": True,
    },
    {
        "query": "How should spill buckets be maintained?",
        "relevant": True,
    },
    {
        "query": "What is release detection?",
        "relevant": True,
    },

    # Irrelevant
    {
        "query": "Who founded Microsoft?",
        "relevant": False,
    },
    {
        "query": "What is the capital of France?",
        "relevant": False,
    },
    {
        "query": "How do I install Python on macOS?",
        "relevant": False,
    },
    {
        "query": "How do neural networks learn?",
        "relevant": False,
    },
    {
        "query": "How do I cook carbonara?",
        "relevant": False,
    },
    {
        "query": "Who won World War II?",
        "relevant": False,
    },
    {
        "query": "How do I center a div in CSS?",
        "relevant": False,
    },
    {
        "query": "What is the weather in London?",
        "relevant": False,
    },
]


model, chunks, embeddings = build_index(
    "data/ust_manual.pdf"
)


evaluation_results = []

for item in QUERIES:
    results = retrieve(
        item["query"],
        model,
        embeddings,
        chunks
    )

    best_score = results[0]["score"]

    evaluation_results.append({
        "query": item["query"],
        "relevant": item["relevant"],
        "score": best_score,
        "page": results[0]["page"],
    })


evaluation_results.sort(key=lambda x: x["score"], reverse=True)

print("\n=== RETRIEVAL SCORES ===\n")

for result in evaluation_results:
    label = "YES" if result["relevant"] else "NO"

    print(
        f"{result['score']:.3f} | "
        f"relevant={label:3} | "
        f"page={result['page']:2} | "
        f"{result['query']}"
    )
