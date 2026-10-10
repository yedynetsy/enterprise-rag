from retrieval import build_index
from vector_store import (
    create_collection,
    search,
    upload_chunks,
    validate_index_data,
)


PDF_PATH = "data/ust_manual.pdf"


model, chunks, embeddings = build_index(
    PDF_PATH,
    chunk_size=150,
    overlap=40,
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
    top_k=3,
)


print(f"\nQuery: {query}")

for rank, result in enumerate(results, start=1):
    print(f"\n--- Result {rank} ---")
    print(f"Score: {result['score']:.3f}")
    print(f"Page: {result['page']}")
    print(result["text"])
