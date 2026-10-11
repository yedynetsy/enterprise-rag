import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ust_manual"
VECTOR_SIZE = 384


client = QdrantClient(url=QDRANT_URL)


def validate_index_data(chunks, embeddings):
    if not chunks:
        raise ValueError("Cannot index an empty list of chunks.")

    vectors = np.asarray(embeddings)

    if vectors.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D array: "
            "(number_of_chunks, vector_size)."
        )

    if len(chunks) != vectors.shape[0]:
        raise ValueError(
            f"Count mismatch: {len(chunks)} chunks, "
            f"{vectors.shape[0]} embeddings."
        )

    if vectors.shape[1] != VECTOR_SIZE:
        raise ValueError(
            f"Expected vector size {VECTOR_SIZE}, "
            f"got {vectors.shape[1]}."
        )

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or infinity.")


def create_collection(recreate=False):
    exists = client.collection_exists(COLLECTION_NAME)

    if exists:
        if not recreate:
            print("Collection already exists.")
            return

        client.delete_collection(
            collection_name=COLLECTION_NAME,
        )
        print("Old collection deleted.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print("Collection created.")


def upload_chunks(chunks, embeddings):
    validate_index_data(chunks, embeddings)

    points = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):
        points.append(
            PointStruct(
                id=index,
                vector=embedding.tolist(),
                payload={
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "document": "ust_manual.pdf",
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    print(f"Uploaded {len(points)} chunks.")


def search(query_embedding, top_k=3):
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=top_k,
        with_payload=True,
    )

    results = []

    for point in response.points:
        results.append({
            "score": point.score,
            "page": point.payload["page"],
            "text": point.payload["text"],
            "document": point.payload["document"],
        })

    return results
