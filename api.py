from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

from main import answer_question
from retrieval import load_embedding_model

from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from vector_store import COLLECTION_NAME, get_index_point_count


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("Question must not be empty.")

        return value


class Source(BaseModel):
    source_id: int
    document: str
    page: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    status: Literal["answered", "insufficient_context"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_ready = False

    load_embedding_model()
    app.state.model_ready = True

    try:
        yield
    finally:
        app.state.model_ready = False


app = FastAPI(
    title="Enterprise RAG",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return answer_question(request.question)


@app.get("/health")
def health(request: Request):
    model_ready = getattr(
        request.app.state,
        "model_ready",
        False,
    )

    if not model_ready:
        raise HTTPException(
            status_code=503,
            detail="Embedding model is not ready.",
        )

    try:
        point_count = get_index_point_count()
    except (ResponseHandlingException, UnexpectedResponse):
        raise HTTPException(
            status_code=503,
            detail="Could not check the Qdrant index.",
        ) from None

    if point_count is None:
        raise HTTPException(
            status_code=503,
            detail="Qdrant collection does not exist.",
        )

    if point_count == 0:
        raise HTTPException(
            status_code=503,
            detail="Qdrant collection is empty.",
        )

    return {
        "status": "ready",
        "model_ready": True,
        "collection": COLLECTION_NAME,
        "points_count": point_count,
    }
