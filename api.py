from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from main import answer_question
from retrieval import load_embedding_model


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
    load_embedding_model()
    yield


app = FastAPI(
    title="Enterprise RAG",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return answer_question(request.question)
