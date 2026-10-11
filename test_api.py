from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def setup_api(monkeypatch):
    with (
        patch("google.genai.Client"),
        patch("qdrant_client.QdrantClient"),
    ):
        import api
        import main

    model = Mock()
    model.encode_query.return_value = [0.1, 0.2, 0.3]

    search_mock = Mock(return_value=[
        {
            "document": "ust_manual.pdf",
            "page": 30,
            "text": "Absorbent materials include sand and sawdust.",
            "score": 0.8,
        }
    ])

    generation_mock = Mock(
        return_value="Sand and sawdust. [Source 1, page 30]"
    )

    monkeypatch.setattr(api, "load_embedding_model", lambda: model)
    monkeypatch.setattr(main, "load_embedding_model", lambda: model)
    monkeypatch.setattr(main, "search", search_mock)
    monkeypatch.setattr(main, "generate_answer", generation_mock)

    with TestClient(api.app) as client:
        yield client, model, search_mock, generation_mock


def test_ask_success(setup_api):
    client, model, search_mock, generation_mock = setup_api

    response = client.post(
        "/ask",
        json={"question": "  What absorbs a fuel spill?  "},
    )

    assert response.status_code == 200

    assert response.json() == {
        "answer": "Sand and sawdust. [Source 1, page 30]",
        "sources": [
            {
                "source_id": 1,
                "document": "ust_manual.pdf",
                "page": 30,
                "text": "Absorbent materials include sand and sawdust.",
                "score": 0.8,
            }
        ],
        "status": "answered",
    }

    model.encode_query.assert_called_once_with(
        "What absorbs a fuel spill?",
        normalize_embeddings=True,
    )

    search_mock.assert_called_once()
    generation_mock.assert_called_once()

    context = generation_mock.call_args.kwargs["context"]
    assert "[Source 1, page 30]" in context
    assert "Absorbent materials include sand and sawdust." in context


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": None},
        {"question": 123},
    ],
)
def test_ask_rejects_invalid_question(setup_api, payload):
    client, model, search_mock, generation_mock = setup_api

    response = client.post("/ask", json=payload)

    assert response.status_code == 422

    model.encode_query.assert_not_called()
    search_mock.assert_not_called()
    generation_mock.assert_not_called()


def test_ask_insufficient_context(setup_api):
    client, model, search_mock, generation_mock = setup_api

    search_mock.return_value = [
        {
            "document": "ust_manual.pdf",
            "page": 30,
            "text": "Some unrelated fragment.",
            "score": 0.1,
        }
    ]

    response = client.post(
        "/ask",
        json={"question": "Who founded Microsoft?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": (
            "The answer cannot be determined "
            "from the provided documents."
        ),
        "sources": [],
        "status": "insufficient_context",
    }

    search_mock.assert_called_once()
    generation_mock.assert_not_called()
