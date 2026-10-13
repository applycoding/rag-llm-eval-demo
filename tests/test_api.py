from fastapi.testclient import TestClient

from ragqa.api.app import create_app
from ragqa.llm import ABSTENTION


def test_health_ingest_and_query(settings) -> None:
    settings = settings.model_copy(update={"auto_ingest": True})
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert body["embedder"] == "hash"
        assert body["llm"] == "extractive"
        assert body["chunks"] > 0

        answer = client.post(
            "/query",
            json={
                "question": "What was the crown ring cast in by the Barbizon Glassworks?",
                "pipeline": "langchain",
                "k": 4,
            },
        )
        assert answer.status_code == 200
        payload = answer.json()
        assert "naval bronze" in payload["answer"]
        assert payload["sources"]

        missing = client.post(
            "/query",
            json={"question": "Who invented the telephone in this archive?", "pipeline": "llamaindex"},
        )
        assert missing.status_code == 200
        assert missing.json()["answer"] == ABSTENTION

        rejected = client.post("/query", json={"question": ""})
        assert rejected.status_code == 422

        unknown = client.post("/query", json={"question": "When?", "pipeline": "other"})
        assert unknown.status_code == 422


def test_ingest_endpoint_adds_a_document(settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/ingest",
            json={
                "documents": [
                    {
                        "document_id": "added",
                        "title": "Added",
                        "text": "The night clerk keeps a brass key stamped with the number 77.",
                    }
                ]
            },
        )
        assert response.status_code == 200
        assert response.json()["chunks"] >= 1
        answer = client.post(
            "/query",
            json={"question": "What number is stamped on the brass key?", "pipeline": "tfidf"},
        )
        assert "77" in answer.json()["answer"]


def test_missing_directory_is_not_found(settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post("/ingest", json={"directory": "data/does-not-exist"})
        assert response.status_code == 404
