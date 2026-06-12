from fastapi.testclient import TestClient

from src.api.routes import app
from src.models.paper import SynthesisReport


client = TestClient(app)


class FakeGraph:
    def __init__(self, result: dict | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.last_input = None
        self.last_config = None

    async def ainvoke(self, input: dict, config: dict):
        self.last_input = input
        self.last_config = config
        if self.error:
            raise self.error
        return self.result or {}


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_rejects_blank_question():
    response = client.post("/research", json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty"


def test_research_maps_synthesis_report(monkeypatch):
    report = SynthesisReport(
        question="How does RAG reduce hallucination?",
        summary="RAG grounds generation in retrieved evidence.",
        contradictions=["Retrieval quality varies across domains."],
        open_questions=["How should retrievers be adapted per domain?"],
        sources=["https://example.com/paper"],
    )
    fake_graph = FakeGraph(result={"synthesis_report": report})

    monkeypatch.setattr("src.api.routes.get_research_graph", lambda: fake_graph)
    monkeypatch.setattr("src.api.routes.get_langfuse_handler", lambda session_id: None)

    response = client.post(
        "/research",
        json={"question": "  How does RAG reduce hallucination?  "},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == report.question
    assert body["summary"] == report.summary
    assert body["contradictions"] == report.contradictions
    assert body["open_questions"] == report.open_questions
    assert body["sources"] == report.sources
    assert fake_graph.last_input["question"] == "How does RAG reduce hallucination?"


def test_research_returns_502_when_report_is_missing(monkeypatch):
    fake_graph = FakeGraph(result={"synthesis_report": None})

    monkeypatch.setattr("src.api.routes.get_research_graph", lambda: fake_graph)
    monkeypatch.setattr("src.api.routes.get_langfuse_handler", lambda session_id: None)

    response = client.post("/research", json={"question": "What is RAG?"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Research pipeline did not produce a synthesis report"


def test_research_returns_503_for_configuration_error(monkeypatch):
    fake_graph = FakeGraph(error=RuntimeError("GROQ_API_KEY is not set"))

    monkeypatch.setattr("src.api.routes.get_research_graph", lambda: fake_graph)
    monkeypatch.setattr("src.api.routes.get_langfuse_handler", lambda session_id: None)

    response = client.post("/research", json={"question": "What is RAG?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "GROQ_API_KEY is not set"
