# Research Synthesizer

Research Synthesizer is a multi-agent AI backend that turns a research question into a sourced synthesis report.

The current Week 1 scope focuses on a stable LangGraph + FastAPI foundation:

- Search Agent retrieves academic sources.
- Reader Agent summarizes sources with structured output.
- Contradiction Agent identifies disagreements and consensus gaps.
- Synthesis Agent produces the final Markdown-ready report.
- FastAPI exposes the workflow through `/research`.

## Architecture

```text
User question
    |
    v
Search Agent
    |
    v
Reader Agent
    |
    v
Contradiction Agent
    |
    v
Synthesis Agent
    |
    v
Synthesis report
```

## Tech Stack

- LangGraph for agent orchestration
- LangChain + Groq for LLM calls
- Tavily for academic web search
- Pydantic for structured agent outputs
- FastAPI for the backend API
- Langfuse for optional observability
- pytest for API-level regression tests

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add your keys to `.env`:

```env
GROQ_API_KEY=...
TAVILY_API_KEY=...
```

## Run The API

```bash
source .venv/bin/activate
python main.py
```

By default, the API runs on:

```text
http://127.0.0.1:8000
```

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Research Request

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question":"How to reduce hallucination in RAG systems?"}'
```

## Tests

```bash
source .venv/bin/activate
python -m pytest
```

The Week 1 tests focus on API stability, request validation, response mapping, and controlled failure handling.

## Week 1 Engineering Notes

- Heavy LangChain, Groq, Tavily, and Langfuse setup is lazy-loaded so FastAPI startup remains fast.
- Environment variables are centralized in `src/core/config.py`.
- Langfuse is disabled by default for local development and can be enabled with `LANGFUSE_ENABLED=true`.
- `/research` converts configuration errors to `503` and upstream pipeline failures to `502`.
