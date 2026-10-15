![Status](https://img.shields.io/badge/status-foundation-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Architecture](https://img.shields.io/badge/docs-C4-orange)

# Aegis AI Platform

> **A Production-Grade Agent Infrastructure & Evaluation Platform**

**Pronunciation:** *EE-jis*

---

## Mission

Aegis AI Platform is a production-grade AI infrastructure platform designed to build, deploy, observe, evaluate, and operate reliable AI agents at scale.

This project follows a documentation-first engineering approach inspired by Amazon's Working Backwards process and modern platform engineering practices.

---

## Why Aegis?

Modern AI applications often struggle with:

- Hallucinations
- Poor observability
- Missing evaluation pipelines
- Weak reliability
- Difficult production deployments

Aegis aims to solve these problems through a platform-first architecture.

---

## Core Technologies

- Python
- FastAPI
- LangGraph
- OpenAI API
- Kafka
- PostgreSQL
- Qdrant
- Kubernetes
- OpenTelemetry
- Prometheus
- Grafana
- Docker
- AWS
- Azure

---

## Project Status

### Phase 1 – Foundation

- [x] Repository created
- [x] Documentation initialized
- [x] PRD
- [x] SRS
- [x] C4 Context Diagram
- [x] ADR Template
- [ ] FastAPI Foundation
- [ ] Agent Engine
- [ ] RAG
- [ ] Kafka
- [ ] Kubernetes

---

## Repository Structure

```text
docs/
services/
infrastructure/
evaluation/
tests/
scripts/
```

---

## Engineering Principles

Every feature follows this workflow.

1. Requirement
2. Architecture
3. Task Planning
4. Implementation
5. Testing
6. Observability
7. Documentation
8. Git Commit

---

## Run Locally

Install the project dependencies:

```powershell
uv sync
```

Create a `.env` file in the repository root and configure one provider. For
OpenAI:

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-openai-api-key
```

Or for NVIDIA:

```dotenv
LLM_PROVIDER=nvidia
NVIDIA_MODEL=openai/gpt-oss-20b
NVIDIA_API_KEY=your-nvidia-api-key
```

Do not commit `.env` or API keys. Start the API from the repository root:

```powershell
uv run uvicorn aegis_platform.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the API
documentation, or check [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
to verify the service is running.

### Live Smoke Tests

With the API running, send a request to `/agent/run`:

```powershell
$body = @{ message = "Say hello in one sentence." } |
		ConvertTo-Json

Invoke-RestMethod `
		-Uri "http://127.0.0.1:8000/agent/run" `
		-Method Post `
		-ContentType "application/json" `
		-Body $body
```

Check the Server-Sent Events response from `/agent/stream` with `curl.exe`:

```powershell
curl.exe -N `
	-X POST `
	"http://127.0.0.1:8000/agent/stream" `
	-H "Content-Type: application/json" `
	--data-raw '{"message":"Explain streaming in two sentences."}'
```

Both requests call the configured LLM provider and may incur API usage. The
current `/agent/stream` implementation waits for a completed response, then
emits it as spaced word chunks. This smoke test checks the SSE response and
incremental delivery, not provider time-to-first-token streaming.

---

## License

MIT License.
