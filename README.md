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

### RAG Documentation

- [RAG API reference](docs/API/AEGIS-API-RAG.md) — `/rag` endpoints and the safety module APIs
- [RAG safety configuration](docs/GUIDES/RAG-SAFETY-CONFIGURATION.md) — installation, settings, troubleshooting
- [RAG safety usage](docs/GUIDES/RAG-SAFETY-USAGE.md) — runnable examples for each safety layer

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
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Set PYTHONPATH to locate the aegis_platform module
$env:PYTHONPATH="src"

# Start the API server
python -m uvicorn aegis_platform.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the API
documentation, or check [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
to verify the service is running.

### Pre-commit Hooks

This repository uses `pre-commit` hooks to automatically check code quality and
consistency before each commit. Hooks are configured in `.pre-commit-config.yaml`
and run automatically when you commit.

#### Install Pre-commit Hooks

After cloning or setting up the project, install the pre-commit hooks:

```powershell
# Ensure pre-commit is installed (already in dev dependencies)
uv sync

# Install the git hooks
pre-commit install
```

You should see output like:
```
pre-commit installed at .git/hooks/pre-commit
```

After installation, hooks run automatically on every `git commit`. If any hook
fails, your commit is rejected and you can fix the issues and try again.

#### What Hooks Run

The following hooks run automatically on each commit:

| Hook | Purpose |
|------|---------|
| `trailing-whitespace` | Removes trailing whitespace from files |
| `end-of-file-fixer` | Ensures files end with a single newline |
| `check-yaml` | Validates YAML file syntax |
| `check-json` | Validates JSON file syntax |
| `check-added-large-files` | Prevents committing files larger than 5MB |
| `ruff` | Lints Python code and auto-fixes issues |
| `ruff-format` | Auto-formats Python code for consistency |

#### Run Hooks Manually

To run all hooks on all files without committing:

```powershell
pre-commit run --all-files
```

To run hooks on only staged files:

```powershell
pre-commit run
```

#### Bypass Hooks (When Necessary)

If you need to bypass hooks temporarily (not recommended), use `--no-verify`:

```powershell
git commit --no-verify -m "Your commit message"
```

> **Note:** Only use `--no-verify` in exceptional cases. Hooks exist to maintain
> code quality and consistency.

#### Troubleshooting

**Problem:** `` `pre-commit` not found. Did you forget to activate your virtualenv? ``

**Solution:** The pre-commit hook can't find the Python interpreter. This usually
happens if:
1. Dependencies weren't installed: Run `uv sync`
2. The hook was installed in a different environment: Run `pre-commit install` again

If issues persist, regenerate the hooks:

```powershell
pre-commit clean
pre-commit install
```

**Problem:** Hooks modify files that I didn't intend to change

**Solution:** This is expected behavior. Hooks like `ruff-format` and
`trailing-whitespace` auto-fix issues. Review the changes and re-stage if needed:

```powershell
# Review the changes
git diff

# Re-stage the fixed files
git add .

# Try committing again
git commit -m "Your commit message"
```

### Run with Docker

Make sure Docker Desktop is running. From the repository root, build the image
and start the container with the provider settings from `.env`:

```powershell
docker build -t aegis-platform .
docker run --rm --name aegis-platform -p 8000:8000 --env-file .env aegis-platform
```

The API is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs),
and the health check is at [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).
Press `Ctrl+C` to stop and remove the container. The `.dockerignore` excludes
`.env` from the image; `--env-file` supplies its settings only when the
container starts.

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
