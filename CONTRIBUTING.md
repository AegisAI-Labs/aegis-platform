# Contributing to Aegis AI Platform

Thank you for your interest in contributing to **Aegis AI Platform**.

Aegis is a production-grade AI infrastructure project focused on building reliable, observable, and scalable AI agent systems. The project follows a **documentation-first engineering workflow**, meaning architecture and requirements are established before implementation.

---

# Engineering Workflow

Every significant feature follows this lifecycle:

PRD → RFC → SRS → C4 → ADR → Implementation → Tests → Documentation → Merge

Before writing code, the corresponding documentation should exist.

---

# Repository Structure

```text
docs/
services/
infrastructure/
evaluation/
tests/
scripts/
```

# Prerequisites

- Python 3.12 (the repository version is pinned in `.python-version`).

---

# Development Process

1. Identify the problem.
2. Update or create an RFC.
3. Update the SRS if behavior changes.
4. Update C4 diagrams if architecture changes.
5. Record architectural decisions in an ADR.
6. Implement the change.
7. Add tests.
8. Update documentation.

---

# Branch Naming

Use descriptive branch names.

Examples:

```
feature/fastapi-foundation
feature/langgraph-agent
feature/kafka-event-bus

fix/retry-logic
fix/health-endpoint

docs/update-prd
docs/c4-diagrams
```

---

# Commit Message Convention

Follow Conventional Commit style.

| Prefix | Purpose |
|---------|---------|
| feat | New feature |
| fix | Bug fix |
| docs | Documentation |
| test | Tests |
| refactor | Internal improvements |
| chore | Maintenance |

Examples:

```text
docs: establish Aegis architecture foundation
feat: add FastAPI gateway
feat: introduce LangGraph workflow
fix: improve retry logic
test: add API integration tests
```

---

# Pull Request Checklist

Before opening a Pull Request, verify:

- [ ] RFC exists (if required)
- [ ] ADR created (if architecture changed)
- [ ] SRS updated
- [ ] C4 updated
- [ ] Tests added
- [ ] Documentation updated
- [ ] Commit history is clean

---

# Coding Standards

## Python

- Type hints required.
- Prefer async where appropriate.
- Keep functions focused and small.
- Write meaningful variable names.

## API Design

- REST-first.
- Consistent HTTP status codes.
- Structured error responses.

## Logging

- Structured logs only.
- Never log secrets.
- Include request correlation IDs.

---

# Testing Expectations

Testing should focus on behavior.

Priority:

1. Unit Tests
2. Integration Tests
3. End-to-End Tests
4. AI Evaluations

Critical business workflows should always be tested.

---

# Documentation Standards

Every major change should update relevant documentation.

Possible updates include:

- PRD
- RFC
- SRS
- C4
- ADR
- README
- Runbooks

Documentation is considered part of the implementation.

---

# Engineering Principles

Aegis values:

- Reliability over cleverness.
- Documentation before implementation.
- Observability by default.
- Security by default.
- Decisions with clear traceability.

Thank you for helping improve Aegis AI Platform.