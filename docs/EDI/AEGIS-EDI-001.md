# AEGIS-EDI-001

# Engineering Decision Index
>This makes the project rather than a "documentation collection" to a "traceable engineering knowledge base"

Version: 1.0

Status: Active

Owner: Kader Beevi

Purpose: Provide complete traceability between product decisions, architecture decisions, implementation, testing, and documentation.

---

# What is EDI?

Engineering Decision Index (EDI) is the master reference that connects every important engineering artifact in Aegis.

Instead of searching through multiple documents, engineers can trace a feature from the original business requirement to the final implementation.

---

# Traceability Flow

PRD → RFC → SRS → C4 → ADR → GitHub Issue → Commit → Pull Request → Tests → Deployment

---

# Decision Register

| Decision | Status | Owner | Latest Stage |
|----------|--------|-------|--------------|
| Platform Foundation | Implemented | Kader Beevi | Documentation |
| FastAPI Foundation | Planned | Kader Beevi | RFC |
| LangGraph Engine | Planned | Kader Beevi | RFC |
| RAG Architecture | Planned | Kader Beevi | RFC |

---

# Cross-Reference Matrix

| Artifact | Related Documents |
|----------|-------------------|
| PRD-001 | RFC-001, SRS-001 |
| RFC-001 | PRD-001, SRS-001, C4-001 |
| SRS-001 | RFC-001 |
| C4-001 | RFC-001 |
| ADR-000 | Future ADRs |

---

# Feature Traceability

## Platform Foundation

| Stage | Reference |
|--------|-----------|
| PRD | AEGIS-PRD-001 |
| RFC | AEGIS-RFC-001 |
| SRS | AEGIS-SRS-001 |
| C4 | AEGIS-C4-001 |
| ADR | Template Created |
| Issue | TBD |
| Commit | Commit #1 |
| Tests | N/A |
| Deployment | N/A |

---

# Implementation Tracker

| Feature | Issue | Commit | Status |
|----------|-------|--------|--------|
| Documentation Foundation | TBD | Commit #1 | Completed |
| FastAPI Setup | TBD | Pending | Planned |
| LangGraph | TBD | Pending | Planned |
| Kafka | TBD | Pending | Planned |

---

# Review History

| Date | Event | Outcome |
|------|-------|---------|
| 2026-09-27 | Project Created | EDI Initialized |

---

# Engineering Rule

Every accepted RFC must eventually be linked to:

- PRD
- SRS
- C4
- ADR
- GitHub Issue
- Commit
- Tests
- Deployment

No major architectural decision should exist without traceability.