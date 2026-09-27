# AEGIS-C4-001
>Context Container Component Code - This documentation talks about the architecture review.How is it structured?

>Context - System Boundaries, Container - Application Building blocks.
# C4 Model – Level 1 Context Diagram

Version: 1.0

---

## System Context

```text
User
 │
 ▼
Aegis AI Platform
 │
 ├── OpenAI
 ├── AWS
 └── Azure
```

---

## External Systems

| System | Purpose |
|---------|---------|
| OpenAI | Language model reasoning |
| AWS | Cloud deployment |
| Azure | Cloud deployment |

---

## Users

Primary user:

- Platform Engineer

Secondary users:

- Backend Engineer
- ML Engineer