# AEGIS-GLOSSARY-001

# Aegis Engineering Glossary

Version: 1.0

Status: Active

Owner: Kader Beevi

Purpose: Maintain a single source of truth for technical terms used throughout Aegis AI Platform.

---

# How to Use This Glossary

Each term includes:

- Definition
- Why it matters
- Where it is used in Aegis
- Interview tip

---

# A

## ADR (Architecture Decision Record)

**Definition**

A document that records an important architectural decision. Explain both the decision and the trade-offs.

**Why it matters**

Explains why a technical decision was made.

**Used in**

`docs/ADR/`


---

## Agent

**Definition**

A software component that uses an LLM together with tools, memory, and planning to complete tasks.

**Used in**

LangGraph Orchestrator.

---

## API Gateway

**Definition**

The entry point that receives requests before routing them to backend services.

**Used in**

FastAPI.

---

# C

## C4 Model

**Definition**

A four-level architecture visualization method.

Levels:

- Context
- Container
- Component
- Code

**Used in**

`docs/C4/`

---

## Circuit Breaker

**Definition**

A reliability pattern that prevents repeated failures from overwhelming a system.

**Used in**

Agent reliability.

---

## CQRS

**Definition**

Command Query Responsibility Segregation.

Separates reads from writes.

**Why it matters**

Improves scalability.

---

# D

## Distributed System

**Definition**

Multiple services working together across different processes or machines.

**Used in**

Entire Aegis architecture.

---

## Docker

**Definition**

Containerization platform.

**Used in**

Local development and deployment.

---

# E

## EDI (Engineering Decision Index)

**Definition**

Master traceability document connecting requirements, architecture, implementation, and deployment.

**Used in**

`docs/EDI/`

---

## Evaluation

**Definition**

Measuring AI system quality.

Metrics include:

- Correctness
- Faithfulness
- Latency
- Hallucination rate

---

# F

## FastAPI

**Definition**

A modern Python web framework.

**Why it matters**

Excellent async performance.

---

# G

## Grafana

**Definition**

Visualization platform for metrics and dashboards.

**Used in**

Observability.

---

# H

## Hallucination

**Definition**

An incorrect AI-generated answer presented confidently.

**Used in**

Evaluation framework.

---

# I

## Idempotency

**Definition**

Performing the same operation multiple times without changing the final result.

**Why it matters**

Critical for distributed systems.

---

# J

## JWT

**Definition**

JSON Web Token.

Used for authentication.

---

# K

## Kafka

**Definition**

Distributed event streaming platform.

**Used in**

Agent events.

Topics:

- requests
- responses
- evaluations
- errors

---

## Kubernetes

**Definition**

Container orchestration platform.

**Used in**

AKS and EKS deployments.

---

# L

## LangGraph

**Definition**

Framework for building stateful AI agent workflows.

**Used in**

Agent orchestration.

---

## LLM

**Definition**

Large Language Model.

Examples:

- GPT
- Claude

---

# M

## MCP (Model Context Protocol)

**Definition**

A protocol for connecting AI models to external tools and data sources.

---

# O

## OpenTelemetry

**Definition**

Standard for logs, metrics, and distributed tracing.

**Used in**

Observability.

---

# P

## PostgreSQL

**Definition**

Relational database.

**Used in**

Metadata storage.

---

## PRD

**Definition**

Product Requirements Document.

Answers:

> Why are we building this?

---

## Prometheus

**Definition**

Metrics collection system.

**Used in**

Monitoring.

---

# Q

## Qdrant

**Definition**

Vector database.

**Used in**

Semantic retrieval.

---

# R

## RAG (Retrieval-Augmented Generation)

**Definition**

Combining document retrieval with LLM reasoning.

Pipeline:

Documents → Embeddings → Retrieval → LLM

---

## Retry

**Definition**

Automatically repeating failed operations.

**Used in**

Reliability.

---

## RFC

**Definition**

Request for Comments.

Proposal document before implementation.

---

# S

## Scalability

**Definition**

Ability to handle increasing workload.

---

## SRS

**Definition**

Software Requirements Specification.

Answers:

> How should the software behave?

---

# T

## Timeout

**Definition**

Maximum waiting time before an operation fails.

---

## Tool Calling

**Definition**

Allowing an AI model to invoke external functions.

Examples:

- Calculator
- Search
- SQL Query

---

# V

## Vector Database

**Definition**

Database optimized for similarity search.

Examples:

- Qdrant
- pgvector

---

# W

## Workflow

**Definition**

Ordered sequence of execution steps.

Example:

Planner → Retriever → Tool → LLM → Evaluation

---

# Aegis Cheat Sheet

| Term | One-Sentence Answer |
|------|-----------------------|
| PRD | Defines why the product exists. |
| RFC | Proposes how a feature should be built. |
| ADR | Records the final architectural decision. |
| SRS | Defines software behavior. |
| C4 | Visualizes architecture. |
| Kafka | Event streaming platform. |
| LangGraph | Agent orchestration framework. |
| RAG | Retrieval plus LLM reasoning. |
| OpenTelemetry | Observability standard. |
| Idempotency | Safe repeated execution. |

---

# Future Terms

New terms should be added alphabetically.

Current Count: **27 Terms**

Target: **100+ Terms**