"""Safety pipeline around RAG retrieval and answer verification.

Used by the ``/rag/search``, ``/rag/query`` and ``/rag/verify`` endpoints:

    guardrails -> [query enhancement] -> vector search -> access control
    -> [cross-encoder reranking] -> PII masking

Access decisions use the access metadata stored with each chunk at ingestion
(``access_level``, ``allowed_roles``, ``organization_id``), so they survive restarts
and do not depend on the in-memory document registry.
"""

import logging
import time
from typing import Any

from aegis_platform.rag.access_control import (
    AccessLevel,
    DocumentAccessControl,
    DocumentMetadata,
    Role,
    UserContext,
)
from aegis_platform.rag.citation_tracker import SourceContext, get_citation_tracker
from aegis_platform.rag.hallucination_detector import get_hallucination_detector
from aegis_platform.rag.query_guardrails import get_query_guardrails
from aegis_platform.rag.service import RAGService

logger = logging.getLogger(__name__)

# Access control and reranking discard hits, so search deeper than top_k.
CANDIDATE_MULTIPLIER = 4
MAX_CANDIDATES = 100


class QueryRejectedError(ValueError):
    """Raised when a query fails the guardrails."""

    def __init__(self, violations: list[str], risk_level: str) -> None:
        super().__init__("Query rejected by guardrails")
        self.violations = violations
        self.risk_level = risk_level


def access_metadata(
    access_level: AccessLevel,
    allowed_roles: list[Role],
    organization_id: str | None,
) -> dict[str, Any]:
    """Access fields to store in a chunk's metadata at ingestion."""
    return {
        "access_level": access_level.value,
        "allowed_roles": [role.value for role in allowed_roles],
        "organization_id": organization_id,
    }


def document_metadata_from_payload(payload: dict[str, Any]) -> DocumentMetadata:
    """Rebuild access metadata from a vector store payload.

    Chunks ingested without access fields default to INTERNAL. Unknown access
    levels are treated as CONFIDENTIAL so a corrupt payload never widens access.
    """
    inner = payload.get("metadata") or {}
    doc_id = str(payload.get("doc_id") or payload.get("document_id") or "unknown")

    try:
        level = AccessLevel(inner.get("access_level", AccessLevel.INTERNAL.value))
    except ValueError:
        level = AccessLevel.CONFIDENTIAL

    roles = []
    for value in inner.get("allowed_roles") or []:
        try:
            roles.append(Role(value))
        except ValueError:
            continue

    return DocumentMetadata(
        doc_id=doc_id,
        title=doc_id,
        access_level=level,
        allowed_roles=roles,
        organization_id=inner.get("organization_id"),
    )


def _hit_key(hit: dict[str, Any]) -> str:
    return hit.get("chunk_id") or f"{hit['metadata'].get('doc_id', '')}:{hit['text']}"


def _pii_detector() -> Any:
    from aegis_platform.rag.pii_detector import get_pii_detector

    try:
        return get_pii_detector()
    except (ImportError, OSError) as e:
        # Fail closed: never return passages that could not be masked.
        raise RuntimeError(f"PII masking unavailable: {e}") from e


async def _expanded_queries(query: str) -> list[str]:
    from aegis_platform.rag.llm_query_enhancer import get_llm_query_enhancer
    from aegis_platform.rag.query_enhancer_base import EnhancementStrategy

    guardrails = get_query_guardrails()
    enhanced = await get_llm_query_enhancer().enhance(query, EnhancementStrategy.EXPANSION)

    queries = [query]
    for candidate in enhanced.expanded_queries:
        if candidate.lower() in (q.lower() for q in queries):
            continue
        # LLM output is untrusted input: validate every expansion.
        if guardrails.validate_query(candidate).is_safe:
            queries.append(candidate)
        else:
            logger.warning("Dropped unsafe query expansion: %r", candidate)
    return queries


async def _rerank(query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    from aegis_platform.rag.cross_encoder_reranker import get_cross_encoder_reranker
    from aegis_platform.rag.reranker_base import RawResult

    ranked = await get_cross_encoder_reranker().rerank(
        query,
        [
            RawResult(
                result_id=str(i),
                text=hit["text"],
                similarity_score=hit["similarity_score"],
                source_id=hit["metadata"].get("doc_id"),
            )
            for i, hit in enumerate(hits)
        ],
        top_k=top_k,
    )
    return [{**hits[int(r.result_id)], "rerank_score": r.relevance_score} for r in ranked]


async def secure_retrieve(
    service: RAGService,
    query: str,
    user: UserContext,
    top_k: int = 5,
    similarity_threshold: float | None = None,
    rerank: bool = False,
    enhance: bool = False,
) -> dict[str, Any]:
    """Search with every safety layer applied.

    Raises:
        QueryRejectedError: If the query fails the guardrails
        ValueError: If the query or top_k is invalid
        RuntimeError: If search, reranking or PII masking fails
    """
    start = time.perf_counter()

    verdict = get_query_guardrails().validate_query(query)
    if not verdict.is_safe:
        logger.warning(
            "Query rejected for user=%s risk=%s violations=%s",
            user.user_id,
            verdict.risk_level,
            verdict.violations,
        )
        raise QueryRejectedError(verdict.violations, verdict.risk_level)

    queries = await _expanded_queries(query) if enhance else [query]

    candidate_k = min(MAX_CANDIDATES, top_k * CANDIDATE_MULTIPLIER)
    merged: dict[str, dict[str, Any]] = {}
    for q in queries:
        found = await service.search_similar(
            query=q, top_k=candidate_k, similarity_threshold=similarity_threshold
        )
        for hit in found["results"]:
            key = _hit_key(hit)
            if key not in merged or hit["similarity_score"] > merged[key]["similarity_score"]:
                merged[key] = hit
    candidates = sorted(merged.values(), key=lambda h: h["similarity_score"], reverse=True)

    acl = DocumentAccessControl.get_instance()
    allowed = [
        hit
        for hit in candidates
        if acl.evaluate(user, document_metadata_from_payload(hit["metadata"])).is_allowed
    ]
    logger.info(
        "Access control: user=%s role=%s kept %d of %d hits",
        user.user_id,
        user.role.value,
        len(allowed),
        len(candidates),
    )

    if rerank and allowed:
        allowed = await _rerank(query, allowed, top_k)
    allowed = allowed[:top_k]

    pii = _pii_detector()
    results = []
    for hit in allowed:
        # The payload duplicates the chunk text; drop it so unmasked text never leaks.
        metadata = {k: v for k, v in hit["metadata"].items() if k != "text"}
        text = pii.mask_pii(hit["text"]) if hit["text"].strip() else hit["text"]
        results.append({**hit, "text": text, "metadata": metadata})

    return {"results": results, "latency_ms": (time.perf_counter() - start) * 1000}


async def verify_answer(
    query: str, answer: str, sources: list[tuple[str, str, str, float]]
) -> dict[str, Any]:
    """Score an LLM answer against the passages it was given.

    Args:
        query: User query
        answer: LLM answer
        sources: (doc_id, title, text, score) for each passage

    Raises:
        ValueError: If the answer is blank
    """
    start = time.perf_counter()
    tracker = get_citation_tracker()

    if sources:
        context = await tracker.track_retrieval_context(
            query=query,
            retrieved_docs=[DocumentMetadata(doc_id=d, title=t) for d, t, _, _ in sources],
            retrieval_scores=[s for _, _, _, s in sources],
            passages=[text for _, _, text, _ in sources],
        )
        citations = await tracker.extract_citations(answer, context)
    else:
        context = SourceContext(query=query)
        citations = []

    confidence = await tracker.calculate_response_confidence(citations, context)
    score = await get_hallucination_detector().detect_hallucinations(answer, context, citations)

    return {
        "citations": [
            {
                "claim": c.claim,
                "doc_ids": [s.doc_id for s in c.sources],
                "confidence": c.confidence,
            }
            for c in citations
        ],
        "confidence": confidence,
        "risk_level": score.risk_level,
        "risk_score": score.risk_score,
        "patterns_detected": [p.value for p in score.patterns_detected],
        "explanation": score.explanation,
        "latency_ms": (time.perf_counter() - start) * 1000,
    }
