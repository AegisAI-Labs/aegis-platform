# AEGIS-RFC-REVIEW-CHECKLIST
>Used as a checklist to determines if the feature is ready to implement in a systematic way.

Version: 1.0

Purpose: Ensure every RFC meets production-grade engineering standards before implementation.

---

# RFC Review Checklist

## 1. Problem Definition

- [ ] The problem is clearly described.
- [ ] The business or engineering impact is explained.
- [ ] The scope is well defined.
- [ ] Out-of-scope items are listed.

---

## 2. Goals & Success Criteria

- [ ] Goals are measurable.
- [ ] Success metrics are defined.
- [ ] Non-goals are documented.

---

## 3. Technical Design

- [ ] The proposed solution is explained.
- [ ] Architecture diagrams are included (if applicable).
- [ ] Components and interactions are documented.
- [ ] Data flow is described.

---

## 4. Alternatives

- [ ] At least one alternative solution was considered.
- [ ] Reasons for rejecting alternatives are documented.
- [ ] Trade-offs are clearly explained.

---

## 5. Security Review

- [ ] Authentication requirements are considered.
- [ ] Authorization requirements are considered.
- [ ] Secrets management is addressed.
- [ ] Sensitive data handling is documented.

---

## 6. Reliability Review

- [ ] Failure scenarios are identified.
- [ ] Retry strategy is documented.
- [ ] Timeout strategy is documented.
- [ ] Idempotency is considered.
- [ ] Recovery procedures are defined.

---

## 7. Performance Review

- [ ] Expected latency is documented.
- [ ] Expected throughput is documented.
- [ ] Scaling approach is explained.
- [ ] Performance risks are identified.

---

## 8. Observability Review

- [ ] Logging requirements are defined.
- [ ] Metrics are identified.
- [ ] Distributed tracing is planned.
- [ ] Dashboards can be created.

---

## 9. Operational Review

- [ ] Deployment approach is documented.
- [ ] Rollback strategy is defined.
- [ ] Runbook requirements are identified.
- [ ] Monitoring requirements are listed.

---

## 10. Testing Review

- [ ] Unit testing strategy exists.
- [ ] Integration testing strategy exists.
- [ ] Evaluation strategy exists (if AI feature).
- [ ] Load testing requirements are identified.

---

## 11. Documentation Review

- [ ] PRD references are included.
- [ ] SRS references are included.
- [ ] C4 diagrams are updated.
- [ ] ADR requirements are identified.
- [ ] README changes are identified.

---

## 12. Final Decision

RFC Status:

- [ ] Draft
- [ ] Proposed
- [ ] Accepted
- [ ] Implemented
- [ ] Deprecated

Approved By:

- Author:
- Reviewer:
- Date:

---

# Aegis Engineering Rule

> No implementation begins until every critical checklist item has been reviewed.