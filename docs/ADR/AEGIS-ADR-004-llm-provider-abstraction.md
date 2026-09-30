# ADR-004: Introduce an LLM Provider Abstraction

- Status: Accepted
- Date: 2026-09-30

## Context

Aegis is intended to support multiple LLM providers.

Directly coupling the Agent Service or LangGraph workflows to a
specific provider SDK would make provider replacement, comparison,
testing, and failover more difficult.

## Decision

Aegis will introduce a provider-independent LLM interface.

The architecture will be:

    Agent
      ↓
    LLM Interface
      ↓
    Provider Adapter
      ↓
    Provider SDK/API

The Agent Service and LangGraph orchestration layer must not import
provider-specific SDKs directly.

## Initial Interface

The abstraction will initially support:

- generate()
- stream()

Additional capabilities will be added only when required.

## Consequences

### Positive

- Provider independence
- Easier testing
- Easier provider comparison
- Reduced coupling
- Future failover support
- Cleaner architecture

### Negative

- Additional abstraction layer
- Some provider-specific capabilities may not map perfectly
- Common interfaces must evolve as capabilities expand

## Alternatives Considered

### Direct SDK Integration

Rejected because it tightly couples Aegis to a provider.

### One Provider Only

Rejected because provider portability and comparative evaluation
are explicit Aegis goals.

### Universal Feature Interface

Deferred because attempting to normalize every provider capability
up front would create unnecessary abstraction complexity.

## Future Considerations

Provider routing, fallback, caching, cost optimization, and advanced
model selection may be introduced after real usage and measurement.
