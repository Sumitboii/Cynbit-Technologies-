# Key Decisions Log

## Decision 1: Opportunity Selection (Phase A)

**Date**: 2026-10-23
**Decision**: Selected Opportunity #1 - Revio v2: Reputation Intelligence Engine
**Scoring**: Impact 5, Feasibility 5, Data 5, Differentiation 5, Risk 3 = **23/25** (Top)

**Justification**:
- Revio is an existing Cynbit product; this upgrade directly serves Cynbit's customer base
- Hinglish/code-mixed sentiment analysis is a gap in the Indian market
- Python/FastAPI/Streamlit stack matches Cynbit's technology stack
- Synthetic data is trivial to generate for reviews
- Commodity models (TF-IDF baseline, MiniLM embeddings) are sufficient for PoC

**Alternatives Considered**:
- #2 TenantEase late-payment prediction: Good but less differentiating (tabular ML is commoditized)
- #3 Inbound-lead scoring: Useful but less tied to existing product
- #4 RFP assistant: Interesting but data-heavy, requires case study corpus
- #5-8: Various but lower feasibility or differentiation

## Decision 2: Architecture Approach

**Date**: 2026-10-23
**Decision**: Hybrid architecture: keyword + embeddings + rules

**Justification**:
- Keyword matching is fast, interpretable, and handles explicit mentions
- Embeddings (MiniLM) handle semantic similarity across languages
- Rule-based crisis detection is safer than ML-only (lower false positive rate)
- Z-score anomaly detection catches location-level spikes

## Decision 3: LLM Integration

**Date**: 2026-10-23
**Decision**: Optional LLM with template-based fallback

**Justification**:
- PoC must run without API keys (offline demo requirement)
- Template fallback ensures the demo never breaks
- LLM improves reply quality when available but isn't critical for PoC
