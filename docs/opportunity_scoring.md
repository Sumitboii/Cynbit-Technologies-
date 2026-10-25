# Opportunity Scoring Analysis

## AI/ML Opportunities for Cynbit Technologies

**Date**: October 23, 2026
**Analyst**: Sumit Kumar Singh (AI/ML Engineer Intern Applicant)

## Scoring Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Impact | 1-5 | Business value to Cynbit and its customers |
| Feasibility | 1-5 | Buildable in 2 days with available tools |
| Data Availability | 1-5 | Can we generate or access sufficient data? |
| Differentiation | 1-5 | How unique is this in the Indian market? |
| Risk (inverted) | 1-5 | Lower risk = higher score |

## Scored Opportunities

| # | Opportunity | Area | Impact | Feasibility | Data | Diff. | Risk | Total |
|---|-------------|------|--------|-----------|------|-------|------|-------|
| 1 | **Revio v2: Hinglish sentiment + crisis + grounded replies** | Product | 5 | 5 | 5 | 5 | 3 | **23** |
| 2 | TenantEase: late-payment risk prediction | Product | 4 | 4 | 3 | 3 | 4 | 18 |
| 3 | Sales: inbound-lead qualification agent | Internal | 4 | 3 | 3 | 3 | 3 | 16 |
| 4 | RFP assistant grounded in case studies | Internal | 4 | 3 | 2 | 4 | 3 | 16 |
| 5 | Cynbit Studio: adaptive placement (IRT/bandit) | Product | 4 | 3 | 3 | 4 | 3 | 17 |
| 6 | Cynbit Studio: AI code-review feedback quality | Product | 3 | 3 | 3 | 3 | 3 | 15 |
| 7 | BitElevate: learner drop-off prediction | Product | 3 | 4 | 3 | 3 | 3 | 16 |
| 8 | SEO/content: automated audit + content-gap | Services | 4 | 4 | 4 | 3 | 3 | 18 |

## Winner: Opportunity #1 - Revio v2

**Score: 23/25** (highest)

### Why Revio v2 wins:
1. **Highest Impact (5)**: Revio is an existing Cynbit product. Upgrading it directly benefits current customers and creates upsell potential.
2. **Highest Feasibility (5)**: Python/FastAPI/Streamlit stack matches Cynbit's core technology. Sentiment models and aspect extraction are well-documented.
3. **Highest Data Availability (5)**: Synthetic reviews are trivial to generate. No external data access needed.
4. **Highest Differentiation (5)**: Hinglish/code-mixed sentiment analysis is an underserved gap in the Indian market. Most existing tools handle English-only or pure Hindi.
5. **Moderate Risk (3)**: Synthetic data optimism and sarcasm detection are risks, but mitigated with diverse templates and documented limitations.

### Runner-ups for "Other Opportunities" section:
- **#2 TenantEase (18)**: Solid tabular ML use case, but gradient boosting is commoditized
- **#4 RFP Assistant (16)**: RAG is interesting but needs case study corpus access
- **#8 SEO Audit (18)**: Good services alignment, but less product differentiation
