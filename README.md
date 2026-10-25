# Revio v2 - Reputation Intelligence Engine

AI-powered reputation management PoC for **Cynbit Technologies** (AI/ML Engineer Intern Assessment).

Hinglish-aware aspect-based sentiment analysis, crisis detection, and grounded reply drafting for multi-location businesses in India.

## Quickstart (Fresh Clone)

```bash
# 1. Install core dependencies (Python 3.11 or 3.12 recommended)
pip install -r requirements.txt

# 2. Generate synthetic data (required before running tests or eval)
make data
# or: python data/generate_reviews.py

# 3. Run evaluation
make eval
# or: python -m src.evaluate

# 4. (Optional) Train ML models and compare with keyword baseline
make train
# or: python -m src.models.train_and_eval

# 5. (Optional) Install embedding + LLM deps
pip install -r requirements-optional.txt

# 6. Start API
make api
# or: uvicorn app.main:app --reload

# 7. Start dashboard
make ui
# or: streamlit run app/dashboard.py
```

Dashboard: http://localhost:8501 | API docs: http://localhost:8000/docs

> **Note for Windows / Python 3.14**: Streamlit and numpy have a threading issue on Python 3.14.
> Use `set NUMPY_DISABLE_CPU_FEATURES=AVX512F` before running Streamlit, or use Python 3.11/3.12.

## What This Does

Revio v2 upgrades Cynbit's existing reputation management product. It processes customer reviews in **English, Hindi, and Hinglish** (romanised code-mixed text), extracts aspect-level sentiment, detects reputation crises, and drafts grounded, policy-compliant replies with an approval gate.

### Key Features

- **Multi-language**: English, Hindi (Devanagari), Hinglish (romanised Hindi)
- **Aspect-based sentiment**: 8 aspects (service, price, cleanliness, staff, delivery, product_quality, wait_time, ambiance)
- **Crisis detection**: Safety, legal, health, fraud, viral risk, location-level spikes
- **Grounded reply generation**: LLM (optional OpenAI) + template fallback with policy grounding
- **Approval gate**: Auto-approve low-risk, human review for crisis/low-confidence
- **PII masking**: Phone, email, card, name masking before LLM calls
- **Prompt injection resistance**: Delimiters, input validation, pattern detection
- **FastAPI + Streamlit**: REST API and interactive dashboard

## Architecture

```
Review Ingestion -> Language ID + Normalise -> PII Mask -> Aspect Extraction
  -> Sentiment Scoring -> Severity -> Crisis Detection
  -> Reply Generation -> Approval Gate -> API + Dashboard
```

## Evaluation Results

Two separate evaluation sets: synthetic data (optimistic) and a manually-authored gold set (honest generalisation).

> **Gold set note**: The gold set of 98 reviews was **manually authored with AI assistance** — these are NOT real customer reviews and were NOT used for training or tuning.

### Synthetic Test Set (OPTIMISTIC — keyword overlap inflates scores)

| Metric | Target | Actual |
|--------|--------|--------|
| Sentiment Macro-F1 | >= 0.82 | 0.943 |
| Crisis Recall | >= 0.90 | 1.000 |
| Crisis Precision | >= 0.95 | 1.000 |
| Aspect F1 | - | 0.979 |
| Cold Start | < 60s | < 1s |
| Avg Latency | < 3s | 0.005s |

### Manually-Authored Gold Set (HONEST — 98 reviews, NOT used for tuning)

| Metric | Synthetic | Gold | Gap |
|--------|-----------|------|-----|
| Sentiment Macro-F1 | 0.943 | **0.879** | -0.064 |
| Crisis Recall | 1.000 | **0.917** | -0.083 |
| Crisis Precision | 1.000 | **0.846** | -0.154 |
| Aspect F1 | 0.979 | **0.805** | -0.174 |
| Severity Macro-F1 | 1.000 | **0.808** | -0.192 |
| Sarcasm Accuracy | - | **0.800** | - |

### Model Comparison

| Model | Synthetic Test F1 | Gold F1 | Note |
|-------|------------------|---------|------|
| Keyword baseline (default) | 0.943 | **0.879** | Best generalisation; no training needed |
| TF-IDF + LR (trained) | 0.974 | 0.448 | Overfits to synthetic vocabulary |
| MiniLM + LR (trained) | 0.945 | 0.565 | Better than TF-IDF on gold; needs `requirements-optional.txt` |

**Key finding**: Both trained models overfit to synthetic vocabulary. The keyword baseline generalises best to the gold set. Trained models will only outperform keyword baseline once trained on **real** customer data (30-day pilot goal).

Full comparison data: `docs/results/metrics_model_comparison.json`

> **[ASSUMPTION]** "Indian reviews are predominantly Hinglish" and "Hinglish-aware tools are underserved" are assumptions based on general observation, not a verified survey. See reports for details.

## Project Structure

```
cynbit.ai/
├── app/              FastAPI endpoint + Streamlit dashboard
├── src/              Core pipeline, preprocessing, models, reply generation
│   └── models/       baseline.py, train_and_eval.py, aspect_extraction.py, crisis.py
├── data/             Synthetic review generator + manually-authored gold set
├── tests/            44 unit tests (all passing)
├── docs/             Reports (MD + PDF), metrics, decisions
│   └── results/      metrics.json, metrics_handwritten.json, metrics_model_comparison.json
├── notebooks/        Colab demo notebook
├── requirements.txt          Core deps (Python 3.11+)
└── requirements-optional.txt sentence-transformers, openai
```

## Configuration

Copy `.env.example` to `.env` and add `OPENAI_API_KEY` (optional). System works without it using template-based replies.

## Tech Stack

Python 3.11+, FastAPI, Streamlit, scikit-learn, sentence-transformers (optional), OpenAI (optional)

## Limitations

- Synthetic data only (no real customer data)
- Trained models overfit to synthetic vocabulary; keyword baseline is stronger on gold set
- No real API integration (CSV/JSON upload only)
- CPU-only, local demo mode
- No authentication
- **[ASSUMPTION]** Hinglish prevalence and market gap are assumptions, not verified facts

## Reports

- `docs/01_AI_Opportunity_Report.md` — Opportunity analysis and business case
- `docs/02_Tech_Approach_Report.md` — Architecture, models, evaluation, roadmap

## Author

**Sumit Kumar Singh** — AI/ML Engineer Intern Applicant, Cynbit Technologies

---

Built as part of Cynbit Technologies AI/ML Intern Assessment. Synthetic data only. Gold set reviews manually authored with AI assistance, NOT real customer reviews.
