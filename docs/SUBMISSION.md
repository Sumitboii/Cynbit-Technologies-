# Submission Package - Revio v2 Reputation Intelligence Engine

**Author**: Sumit Kumar Singh
**Date**: October 23, 2026
**Application**: AI/ML Engineer Intern, Cynbit Technologies

---

## 1. Execution Notes (150 words)

I chose to upgrade Revio, Cynbit's existing AI reputation engine, with Hinglish-aware aspect-based sentiment, crisis detection, and grounded reply drafting. Revio already does sentiment analysis and multi-location control; Indian reviews contain a significant share of Hinglish code-mix content — **[ASSUMPTION]**: exact prevalence varies by platform and region, and this is treated as an assumption, not a verified figure. Similarly, calling this gap "underserved" is an **[ASSUMPTION]** based on limited availability of Hinglish-aware NLP tools, not a systematic market survey. The PoC runs CPU-only with Python, FastAPI, and Streamlit, matching Cynbit's stack. It ingests reviews, detects language (English/Hindi/Hinglish), extracts 8 aspects with per-aspect sentiment, flags 6 crisis categories including location-level z-score spikes, and drafts policy-grounded replies with an approval gate, PII masking, and injection resistance. On a held-out evaluation set of 98 reviews **manually authored with AI assistance** (NOT real customer reviews, NOT used for tuning), the engine achieves 0.879 sentiment Macro-F1 and 0.917 crisis recall. Synthetic-data scores (0.943 F1) are optimistic and reported separately. The honest caveat: trained models (TF-IDF+LR: 0.448 gold F1; MiniLM+LR: 0.565 gold F1) overfit to synthetic vocabulary and require real-data fine-tuning to outperform the keyword baseline. Real-world accuracy needs a 30/60/90-day pilot with live Revio customer data.

---

## 2. Email to opportunities@cynbit.in

**Subject**: AI/ML Engineer Intern Application - Revio v2 PoC Submission

Dear Cynbit Technologies Team,

I'm Sumit Kumar Singh, applying for the AI/ML Engineer Intern role. For the assessment task, I analysed Cynbit's website and products and chose to build a proof-of-concept upgrade for Revio, your AI reputation engine.

My PoC, Revio v2, adds Hinglish-aware aspect-based sentiment analysis, crisis detection (safety, legal, health, fraud, viral, location spikes), and grounded policy-compliant reply drafting with an approval gate. It's built in Python with FastAPI and Streamlit, matching your stack, and runs CPU-only. I evaluated it on both synthetic data and a set of 98 reviews manually authored with AI assistance (not real customer data) to give honest generalisation metrics.

The repository contains the full code, two reports (opportunity analysis and technical approach), a Colab notebook, and a Streamlit dashboard. The primary submission link is below:

- Primary (GitHub repo): [YOUR_GITHUB_URL]

I'd welcome the chance to discuss the 30/60/90-day pilot plan for deploying this with real Revio customer data.

Thank you for your time and consideration.

Best regards,
Sumit Kumar Singh
[Your phone number]
[Your email]

---

## 3. Submission Checklist

### Links to Submit (1 primary + up to 4 additional)

| Slot | Item | Status | How to Generate |
|------|------|--------|-----------------|
| 1 (Primary) | GitHub repo | PENDING - needs push | `git remote add origin https://github.com/YOUR_USERNAME/cynbit-revio-v2.git && git push -u origin master` |
| 2 | Live demo (Streamlit) | OPTIONAL | Deploy to Streamlit Community Cloud or Hugging Face Space |
| 3 | Colab notebook | READY | `notebooks/revio_v2_demo.ipynb` (upload to Google Colab, share link) |
| 4 | Reports as PDF | READY | `docs/01_AI_Opportunity_Report.pdf`, `docs/02_Tech_Approach_Report.pdf` (upload to Google Drive, share folder) |
| 5 | Demo video | OPTIONAL | Script in `docs/02_Tech_Approach_Report.md` section 10 |

### Pre-Submission Checklist

- [x] Opportunity chosen via sourced scoring table (docs/opportunity_scoring.md)
- [x] PoC runs from clean clone: `pip install -r requirements.txt && make data && make eval`
- [x] Metrics honest with synthetic-data caveat (synthetic vs manually-authored gold reported separately)
- [x] Model comparison table added (keyword vs TF-IDF+LR vs MiniLM+LR on test + gold)
- [x] Both reports exported to PDF (docs/*.pdf)
- [x] All Cynbit claims fact-checked against cynbit.in (docs/research/company_notes.md)
- [x] Hinglish prevalence and "underserved" marked as [ASSUMPTION] in both reports and Execution Notes
- [x] Gold set labelled as "manually authored with AI assistance, NOT real customer reviews" everywhere
- [x] README has problem, architecture, quickstart, results, limitations
- [x] 44 unit tests pass
- [x] No hard-coded secrets (secret scan clean)
- [x] requirements.txt installable on Python 3.11/3.12 (no torch pin, no --index-url)
- [x] requirements-optional.txt for sentence-transformers and openai
- [ ] Push to GitHub
- [ ] Upload Colab notebook to Google Colab and get share link
- [ ] Upload PDFs to Google Drive and get share link
- [ ] (Optional) Deploy Streamlit to Community Cloud
- [ ] (Optional) Record demo video
- [ ] Email opportunities@cynbit.in with primary link

### Files to Submit

1. **GitHub repo** (primary) - all code, README, reports, notebook
2. **Colab notebook** - `notebooks/revio_v2_demo.ipynb`
3. **PDF reports** - `docs/01_AI_Opportunity_Report.pdf` + `docs/02_Tech_Approach_Report.pdf`
4. **Email** - to opportunities@cynbit.in with one-line note + primary link

### If Deadline Passed

The stated deadline was 10/22/2026 12:58 PM. If it has passed, submit anyway and email opportunities@cynbit.in with the link and a one-line note: "Submitting my AI/ML intern assessment - Revio v2 reputation engine PoC. Apologies if past the deadline; happy to discuss further."
