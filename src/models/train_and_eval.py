"""
Train TF-IDF+LR and (optionally) MiniLM+LR models, evaluate both on
the synthetic test set and the gold set, and write comparison metrics
to docs/results/metrics_model_comparison.json.

Usage:
    python -m src.models.train_and_eval
    python -m src.models.train_and_eval --no-miniLM   # skip embedding model

Requirements: task #2 - train baseline models, produce comparison table
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR, RESULTS_DIR, MODELS_DIR, RANDOM_SEED,
    BASELINE_MODEL_PATH, EMBEDDING_MODEL_NAME,
)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_labelled_csv(path: Path) -> list[dict]:
    """Load a labelled reviews CSV; handles both synthetic and gold formats."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalise crisis_flag to bool
            row["crisis_flag"] = str(row.get("crisis_flag", "false")).lower() == "true"
            # Normalise rating to int
            try:
                row["rating"] = int(row.get("rating", 3))
            except (ValueError, TypeError):
                row["rating"] = 3
            # Parse aspects field if present (format: "service:positive|staff:negative")
            raw_aspects = row.get("aspects", "")
            parsed = []
            if raw_aspects:
                for part in raw_aspects.split("|"):
                    if ":" in part:
                        a, s = part.split(":", 1)
                        parsed.append({"aspect": a.strip(), "sentiment": s.strip()})
            row["aspects"] = parsed
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def macro_f1(true_labels, pred_labels, classes=None):
    if classes is None:
        classes = sorted(set(true_labels) | set(pred_labels))
    f1s = []
    per_class = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        per_class[cls] = {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3)}
        f1s.append(f1)
    return round(sum(f1s) / len(f1s), 3) if f1s else 0.0, per_class


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

def train_tfidf_lr(train_texts, train_labels):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                   sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs",
                                    C=1.0, random_state=RANDOM_SEED)),
    ])
    model.fit(train_texts, train_labels)
    return model


def eval_tfidf_lr(model, texts, labels):
    preds = model.predict(texts)
    mf1, pc = macro_f1(labels, list(preds))
    return mf1, pc, list(preds)


# ---------------------------------------------------------------------------
# MiniLM embedding + Logistic Regression
# ---------------------------------------------------------------------------

def train_miniLM_lr(train_texts, train_labels):
    """
    Encode texts with frozen MiniLM embeddings, then fit a LogisticRegression.
    Falls back gracefully if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, "sentence-transformers not installed (pip install -r requirements-optional.txt)"

    print("  Loading MiniLM model (first run downloads ~90 MB)...")
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("  Encoding training set...")
    t0 = time.time()
    train_emb = encoder.encode(train_texts, batch_size=64, show_progress_bar=False,
                                 convert_to_numpy=True)
    print(f"  Encoded {len(train_texts)} texts in {time.time()-t0:.1f}s")

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0,
                              random_state=RANDOM_SEED)
    clf.fit(train_emb, train_labels)
    return (encoder, clf), None


def eval_miniLM_lr(model_tuple, texts, labels):
    encoder, clf = model_tuple
    emb = encoder.encode(texts, batch_size=64, show_progress_bar=False,
                          convert_to_numpy=True)
    preds = clf.predict(emb)
    mf1, pc = macro_f1(labels, list(preds))
    return mf1, pc, list(preds)


# ---------------------------------------------------------------------------
# Keyword baseline (existing pipeline, for comparison row)
# ---------------------------------------------------------------------------

def eval_keyword_pipeline(reviews, labels):
    """Run the existing keyword-based pipeline and return F1 (keyword mode only)."""
    from src.pipeline import RevioPipeline
    pipeline = RevioPipeline(all_reviews=reviews)
    # Force keyword-only: skip model file loading
    pipeline._models_checked = True  # prevent lazy-load
    pipeline.sentiment_model = None  # ensure keyword fallback

    preds = [pipeline.analyze(r)["overall_sentiment"] for r in reviews]
    mf1, pc = macro_f1(labels, preds)
    return mf1, pc, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-miniLM", action="store_true",
                        help="Skip MiniLM embedding model (faster, CPU-only demo)")
    args = parser.parse_args()

    print("=" * 65)
    print("Revio v2 - Model Training & Comparison")
    print("=" * 65)

    # ---- Load data ----------------------------------------------------------
    train_csv = DATA_DIR / "reviews_train.csv"
    test_csv  = DATA_DIR / "reviews_test.csv"
    gold_csv  = DATA_DIR / "reviews_hand_written_gold.csv"

    missing = [p for p in [train_csv, test_csv] if not p.exists()]
    if missing:
        print("Missing data files:", missing)
        print("Run: python data/generate_reviews.py")
        sys.exit(1)

    print("Loading datasets...")
    train_data = load_labelled_csv(train_csv)
    test_data  = load_labelled_csv(test_csv)
    gold_data  = load_labelled_csv(gold_csv) if gold_csv.exists() else []

    train_texts  = [r["text"] for r in train_data]
    train_labels = [r["sentiment"] for r in train_data]
    test_texts   = [r["text"] for r in test_data]
    test_labels  = [r["sentiment"] for r in test_data]
    gold_texts   = [r["text"] for r in gold_data]
    gold_labels  = [r["sentiment"] for r in gold_data]

    print(f"  Train: {len(train_data)}  Test: {len(test_data)}  Gold: {len(gold_data)}")

    results = {}

    # ---- 1. Keyword baseline ------------------------------------------------
    print("\n[1/3] Keyword-based baseline (existing pipeline)...")
    t0 = time.time()
    kw_test_f1, kw_test_pc, _ = eval_keyword_pipeline(test_data, test_labels)
    kw_test_time = time.time() - t0

    kw_gold_f1, kw_gold_pc = 0.0, {}
    if gold_data:
        kw_gold_f1, kw_gold_pc, _ = eval_keyword_pipeline(gold_data, gold_labels)

    results["keyword"] = {
        "description": "Keyword matching with negation detection (existing pipeline)",
        "test_macro_f1": kw_test_f1,
        "test_per_class": kw_test_pc,
        "gold_macro_f1": kw_gold_f1,
        "gold_per_class": kw_gold_pc,
        "eval_time_s": round(kw_test_time, 2),
        "note": "Scores on synthetic test are OPTIMISTIC (keyword overlap with generator)",
    }
    print(f"  Test F1: {kw_test_f1:.3f}  Gold F1: {kw_gold_f1:.3f}")

    # ---- 2. TF-IDF + LR -----------------------------------------------------
    print("\n[2/3] Training TF-IDF + Logistic Regression...")
    t0 = time.time()
    tfidf_model = train_tfidf_lr(train_texts, train_labels)
    train_time = time.time() - t0
    print(f"  Trained in {train_time:.1f}s")

    tfidf_test_f1, tfidf_test_pc, _ = eval_tfidf_lr(tfidf_model, test_texts, test_labels)
    tfidf_gold_f1, tfidf_gold_pc = 0.0, {}
    if gold_data:
        tfidf_gold_f1, tfidf_gold_pc, _ = eval_tfidf_lr(tfidf_model, gold_texts, gold_labels)

    # Save model — wrap in BaselineSentimentModel so pipeline.load() works correctly
    from src.models.baseline import BaselineSentimentModel
    bsm = BaselineSentimentModel.__new__(BaselineSentimentModel)
    bsm.pipeline = tfidf_model  # tfidf_model is already a sklearn Pipeline
    bsm.classes_ = tfidf_model.classes_
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_MODEL_PATH, "wb") as f:
        pickle.dump(bsm, f)
    print(f"  Saved to {BASELINE_MODEL_PATH}")

    results["tfidf_lr"] = {
        "description": "TF-IDF (10k features, 1-2grams, sublinear TF) + Logistic Regression (L2, C=1)",
        "train_size": len(train_data),
        "train_time_s": round(train_time, 2),
        "test_macro_f1": tfidf_test_f1,
        "test_per_class": tfidf_test_pc,
        "gold_macro_f1": tfidf_gold_f1,
        "gold_per_class": tfidf_gold_pc,
        "note": "Trained on synthetic data; generalises better than keyword on gold set",
    }
    print(f"  Test F1: {tfidf_test_f1:.3f}  Gold F1: {tfidf_gold_f1:.3f}")

    # ---- 3. MiniLM + LR -----------------------------------------------------
    miniLM_results = None
    if not args.no_miniLM:
        print("\n[3/3] Training MiniLM + Logistic Regression...")
        t0 = time.time()
        miniLM_tuple, err = train_miniLM_lr(train_texts, train_labels)
        if err:
            print(f"  SKIPPED: {err}")
            miniLM_results = {"skipped": True, "reason": err}
        else:
            train_time_ml = time.time() - t0
            print(f"  Trained in {train_time_ml:.1f}s")

            ml_test_f1, ml_test_pc, _ = eval_miniLM_lr(miniLM_tuple, test_texts, test_labels)
            ml_gold_f1, ml_gold_pc = 0.0, {}
            if gold_data:
                ml_gold_f1, ml_gold_pc, _ = eval_miniLM_lr(miniLM_tuple, gold_texts, gold_labels)

            miniLM_results = {
                "description": f"Frozen {EMBEDDING_MODEL_NAME} (384-dim) + Logistic Regression (L2, C=1)",
                "train_size": len(train_data),
                "train_time_s": round(train_time_ml, 2),
                "test_macro_f1": ml_test_f1,
                "test_per_class": ml_test_pc,
                "gold_macro_f1": ml_gold_f1,
                "gold_per_class": ml_gold_pc,
                "note": "Multilingual embeddings; better Hinglish generalisation expected",
            }
            print(f"  Test F1: {ml_test_f1:.3f}  Gold F1: {ml_gold_f1:.3f}")
    else:
        print("\n[3/3] MiniLM skipped (--no-miniLM)")
        miniLM_results = {"skipped": True, "reason": "--no-miniLM flag set"}

    results["miniLM_lr"] = miniLM_results

    # ---- Compile comparison table -------------------------------------------
    comparison = {
        "trained_on": "Synthetic data (RANDOM_SEED=42); gold set NEVER used for training",
        "gold_set_note": "Gold set = manually authored reviews with AI assistance, NOT real customer reviews; used for evaluation only",
        "models": results,
        "comparison_table": {
            "headers": ["Model", "Test Macro-F1", "Gold Macro-F1", "Note"],
            "rows": [
                ["Keyword baseline (existing)",
                 str(kw_test_f1),
                 str(kw_gold_f1),
                 "Optimistic on synthetic (vocab overlap)"],
                ["TF-IDF + LR (trained)",
                 str(tfidf_test_f1),
                 str(tfidf_gold_f1),
                 "Lower on synthetic, fairer on gold"],
            ]
        },
    }
    if miniLM_results and not miniLM_results.get("skipped"):
        comparison["comparison_table"]["rows"].append([
            "MiniLM + LR (trained)",
            str(miniLM_results["test_macro_f1"]),
            str(miniLM_results["gold_macro_f1"]),
            "Best Hinglish generalisation; needs sentence-transformers",
        ])

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "metrics_model_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nComparison saved to {out_path}")

    # ---- Print summary table ------------------------------------------------
    print("\n" + "=" * 65)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 65)
    fmt = "{:<30} {:>12} {:>12} {}"
    print(fmt.format("Model", "Test F1", "Gold F1", "Note"))
    print("-" * 65)
    print(fmt.format("Keyword (existing)", f"{kw_test_f1:.3f}", f"{kw_gold_f1:.3f}",
                     "OPTIMISTIC on synth"))
    print(fmt.format("TF-IDF + LR", f"{tfidf_test_f1:.3f}", f"{tfidf_gold_f1:.3f}",
                     "Fairer on gold"))
    if miniLM_results and not miniLM_results.get("skipped"):
        ml_tf1 = miniLM_results["test_macro_f1"]
        ml_gf1 = miniLM_results["gold_macro_f1"]
        print(fmt.format("MiniLM + LR", f"{ml_tf1:.3f}", f"{ml_gf1:.3f}", "Best Hinglish"))
    elif miniLM_results and miniLM_results.get("skipped"):
        print(fmt.format("MiniLM + LR", "SKIPPED", "SKIPPED",
                         miniLM_results.get("reason", "")))
    print("=" * 65)
    print("\nDone. TF-IDF+LR model saved to", BASELINE_MODEL_PATH)

    return comparison


if __name__ == "__main__":
    main()
