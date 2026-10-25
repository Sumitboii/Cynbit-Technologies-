"""
Evaluation script for Revio v2.

Computes metrics on held-out test set and gold set.
Generates charts and metrics.json.

Requirements: FR-10.1 through FR-10.6, design Section 6
"""

import json
import os
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATA_DIR, RESULTS_DIR, RANDOM_SEED
from src.pipeline import RevioPipeline
from data.generate_reviews import load_reviews


def compute_f1(true_labels, pred_labels, classes=None):
    """Compute macro-F1, per-class precision/recall/F1."""
    if classes is None:
        classes = sorted(set(true_labels) | set(pred_labels))

    results = {}
    f1s = []
    for cls in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[cls] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"macro_f1": round(macro_f1, 3), "per_class": results}


def compute_confusion_matrix(true_labels, pred_labels, classes=None):
    """Compute confusion matrix."""
    if classes is None:
        classes = sorted(set(true_labels) | set(pred_labels))
    matrix = {true: {pred: 0 for pred in classes} for true in classes}
    for t, p in zip(true_labels, pred_labels):
        matrix[t][p] += 1
    return matrix


def evaluate_pipeline(pipeline, reviews):
    """Run pipeline on reviews and collect results."""
    results = []
    for r in reviews:
        result = pipeline.analyze(r)
        results.append(result)
    return results


def evaluate_sentiment(results, gold_reviews):
    """Evaluate sentiment classification."""
    true = [r["sentiment"] for r in gold_reviews]
    pred = [r["overall_sentiment"] for r in results]
    metrics = compute_f1(true, pred)
    cm = compute_confusion_matrix(true, pred)
    return {**metrics, "confusion_matrix": cm}


def evaluate_crisis(results, gold_reviews):
    """Evaluate crisis detection."""
    # Binary: crisis vs non-crisis
    true_binary = [1 if r["crisis_flag"] else 0 for r in gold_reviews]
    pred_binary = [1 if r["crisis_flag"] else 0 for r in results]

    tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate_aspects(results, gold_reviews):
    """Evaluate aspect extraction."""
    # Precision and recall
    correct = 0
    extracted = 0
    gold_total = 0

    for result, gold in zip(results, gold_reviews):
        pred_aspects = set(a["aspect"] for a in result.get("aspects", []))
        gold_aspects = set(a["aspect"] for a in gold.get("aspects", []))

        correct += len(pred_aspects & gold_aspects)
        extracted += len(pred_aspects)
        gold_total += len(gold_aspects)

    precision = correct / extracted if extracted > 0 else 0.0
    recall = correct / gold_total if gold_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def evaluate_severity(results, gold_reviews):
    """Evaluate severity classification."""
    true = [r["severity"] for r in gold_reviews]
    pred = [r["severity"] for r in results]
    return compute_f1(true, pred, classes=["critical", "high", "medium", "low"])


def measure_latency(pipeline, reviews, n=100):
    """Measure latency metrics."""
    # Cold start
    start = time.time()
    pipeline.analyze(reviews[0])
    cold_start = time.time() - start

    # Per-review
    latencies = []
    for r in reviews[:n]:
        start = time.time()
        pipeline.analyze(r)
        latencies.append(time.time() - start)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    return {
        "cold_start_s": round(cold_start, 2),
        "avg_latency_s": round(avg_latency, 3),
        "max_latency_s": round(max_latency, 3),
    }


def generate_explanations(pipeline, reviews, n=5):
    """Generate explanation samples."""
    import random
    random.seed(RANDOM_SEED)
    samples = random.sample(reviews, min(n, len(reviews)))

    explanations = []
    for r in samples:
        result = pipeline.analyze(r)
        explanations.append({
            "review_id": r.get("id", ""),
            "text": r.get("text", ""),
            "sentiment": result["overall_sentiment"],
            "sentiment_explanation": f"Detected via {'baseline model' if pipeline.sentiment_model else 'keyword analysis'}. Confidence: {result['sentiment_confidence']:.2f}",
            "aspects": result["aspects"],
            "crisis_explanation": result.get("crisis_evidence", []),
        })

    return explanations


def main():
    """Run full evaluation."""
    print("=" * 60)
    print("Revio v2 - Evaluation")
    print("=" * 60)

    # Load test set
    test_csv = DATA_DIR / "reviews_test.csv"
    gold_csv = DATA_DIR / "reviews_hand_labelled_gold_set.csv"

    if not test_csv.exists():
        print("Test data not found. Run: python data/generate_reviews.py")
        return

    print("Loading test data...")
    test_reviews = load_reviews(test_csv)
    gold_reviews = load_reviews(gold_csv) if gold_csv.exists() else test_reviews[:100]

    print(f"Test set: {len(test_reviews)} reviews")
    print(f"Gold set: {len(gold_reviews)} reviews")

    # Initialise pipeline
    print("\nInitialising pipeline...")
    pipeline = RevioPipeline(all_reviews=test_reviews)

    # Run evaluation on gold set
    print("Running inference on gold set...")
    results = evaluate_pipeline(pipeline, gold_reviews)

    # Compute metrics
    print("\nComputing metrics...")
    sentiment_metrics = evaluate_sentiment(results, gold_reviews)
    crisis_metrics = evaluate_crisis(results, gold_reviews)
    aspect_metrics = evaluate_aspects(results, gold_reviews)
    severity_metrics = evaluate_severity(results, gold_reviews)

    # Latency
    print("Measuring latency...")
    latency = measure_latency(pipeline, test_reviews, n=50)

    # Explanations
    explanations = generate_explanations(pipeline, gold_reviews, n=5)

    # Language-specific evaluation
    lang_metrics = {}
    for lang in ["english", "hinglish", "hindi"]:
        lang_reviews = [r for r in gold_reviews if r.get("language") == lang]
        if lang_reviews:
            lang_results = evaluate_pipeline(pipeline, lang_reviews)
            lang_true = [r["sentiment"] for r in lang_reviews]
            lang_pred = [r["overall_sentiment"] for r in lang_results]
            lang_metrics[lang] = compute_f1(lang_true, lang_pred)
        else:
            lang_metrics[lang] = {"macro_f1": 0.0, "note": "No reviews in this language"}

    # Compile all metrics
    all_metrics = {
        "sentiment": sentiment_metrics,
        "crisis": crisis_metrics,
        "aspects": aspect_metrics,
        "severity": severity_metrics,
        "latency": latency,
        "language_specific": lang_metrics,
        "explanations": explanations,
        "summary": {
            "sentiment_macro_f1": sentiment_metrics["macro_f1"],
            "crisis_recall": crisis_metrics["recall"],
            "crisis_precision": crisis_metrics["precision"],
            "aspect_f1": aspect_metrics["f1"],
            "cold_start_s": latency["cold_start_s"],
            "avg_latency_s": latency["avg_latency_s"],
        },
    }

    # Save metrics
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\nMetrics saved to {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Sentiment Macro-F1:     {sentiment_metrics['macro_f1']:.3f}")
    print(f"Crisis Recall:          {crisis_metrics['recall']:.3f}")
    print(f"Crisis Precision:       {crisis_metrics['precision']:.3f}")
    print(f"Aspect F1:              {aspect_metrics['f1']:.3f}")
    print(f"Severity Macro-F1:      {severity_metrics['macro_f1']:.3f}")
    print(f"Cold Start:             {latency['cold_start_s']:.2f}s")
    print(f"Avg Latency:            {latency['avg_latency_s']:.3f}s")
    print()
    for lang, m in lang_metrics.items():
        print(f"  {lang} F1: {m.get('macro_f1', 0):.3f}")

    # Check targets
    print("\n" + "-" * 40)
    print("Target Check:")
    targets = {
        "Sentiment F1 >= 0.82": sentiment_metrics["macro_f1"] >= 0.82,
        "Crisis Recall >= 0.90": crisis_metrics["recall"] >= 0.90,
        "Crisis Precision >= 0.95": crisis_metrics["precision"] >= 0.95,
        "Cold start < 60s": latency["cold_start_s"] < 60,
    }
    for target, passed in targets.items():
        status = "PASS" if passed else "NEEDS IMPROVEMENT"
        print(f"  [{status}] {target}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
