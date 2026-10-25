"""
Evaluation on the MANUALLY AUTHORED gold set.

This is a held-out evaluation on realistic Hinglish reviews manually authored
with AI assistance (sarcasm, mixed aspects, typos, borderline crisis) — these
are NOT real customer reviews and were NOT produced by the synthetic generator.
The pipeline is NOT tuned on this set — metrics reflect true generalisation.

Results are reported SEPARATELY from synthetic data metrics.
"""

import json
import sys
import csv
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import RevioPipeline
from src.config import RESULTS_DIR, RANDOM_SEED, DATA_DIR


def load_handwritten_gold(path):
    """Load the hand-written gold set CSV."""
    reviews = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aspects = []
            if row.get("aspects"):
                for part in row["aspects"].split("|"):
                    if ":" in part:
                        a, s = part.split(":", 1)
                        aspects.append({"aspect": a, "sentiment": s})
            row["aspects"] = aspects
            row["crisis_flag"] = row["crisis_flag"].lower() == "true"
            row["rating"] = int(row["rating"])
            reviews.append(row)
    return reviews


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


def evaluate_sentiment(results, gold_reviews):
    true = [r["sentiment"] for r in gold_reviews]
    pred = [r["overall_sentiment"] for r in results]
    return compute_f1(true, pred)


def evaluate_crisis(results, gold_reviews):
    true_binary = [1 if r["crisis_flag"] else 0 for r in gold_reviews]
    pred_binary = [1 if r["crisis_flag"] else 0 for r in results]
    tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate_aspects(results, gold_reviews):
    correct, extracted, gold_total = 0, 0, 0
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
    true = [r["severity"] for r in gold_reviews]
    pred = [r["severity"] for r in results]
    return compute_f1(true, pred, classes=["critical", "high", "medium", "low"])


def main():
    print("=" * 70)
    print("Revio v2 - MANUALLY AUTHORED Gold Set Evaluation (NO tuning)")
    print("Reviews manually authored with AI assistance, NOT real customer reviews")
    print("=" * 70)

    gold_path = DATA_DIR / "reviews_hand_written_gold.csv"
    if not gold_path.exists():
        print(f"Gold set not found at {gold_path}")
        print("Run: python data/hand_written_gold.py")
        return

    gold_reviews = load_handwritten_gold(gold_path)
    print(f"Loaded {len(gold_reviews)} hand-written reviews")

    crisis = sum(1 for r in gold_reviews if r["crisis_flag"])
    print(f"  Crisis: {crisis}")
    print(f"  Positive: {sum(1 for r in gold_reviews if r['sentiment']=='positive')}")
    print(f"  Negative: {sum(1 for r in gold_reviews if r['sentiment']=='negative')}")
    print(f"  Neutral: {sum(1 for r in gold_reviews if r['sentiment']=='neutral')}")

    print("\nRunning pipeline (no tuning)...")
    pipeline = RevioPipeline(all_reviews=gold_reviews)

    results = []
    for r in gold_reviews:
        result = pipeline.analyze(r)
        results.append(result)

    sentiment_metrics = evaluate_sentiment(results, gold_reviews)
    crisis_metrics = evaluate_crisis(results, gold_reviews)
    aspect_metrics = evaluate_aspects(results, gold_reviews)
    severity_metrics = evaluate_severity(results, gold_reviews)

    # Language-specific
    lang_metrics = {}
    for lang in ["english", "hinglish", "hindi"]:
        lang_reviews = [r for r in gold_reviews if r.get("language") == lang]
        if lang_reviews:
            lang_results = [pipeline.analyze(r) for r in lang_reviews]
            lang_true = [r["sentiment"] for r in lang_reviews]
            lang_pred = [r["overall_sentiment"] for r in lang_results]
            lang_metrics[lang] = compute_f1(lang_true, lang_pred)
        else:
            lang_metrics[lang] = {"macro_f1": 0.0, "note": "No reviews"}

    # Sarcasm analysis (first 10 reviews are sarcastic)
    sarcasm_ids = [f"hw_{i:03d}" for i in range(1, 11)]
    sarcasm_reviews = [r for r in gold_reviews if r["id"] in sarcasm_ids]
    if sarcasm_reviews:
        sarcasm_true = [r["sentiment"] for r in sarcasm_reviews]
        sarcasm_results = [pipeline.analyze(r) for r in sarcasm_reviews]
        sarcasm_pred = [r["overall_sentiment"] for r in sarcasm_results]
        sarcasm_correct = sum(1 for t, p in zip(sarcasm_true, sarcasm_pred) if t == p)
        sarcasm_accuracy = sarcasm_correct / len(sarcasm_reviews)
    else:
        sarcasm_accuracy = 0.0

    all_metrics = {
        "dataset": "manually-authored-gold (98 reviews manually authored with AI assistance, NOT real customer reviews, NOT used for tuning)",
        "sentiment": sentiment_metrics,
        "crisis": crisis_metrics,
        "aspects": aspect_metrics,
        "severity": severity_metrics,
        "language_specific": lang_metrics,
        "sarcasm_accuracy": round(sarcasm_accuracy, 3),
        "summary": {
            "sentiment_macro_f1": sentiment_metrics["macro_f1"],
            "crisis_recall": crisis_metrics["recall"],
            "crisis_precision": crisis_metrics["precision"],
            "aspect_f1": aspect_metrics["f1"],
            "severity_macro_f1": severity_metrics["macro_f1"],
            "sarcasm_accuracy": round(sarcasm_accuracy, 3),
        },
        "comparison_with_synthetic": {
            "synthetic_sentiment_f1": 0.977,
            "manually_authored_gold_sentiment_f1": sentiment_metrics["macro_f1"],
            "note": "Manually-authored gold metrics are LOWER due to sarcasm, mixed aspects, and realistic phrasing not present in synthetic data.",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "metrics_handwritten.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {out_path}")

    print("\n" + "=" * 70)
    print("MANUALLY AUTHORED GOLD SET RESULTS (NO TUNING)")
    print("=" * 70)
    print(f"Sentiment Macro-F1:     {sentiment_metrics['macro_f1']:.3f}  (synthetic was 0.977)")
    print(f"Crisis Recall:          {crisis_metrics['recall']:.3f}  (synthetic was 1.000)")
    print(f"Crisis Precision:       {crisis_metrics['precision']:.3f}  (synthetic was 1.000)")
    print(f"Aspect F1:              {aspect_metrics['f1']:.3f}  (synthetic was 0.979)")
    print(f"Severity Macro-F1:      {severity_metrics['macro_f1']:.3f}  (synthetic was 1.000)")
    print(f"Sarcasm Accuracy:       {sarcasm_accuracy:.3f}")
    print()
    for lang, m in lang_metrics.items():
        print(f"  {lang} F1: {m.get('macro_f1', 0):.3f}")

    print("\n" + "-" * 40)
    print("COMPARISON: Synthetic vs Manually-Authored Gold")
    print("-" * 40)
    print(f"  Sentiment F1:  0.977 -> {sentiment_metrics['macro_f1']:.3f}")
    print(f"  Crisis Recall: 1.000 -> {crisis_metrics['recall']:.3f}")
    print(f"  Aspect F1:     0.979 -> {aspect_metrics['f1']:.3f}")
    print("\nNOTE: Manually-authored gold scores are HONEST generalisation metrics.")
    print("Synthetic scores are OPTIMISTIC due to keyword overlap.")
    print("Gold reviews were manually authored with AI assistance, NOT real customer reviews.")


if __name__ == "__main__":
    main()
