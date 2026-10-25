"""
Training script for Revio v2 models.

Trains baseline sentiment model on synthetic data.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATA_DIR, BASELINE_MODEL_PATH, RANDOM_SEED
from src.models.baseline import BaselineSentimentModel
from data.generate_reviews import load_reviews


def train_baseline():
    """Train the TF-IDF + Logistic Regression baseline."""
    print("Loading training data...")
    train_csv = DATA_DIR / "reviews_train.csv"
    if not train_csv.exists():
        print("Training data not found. Run: python data/generate_reviews.py")
        return

    reviews = load_reviews(train_csv)
    texts = [r["text"] for r in reviews]
    labels = [r["sentiment"] for r in reviews]

    print(f"Training baseline on {len(reviews)} reviews...")
    model = BaselineSentimentModel()
    model.fit(texts, labels)

    # Evaluate on test set
    test_csv = DATA_DIR / "reviews_test.csv"
    if test_csv.exists():
        test_reviews = load_reviews(test_csv)
        test_texts = [r["text"] for r in test_reviews]
        test_labels = [r["sentiment"] for r in test_reviews]

        predictions = model.predict_batch(test_texts)
        correct = sum(1 for (pred, _), true in zip(predictions, test_labels) if pred == true)
        accuracy = correct / len(test_labels)
        print(f"Test accuracy: {accuracy:.3f}")

    # Save model
    model.save()
    print(f"Model saved to {BASELINE_MODEL_PATH}")


if __name__ == "__main__":
    train_baseline()
