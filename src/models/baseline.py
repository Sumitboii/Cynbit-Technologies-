"""
Baseline sentiment model: TF-IDF + Logistic Regression.

Requirements: FR-4.1, design Section 3.1
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_SEED, BASELINE_MODEL_PATH


class BaselineSentimentModel:
    """TF-IDF + Logistic Regression sentiment classifier."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=RANDOM_SEED,
            )),
        ])
        self.classes_ = None

    def fit(self, texts: list, labels: list):
        """Train on (text, label) pairs."""
        self.pipeline.fit(texts, labels)
        self.classes_ = self.pipeline.classes_
        return self

    def predict(self, text: str) -> tuple:
        """
        Predict sentiment for a single text.

        Returns:
            Tuple of (sentiment_class, confidence)
        """
        probs = self.pipeline.predict_proba([text])[0]
        idx = np.argmax(probs)
        sentiment = self.classes_[idx]
        confidence = float(probs[idx])
        return (sentiment, confidence)

    def predict_batch(self, texts: list) -> list:
        """Predict for a batch of texts."""
        probs = self.pipeline.predict_proba(texts)
        results = []
        for p in probs:
            idx = np.argmax(p)
            results.append((self.classes_[idx], float(p[idx])))
        return results

    def save(self, path: Path = BASELINE_MODEL_PATH):
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = BASELINE_MODEL_PATH) -> "BaselineSentimentModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
