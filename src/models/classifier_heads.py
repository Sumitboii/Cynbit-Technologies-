"""
Classifier heads for sentiment and severity.

Uses PyTorch with frozen embeddings and trainable classifier heads.
For CPU-only deployment with minimal dependencies, we use a simple
MLP approach with scikit-learn fallback.

Requirements: FR-4.1, FR-4.3, FR-4.2, design Section 3.2
"""

import numpy as np
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import RANDOM_SEED, CLASSIFIER_HEADS_PATH


class SentimentHead:
    """Classifier head for sentiment: positive/negative/neutral."""

    def __init__(self):
        self.clf = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            max_iter=200,
            random_state=RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.15,
        )
        self.label_encoder = LabelEncoder()
        self._fitted = False

    def fit(self, embeddings: np.ndarray, labels: list):
        """Train on embeddings and labels."""
        y = self.label_encoder.fit_transform(labels)
        self.clf.fit(embeddings, y)
        self._fitted = True
        return self

    def predict(self, embedding: np.ndarray) -> tuple:
        """
        Predict sentiment for a single embedding.

        Returns:
            Tuple of (sentiment_class, confidence)
        """
        emb = embedding.reshape(1, -1)
        probs = self.clf.predict_proba(emb)[0]
        idx = np.argmax(probs)
        label = self.label_encoder.inverse_transform([idx])[0]
        return (label, float(probs[idx]))

    def predict_batch(self, embeddings: np.ndarray) -> list:
        """Predict for a batch."""
        probs = self.clf.predict_proba(embeddings)
        results = []
        for p in probs:
            idx = np.argmax(p)
            label = self.label_encoder.inverse_transform([idx])[0]
            results.append((label, float(p[idx])))
        return results


class SeverityHead:
    """Classifier head for severity: critical/high/medium/low."""

    def __init__(self):
        self.clf = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            max_iter=200,
            random_state=RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.15,
        )
        self.label_encoder = LabelEncoder()
        self._fitted = False

    def fit(self, embeddings: np.ndarray, labels: list):
        """Train on embeddings and labels."""
        y = self.label_encoder.fit_transform(labels)
        self.clf.fit(embeddings, y)
        self._fitted = True
        return self

    def predict(self, embedding: np.ndarray, sentiment: str = "", rating: int = 3) -> tuple:
        """
        Predict severity for a single embedding.

        Enforces constraint: never assign critical/high to neutral/positive.

        Returns:
            Tuple of (severity_class, confidence)
        """
        emb = embedding.reshape(1, -1)
        probs = self.clf.predict_proba(emb)[0]
        idx = np.argmax(probs)
        label = self.label_encoder.inverse_transform([idx])[0]

        # Post-hoc rule: don't assign critical/high to positive/neutral
        if sentiment in ("positive", "neutral") and label in ("critical", "high"):
            # Downgrade to medium/low
            if sentiment == "positive":
                label = "low"
            else:
                label = "medium"
            # Recompute confidence
            label_idx = list(self.label_encoder.classes_).index(label)
            probs_arr = self.clf.predict_proba(emb)[0]
            confidence = float(probs_arr[label_idx])
            return (label, confidence)

        return (label, float(probs[idx]))

    def predict_batch(self, embeddings: np.ndarray, sentiments: list = None, ratings: list = None) -> list:
        """Predict for a batch."""
        probs = self.clf.predict_proba(embeddings)
        results = []
        for i, p in enumerate(probs):
            idx = np.argmax(p)
            label = self.label_encoder.inverse_transform([idx])[0]

            sent = sentiments[i] if sentiments else ""
            if sent in ("positive", "neutral") and label in ("critical", "high"):
                if sent == "positive":
                    label = "low"
                else:
                    label = "medium"

            results.append((label, float(p[idx])))
        return results
