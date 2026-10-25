"""
Crisis detection: rule-based keywords + model signals + location anomaly.

Requirements: FR-5.1 through FR-5.6, design Section 3.5
"""

import re
from collections import defaultdict
from datetime import datetime, timedelta

from src.config import (
    CRISIS_KEYWORDS,
    CRISIS_KEYWORD_CONFIDENCE,
    LOCATION_SPIKE_CONFIDENCE,
    SENTIMENT_UNCERTAINTY_THRESHOLD,
    SENTIMENT_UNCERTAINTY_CONFIDENCE,
    SPIKE_Z_SCORE_THRESHOLD,
    SPIKE_MIN_NEGATIVES,
    SPIKE_WINDOW_DAYS,
)


class CrisisDetector:
    """Detect reputation crises from review text and metadata."""

    def __init__(self):
        self.crisis_keywords = CRISIS_KEYWORDS

    def _keyword_detection(self, text: str) -> list:
        """
        Rule-based keyword detection.

        Returns list of (category, confidence, evidence) tuples.
        """
        flags = []
        text_lower = text.lower()

        for category, keywords in self.crisis_keywords.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    # Extract evidence
                    start = max(0, text_lower.index(kw.lower()) - 20)
                    end = min(len(text), text_lower.index(kw.lower()) + len(kw) + 20)
                    evidence = text[start:end].strip()
                    flags.append((category, CRISIS_KEYWORD_CONFIDENCE, evidence))
                    break  # One flag per category

        return flags

    def _viral_risk_detection(self, text: str) -> list:
        """Detect viral risk signals: excessive caps, multiple exclamations, profanity."""
        flags = []

        # Excessive caps (>50% of letters are caps)
        letters = [c for c in text if c.isalpha()]
        if letters:
            caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if caps_ratio > 0.5 and len(letters) > 10:
                flags.append(("viral_risk", 0.70, "Excessive capitalisation"))

        # Multiple exclamation marks (>3)
        exclam_count = text.count("!")
        if exclam_count >= 3:
            flags.append(("viral_risk", 0.65, "Multiple exclamation marks"))

        # Social media threats
        social_patterns = ["will post", "social media", "going public", "expose", "everyone"]
        text_lower = text.lower()
        for pattern in social_patterns:
            if pattern in text_lower:
                flags.append(("viral_risk", 0.80, f"Mentions: {pattern}"))
                break

        return flags

    def _sentiment_uncertainty_signal(self, sentiment: str, confidence: float) -> list:
        """Flag uncertain negative reviews."""
        if sentiment == "negative" and confidence < SENTIMENT_UNCERTAINTY_THRESHOLD:
            return [("uncertain_negative", SENTIMENT_UNCERTAINTY_CONFIDENCE, "Low-confidence negative")]
        return []

    @staticmethod
    def detect_location_spike(reviews: list, location: str, window_days: int = SPIKE_WINDOW_DAYS) -> dict:
        """
        Detect location-level negative spike using z-score.

        Args:
            reviews: List of review dicts with location, sentiment, date
            location: Target location to check
            window_days: Rolling window for spike detection

        Returns:
            Dict: {is_spike, z_score, negative_count, threshold}
        """
        now = datetime(2026, 10, 23)
        window_start = now - timedelta(days=window_days)

        # Count negatives per location in window
        location_negatives = defaultdict(int)
        for r in reviews:
            try:
                review_date = datetime.strptime(r.get("date", ""), "%Y-%m-%d")
            except (ValueError, TypeError):
                continue
            if review_date >= window_start and r.get("sentiment") == "negative":
                location_negatives[r["location"]] += 1

        all_negatives = list(location_negatives.values())
        if not all_negatives or len(all_negatives) < 2:
            return {"is_spike": False, "z_score": 0.0, "negative_count": 0, "threshold": SPIKE_Z_SCORE_THRESHOLD}

        mean_neg = sum(all_negatives) / len(all_negatives)
        variance = sum((x - mean_neg) ** 2 for x in all_negatives) / len(all_negatives)
        stdev = variance ** 0.5 if variance > 0 else 0.001

        loc_count = location_negatives.get(location, 0)
        z_score = (loc_count - mean_neg) / stdev

        is_spike = z_score > SPIKE_Z_SCORE_THRESHOLD and loc_count >= SPIKE_MIN_NEGATIVES

        return {
            "is_spike": is_spike,
            "z_score": round(z_score, 2),
            "negative_count": loc_count,
            "threshold": SPIKE_Z_SCORE_THRESHOLD,
        }

    def detect(self, text: str, sentiment: str = "", confidence: float = 1.0,
               location: str = "", all_reviews: list = None) -> dict:
        """
        Full crisis detection.

        Args:
            text: Normalised review text
            sentiment: Overall sentiment (for uncertainty signal)
            confidence: Sentiment confidence (for uncertainty signal)
            location: Review location (for spike detection)
            all_reviews: All reviews for location spike computation

        Returns:
            Dict: {crisis_flag, crisis_categories, crisis_confidence, severity, evidence}
        """
        all_flags = []

        # Keyword detection
        keyword_flags = self._keyword_detection(text)
        all_flags.extend(keyword_flags)

        # Viral risk
        viral_flags = self._viral_risk_detection(text)
        all_flags.extend(viral_flags)

        # Sentiment uncertainty
        uncertainty_flags = self._sentiment_uncertainty_signal(sentiment, confidence)
        all_flags.extend(uncertainty_flags)

        # Location spike
        if location and all_reviews:
            spike = self.detect_location_spike(all_reviews, location)
            if spike["is_spike"]:
                all_flags.append(("location_spike", LOCATION_SPIKE_CONFIDENCE, f"z_score={spike['z_score']}"))

        # Aggregate
        crisis_flag = len(all_flags) > 0
        categories = list(set(f[0] for f in all_flags))
        max_confidence = max((f[1] for f in all_flags), default=0.0)
        evidence_list = [{"category": f[0], "confidence": f[1], "detail": f[2]} for f in all_flags]

        # Determine severity
        if crisis_flag:
            if any(f[0] in ("safety", "health", "legal") for f in all_flags):
                severity = "critical"
            else:
                severity = "high"
        else:
            severity = None  # Will be set by caller from sentiment

        return {
            "crisis_flag": crisis_flag,
            "crisis_categories": categories,
            "crisis_confidence": round(max_confidence, 2),
            "severity": severity,
            "evidence": evidence_list,
        }
