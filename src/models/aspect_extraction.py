"""
Aspect extraction: keyword matching + per-aspect sentiment classification.

Requirements: FR-3.1 through FR-3.7, design Section 3.3
"""

import re

from src.config import ASPECT_KEYWORDS


class AspectExtractor:
    """Extract aspects from review text and classify per-aspect sentiment."""

    def __init__(self, sentiment_predictor=None):
        """
        Args:
            sentiment_predictor: callable(embedding) -> (sentiment, confidence)
                If None, uses keyword-based sentiment only.
        """
        self.sentiment_predictor = sentiment_predictor
        self.aspect_keywords = ASPECT_KEYWORDS

    def _keyword_match(self, text: str) -> list:
        """
        First pass: match aspect keywords against text.

        Returns list of (aspect, evidence_span) tuples.
        """
        matches = []
        text_lower = text.lower()

        for aspect, keywords in self.aspect_keywords.items():
            for kw in keywords:
                pattern = r"\b" + re.escape(kw.lower()) + r"\b"
                m = re.search(pattern, text_lower)
                if m:
                    # Extract evidence span (sentence containing the keyword)
                    start = max(0, m.start() - 30)
                    end = min(len(text), m.end() + 30)
                    evidence = text[start:end].strip()
                    matches.append((aspect, evidence))
                    break  # Only one match per aspect

        return matches

    def _detect_negation(self, text: str, keyword: str) -> bool:
        """Check if a keyword is negated in the text."""
        negations = ["not", "no", "never", "n't", "bilkul", "nahi", "nahin"]
        words = text.lower().split()
        try:
            idx = words.index(keyword.lower())
        except ValueError:
            return False

        # Check 3 words before for negation
        for i in range(max(0, idx - 3), idx):
            if words[i] in negations:
                return True
        return False

    def _keyword_sentiment(self, text: str, aspect: str, keyword: str) -> str:
        """Determine sentiment for an aspect using keyword context."""
        negated = self._detect_negation(text, keyword)

        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "outstanding",
                          "accha", "badhiya", "zabardast", "mast", "sahi", "best", "love", "loved",
                          "happy", "impressed", "recommend", "perfect"]
        negative_words = ["bad", "worst", "terrible", "awful", "poor", "horrible",
                         "bekar", "kharab", "baad", "bura", "dirty", "rude", "slow",
                         "disappointed", "waste", "late", "stale", "broken", "defect"]

        words = text.lower().split()
        try:
            idx = words.index(keyword.lower())
        except ValueError:
            idx = 0

        # Check words within a window of the keyword
        window = words[max(0, idx - 3):idx + 5]
        pos_count = sum(1 for w in window if w in positive_words)
        neg_count = sum(1 for w in window if w in negative_words)

        if negated:
            # "not bad" -> positive
            if neg_count > 0:
                return "positive"
            elif pos_count > 0:
                return "negative"
            else:
                return "neutral"

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

    def extract(self, text: str, embedding: object = None) -> list:
        """
        Extract aspects from text with per-aspect sentiment.

        Args:
            text: Normalised review text
            embedding: Optional precomputed embedding for model-based sentiment

        Returns:
            List of dicts: {aspect, sentiment, confidence, evidence}
        """
        results = []
        keyword_matches = self._keyword_match(text)

        for aspect, evidence in keyword_matches:
            # Find the keyword that matched
            kw = None
            for k in self.aspect_keywords.get(aspect, []):
                if k.lower() in text.lower():
                    kw = k
                    break

            if kw:
                sentiment = self._keyword_sentiment(text, aspect, kw)
                confidence = 0.80  # keyword-based confidence
            else:
                sentiment = "neutral"
                confidence = 0.50

            results.append({
                "aspect": aspect,
                "sentiment": sentiment,
                "confidence": confidence,
                "evidence": evidence,
            })

        # Deduplicate by aspect (keep first occurrence)
        seen = set()
        deduped = []
        for r in results:
            if r["aspect"] not in seen:
                seen.add(r["aspect"])
                deduped.append(r)

        return deduped
