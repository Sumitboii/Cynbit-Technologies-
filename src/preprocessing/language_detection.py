"""
Language detection for English, Hindi, and Hinglish.

Requirements: FR-2.1, FR-2.5, design Section 3.4
"""

import re

from src.config import HINGLISH_WORDS


def _has_devanagari(text: str) -> bool:
    """Check if text contains Devanagari script characters."""
    devanagari_range = re.compile(r"[\u0900-\u097F]")
    return bool(devanagari_range.search(text))


def _devanagari_ratio(text: str) -> float:
    """Ratio of Devanagari characters to total alphabetic characters."""
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return 0.0
    devanagari_count = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    return devanagari_count / total_alpha


def detect_language(text: str) -> tuple:
    """
    Detect language of text: English, Hindi, or Hinglish.

    Returns:
        Tuple of (language, script, confidence)
        language: "english", "hindi", or "hinglish"
        script: "latin" or "devanagari"
        confidence: float 0-1
    """
    if not text or not text.strip():
        return ("english", "latin", 0.5)

    dev_ratio = _devanagari_ratio(text)

    # Devanagari dominant -> Hindi
    if dev_ratio > 0.3:
        confidence = min(1.0, dev_ratio + 0.3)
        return ("hindi", "devanagari", round(confidence, 2))

    # Latin script: classify English vs Hinglish
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return ("english", "latin", 0.5)

    hinglish_count = sum(1 for w in words if w in HINGLISH_WORDS)
    total_words = len(words)

    if total_words == 0:
        return ("english", "latin", 0.5)

    hinglish_ratio = hinglish_count / total_words

    if hinglish_ratio > 0.15:
        confidence = min(1.0, 0.6 + hinglish_ratio)
        return ("hinglish", "latin", round(confidence, 2))
    else:
        return ("english", "latin", 0.85)
