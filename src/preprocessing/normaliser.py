"""
Text normalisation for reviews in English, Hindi, and Hinglish.

Requirements: FR-2.2, FR-2.3, FR-2.4, FR-2.6, design Section 3.4
"""

import re


# Abbreviation expansion map
ABBREVIATIONS = {
    "u": "you",
    "ur": "your",
    "thx": "thanks",
    "ty": "thank you",
    "plz": "please",
    "pls": "please",
    "gr8": "great",
    "b4": "before",
    "2": "to",  # context-dependent; applied conservatively
    "asap": "as soon as possible",
    "ok": "okay",
    "k": "okay",
}


def _expand_abbreviations(text: str) -> str:
    """Expand common abbreviations."""
    words = text.split()
    expanded = []
    for word in words:
        clean = re.sub(r"[^a-zA-Z0-9]", "", word.lower())
        if clean in ABBREVIATIONS:
            # Preserve surrounding punctuation
            prefix_match = re.match(r"^[^a-zA-Z]*", word)
            suffix_match = re.match(r"^.*?[a-zA-Z]+", word)
            if prefix_match and suffix_match:
                prefix = prefix_match.group()
                suffix = suffix_match.group()
                rest = word[len(suffix):]
                expanded.append(prefix + ABBREVIATIONS[clean] + rest)
            else:
                expanded.append(word)
        else:
            expanded.append(word)
    return " ".join(expanded)


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs into one."""
    return re.sub(r"\s+", " ", text).strip()


def _reduce_exclamation(text: str) -> str:
    """Reduce multiple exclamation marks to one."""
    return re.sub(r"!{2,}", "!", text)


def _reduce_repetitive_chars(text: str) -> str:
    """Reduce repetitive characters: 'sooooo' -> 'so'."""
    return re.sub(r"(.)\1{3,}", r"\1", text)


def _reduce_caps(text: str) -> str:
    """Reduce excessive caps: 'GREAT' -> 'great' (keep emphasis with one cap)."""
    words = text.split()
    normalised = []
    for word in words:
        if len(word) > 3 and word.isupper():
            normalised.append(word.lower().capitalize())
        else:
            normalised.append(word)
    return " ".join(normalised)


def normalise(text: str, language: str = "english") -> str:
    """
    Normalise review text.

    Preserves negations (e.g. "not good" stays distinct from "good").

    Args:
        text: Raw review text
        language: Detected language ("english", "hindi", "hinglish")

    Returns:
        Normalised text
    """
    if not text:
        return ""

    # Lowercase (preserve Devanagari)
    text = text.lower()

    # Whitespace normalisation
    text = _collapse_whitespace(text)

    # Expand abbreviations
    text = _expand_abbreviations(text)

    # Reduce noise
    text = _reduce_exclamation(text)
    text = _reduce_repetitive_chars(text)
    text = _reduce_caps(text)

    # Final whitespace cleanup
    text = _collapse_whitespace(text)

    return text

