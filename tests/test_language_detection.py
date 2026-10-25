"""
Tests for language detection.

Requirements: FR-2.1, FR-2.5
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.language_detection import detect_language


def test_english():
    """Test English detection."""
    lang, script, conf = detect_language("The food was great and service was excellent")
    assert lang == "english"
    assert script == "latin"


def test_hinglish():
    """Test Hinglish detection."""
    lang, script, conf = detect_language("bahut accha service tha, bilkul zabardast")
    assert lang == "hinglish"
    assert script == "latin"


def test_hindi():
    """Test Hindi (Devanagari) detection."""
    lang, script, conf = detect_language("यह रेस्टोरेंट बहुत अच्छा है")
    assert lang == "hindi"
    assert script == "devanagari"


def test_empty():
    """Test empty text."""
    lang, script, conf = detect_language("")
    assert lang == "english"


def test_mixed_script():
    """Test mixed script."""
    lang, script, conf = detect_language("Great service but थोड़ा slow")
    assert lang in ("english", "hindi", "hinglish")
