"""
Tests for text normaliser.

Requirements: FR-2.2 through FR-2.6
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.normaliser import normalise


def test_lowercase():
    """Test lowercasing."""
    assert normalise("GREAT Service") == "great service"


def test_whitespace_collapse():
    """Test whitespace normalisation."""
    assert normalise("too   many    spaces") == "too many spaces"


def test_abbreviation_expansion():
    """Test abbreviation expansion."""
    result = normalise("thx for the gr8 service")
    assert "thanks" in result
    assert "great" in result


def test_exclamation_reduction():
    """Test multiple exclamation reduction."""
    result = normalise("amazing!!!")
    assert result.count("!") == 1


def test_repetitive_chars():
    """Test repetitive character reduction."""
    result = normalise("sooooo goooood")
    assert "so" in result
    assert "good" in result or "god" in result


def test_negation_preservation():
    """Test that negations are preserved."""
    result = normalise("not good at all")
    assert "not" in result


def test_empty():
    """Test empty text."""
    assert normalise("") == ""
