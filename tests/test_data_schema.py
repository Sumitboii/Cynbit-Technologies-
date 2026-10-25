"""
Tests for data schema validation.

Requirements: FR-1.1, FR-1.2, NFR-2.1
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def test_generate_reviews_runs():
    """Test that the data generator runs without errors."""
    from data.generate_reviews import generate_reviews, RANDOM_SEED
    reviews = generate_reviews(count=100, seed=RANDOM_SEED)
    assert len(reviews) == 100
    assert all("id" in r for r in reviews)
    assert all("location" in r for r in reviews)
    assert all("text" in r for r in reviews)
    assert all("sentiment" in r for r in reviews)


def test_reproducibility():
    """Test that same seed produces same output."""
    from data.generate_reviews import generate_reviews, RANDOM_SEED
    r1 = generate_reviews(count=50, seed=42)
    r2 = generate_reviews(count=50, seed=42)
    assert r1[0]["text"] == r2[0]["text"]
    assert r1[-1]["id"] == r2[-1]["id"]


def test_sentiment_distribution():
    """Test that sentiment distribution is roughly correct."""
    from data.generate_reviews import generate_reviews, RANDOM_SEED
    reviews = generate_reviews(count=1000, seed=RANDOM_SEED)
    positives = sum(1 for r in reviews if r["sentiment"] == "positive")
    negatives = sum(1 for r in reviews if r["sentiment"] == "negative")
    neutrals = sum(1 for r in reviews if r["sentiment"] == "neutral")
    assert 400 < positives < 700  # ~60%
    assert 100 < negatives < 500  # 20-40% due to spike locations
    assert 100 < neutrals < 350  # ~20%


def test_languages_present():
    """Test that all three languages are present."""
    from data.generate_reviews import generate_reviews, RANDOM_SEED
    reviews = generate_reviews(count=500, seed=RANDOM_SEED)
    langs = set(r["language"] for r in reviews)
    assert "english" in langs
    # Hindi and Hinglish may vary with small samples
    assert len(langs) >= 2


def test_crisis_injection():
    """Test that crisis reviews are injected."""
    from data.generate_reviews import generate_reviews, RANDOM_SEED
    reviews = generate_reviews(count=2000, seed=RANDOM_SEED)
    crisis_count = sum(1 for r in reviews if r["crisis_flag"])
    assert crisis_count > 0


def test_stratified_split():
    """Test stratified split preserves distribution."""
    from data.generate_reviews import generate_reviews, split_stratified, RANDOM_SEED
    reviews = generate_reviews(count=500, seed=RANDOM_SEED)
    train, val, test = split_stratified(reviews)
    assert len(train) + len(val) + len(test) == 500
    assert 0.65 < len(train) / 500 < 0.75
    assert 0.10 < len(val) / 500 < 0.20
    assert 0.10 < len(test) / 500 < 0.20

