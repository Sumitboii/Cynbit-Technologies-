"""
Tests for crisis detection.

Requirements: FR-5.1 through FR-5.6
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.crisis import CrisisDetector


def test_safety_crisis():
    """Test safety keyword detection."""
    detector = CrisisDetector()
    result = detector.detect("Found broken glass in my food, this is dangerous!")
    assert result["crisis_flag"] == True
    assert "safety" in result["crisis_categories"]


def test_legal_crisis():
    """Test legal keyword detection."""
    detector = CrisisDetector()
    result = detector.detect("I will sue this company, taking legal action")
    assert result["crisis_flag"] == True
    assert "legal" in result["crisis_categories"]


def test_health_crisis():
    """Test health crisis detection."""
    detector = CrisisDetector()
    result = detector.detect("Got food poisoning after eating here, vomit all night")
    assert result["crisis_flag"] == True
    assert "health" in result["crisis_categories"]


def test_fraud_crisis():
    """Test fraud detection."""
    detector = CrisisDetector()
    result = detector.detect("This is a scam, they cheated me with fake products")
    assert result["crisis_flag"] == True
    assert "fraud" in result["crisis_categories"]


def test_no_crisis_positive():
    """No crisis for positive reviews."""
    detector = CrisisDetector()
    result = detector.detect("Great service, loved the food!", sentiment="positive", confidence=0.9)
    assert result["crisis_flag"] == False


def test_no_crisis_neutral():
    """No crisis for neutral reviews."""
    detector = CrisisDetector()
    result = detector.detect("The food was okay, nothing special", sentiment="neutral", confidence=0.8)
    assert result["crisis_flag"] == False


def test_viral_risk():
    """Test viral risk detection."""
    detector = CrisisDetector()
    result = detector.detect("I WILL POST THIS ON SOCIAL MEDIA AND EXPOSE EVERYONE!!!")
    assert result["crisis_flag"] == True
    assert "viral_risk" in result["crisis_categories"]


def test_severity_assignment():
    """Test severity assignment for crisis."""
    detector = CrisisDetector()
    result = detector.detect("Found glass in food, safety hazard!")
    assert result["severity"] == "critical"

    result2 = detector.detect("I will post on social media about this!")
    assert result2["severity"] == "high"


def test_location_spike():
    """Test location spike detection with enough data."""
    detector = CrisisDetector()
    reviews = [
        {"location": "Mumbai North", "sentiment": "negative", "date": "2026-10-20"},
        {"location": "Mumbai North", "sentiment": "negative", "date": "2026-10-21"},
        {"location": "Mumbai North", "sentiment": "negative", "date": "2026-10-22"},
        {"location": "Delhi CP", "sentiment": "negative", "date": "2026-10-20"},
        {"location": "Delhi CP", "sentiment": "positive", "date": "2026-10-21"},
        {"location": "Bangalore", "sentiment": "positive", "date": "2026-10-20"},
        {"location": "Bangalore", "sentiment": "positive", "date": "2026-10-21"},
    ]
    spike = detector.detect_location_spike(reviews, "Mumbai North")
    assert spike["negative_count"] >= 3
