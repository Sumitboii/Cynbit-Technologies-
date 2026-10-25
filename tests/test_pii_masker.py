"""
Tests for PII masker.

Requirements: NFR-3.1 through NFR-3.6
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.pii_masker import mask_pii


def test_phone_masking():
    """Test phone number masking."""
    text = "Call me at 9876543210 for details"
    masked = mask_pii(text)
    assert "<PHONE_MASKED>" in masked
    assert "9876543210" not in masked


def test_phone_with_country_code():
    """Test phone with +91 prefix."""
    text = "My number is +91-9876543210"
    masked = mask_pii(text)
    assert "<PHONE_MASKED>" in masked


def test_email_masking():
    """Test email masking."""
    text = "Email me at john@example.com please"
    masked = mask_pii(text)
    assert "<EMAIL_MASKED>" in masked
    assert "john@example.com" not in masked


def test_card_masking():
    """Test credit card masking."""
    text = "Card 1234 5678 9012 3456 charged"
    masked = mask_pii(text)
    assert "<CARD_MASKED>" in masked


def test_name_masking():
    """Test common Indian name masking."""
    text = "Rajesh was our waiter and Priya was the manager"
    masked = mask_pii(text)
    assert "<NAME_MASKED>" in masked


def test_no_false_positive():
    """Test that non-PII text is not masked."""
    text = "The food was great and service was good"
    masked = mask_pii(text)
    assert masked == text


def test_multiple_pii():
    """Test multiple PII types in one text."""
    text = "Call Amit at 9876543210 or email amit@test.com"
    masked = mask_pii(text)
    assert "<NAME_MASKED>" in masked
    assert "<PHONE_MASKED>" in masked
    assert "<EMAIL_MASKED>" in masked


def test_empty_text():
    """Test empty text."""
    assert mask_pii("") == ""
    assert mask_pii(None) == ""

