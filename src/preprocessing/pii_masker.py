"""
PII masking module.

Detects and masks phone numbers, email addresses, credit cards, and
common Indian names before LLM processing.

Requirements: NFR-3.1 through NFR-3.6, design Section 3.6
"""

import re

from src.config import COMMON_INDIAN_NAMES


# Regex patterns
PHONE_PATTERN = re.compile(r"\+?91[-.\s]?\d{10}|\b\d{10}\b")
EMAIL_PATTERN = re.compile(r"[\w.-]+@[\w.-]+\.\w+")
CARD_PATTERN = re.compile(r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}")

# Build name pattern from common Indian names
_name_alternatives = "|".join(re.escape(name) for name in COMMON_INDIAN_NAMES)
NAME_PATTERN = re.compile(rf"\b({_name_alternatives})\b", re.IGNORECASE)


def mask_pii(text: str) -> str:
    """
    Mask PII in text: phone numbers, emails, credit cards, and common names.

    Args:
        text: Raw review text

    Returns:
        Text with PII replaced by mask tokens:
        - <PHONE_MASKED> for phone numbers
        - <EMAIL_MASKED> for emails
        - <CARD_MASKED> for credit cards
        - <NAME_MASKED> for common Indian names
    """
    if not text:
        return ""

    masked = text

    # Mask phone numbers
    masked = PHONE_PATTERN.sub("<PHONE_MASKED>", masked)

    # Mask emails
    masked = EMAIL_PATTERN.sub("<EMAIL_MASKED>", masked)

    # Mask credit cards
    masked = CARD_PATTERN.sub("<CARD_MASKED>", masked)

    # Mask common Indian names
    masked = NAME_PATTERN.sub("<NAME_MASKED>", masked)

    return masked


def mask_pii_batch(texts: list) -> list:
    """Mask PII for a batch of texts."""
    return [mask_pii(t) for t in texts]
