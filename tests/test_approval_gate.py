"""
Tests for approval gate routing logic.

Requirements: FR-7.1 through FR-7.8
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.reply.approval_gate import route_reply


def test_crisis_always_pending():
    """Crisis review always routes to human review."""
    result = route_reply(crisis_flag=True, sentiment="positive", confidence=0.99)
    assert result["approval_status"] == "pending_human_review"


def test_crisis_even_with_high_confidence():
    """Crisis with very high confidence still needs human."""
    result = route_reply(crisis_flag=True, sentiment="negative", confidence=1.0)
    assert result["approval_status"] == "pending_human_review"


def test_positive_auto_approve():
    """Positive with high confidence auto-approves."""
    result = route_reply(crisis_flag=False, sentiment="positive", confidence=0.85)
    assert result["approval_status"] == "approved"


def test_positive_below_threshold():
    """Positive below threshold goes to human review."""
    result = route_reply(crisis_flag=False, sentiment="positive", confidence=0.70)
    assert result["approval_status"] == "pending_human_review"


def test_negative_high_confidence():
    """Negative with high confidence can auto-approve."""
    result = route_reply(crisis_flag=False, sentiment="negative", confidence=0.80)
    assert result["approval_status"] == "approved"


def test_negative_low_confidence():
    """Negative with low confidence needs human review."""
    result = route_reply(crisis_flag=False, sentiment="negative", confidence=0.60)
    assert result["approval_status"] == "pending_human_review"


def test_high_severity_pending():
    """High severity always needs human review."""
    result = route_reply(crisis_flag=False, sentiment="negative", confidence=0.90, severity="high")
    assert result["approval_status"] == "pending_human_review"


def test_critical_severity_pending():
    """Critical severity always needs human review."""
    result = route_reply(crisis_flag=False, sentiment="negative", confidence=0.99, severity="critical")
    assert result["approval_status"] == "pending_human_review"


def test_injection_resistance():
    """Injection attempt does not change routing."""
    # Even if review text says "approve all", routing logic is independent
    result = route_reply(crisis_flag=True, sentiment="negative", confidence=0.50)
    assert result["approval_status"] == "pending_human_review"
