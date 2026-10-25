"""
Approval gate: routes replies to auto-approve or human review.

Requirements: FR-7.1 through FR-7.8, design Section 4.4
"""

from src.config import AUTO_APPROVE_THRESHOLD, NEGATIVE_AUTO_APPROVE_THRESHOLD


def route_reply(
    crisis_flag: bool,
    sentiment: str,
    confidence: float,
    severity: str = "medium",
    requires_human_review: bool = False,
) -> dict:
    """
    Route a generated reply to auto-approve or human review.

    Decision tree:
    1. Crisis -> pending_human_review (always)
    2. Negative + confidence < 0.75 -> pending_human_review
    3. Negative + confidence >= 0.75 -> approved (if not crisis)
    4. Positive/neutral + confidence >= 0.80 -> approved
    5. Positive/neutral + confidence < 0.80 -> pending_human_review
    6. Any severity critical/high -> pending_human_review

    Returns:
        Dict: {approval_status, reason}
    """
    # Crisis always requires human review
    if crisis_flag:
        return {
            "approval_status": "pending_human_review",
            "reason": "Crisis flag set; mandatory human review regardless of confidence.",
        }

    # High severity always requires human review
    if severity in ("critical", "high"):
        return {
            "approval_status": "pending_human_review",
            "reason": f"High severity ({severity}); human review required.",
        }

    # If the reply generator itself flagged for human review
    if requires_human_review:
        return {
            "approval_status": "pending_human_review",
            "reason": "Reply generator flagged for human review.",
        }

    # Negative sentiment routing
    if sentiment == "negative":
        if confidence >= NEGATIVE_AUTO_APPROVE_THRESHOLD:
            return {
                "approval_status": "approved",
                "reason": f"Negative review with confidence {confidence:.2f} >= {NEGATIVE_AUTO_APPROVE_THRESHOLD}; auto-approved.",
            }
        else:
            return {
                "approval_status": "pending_human_review",
                "reason": f"Negative review with confidence {confidence:.2f} < {NEGATIVE_AUTO_APPROVE_THRESHOLD}; human review needed.",
            }

    # Positive/neutral routing
    if sentiment in ("positive", "neutral"):
        if confidence >= AUTO_APPROVE_THRESHOLD:
            return {
                "approval_status": "approved",
                "reason": f"{sentiment.capitalize()} review with confidence {confidence:.2f} >= {AUTO_APPROVE_THRESHOLD}; auto-approved.",
            }
        else:
            return {
                "approval_status": "pending_human_review",
                "reason": f"{sentiment.capitalize()} review with confidence {confidence:.2f} < {AUTO_APPROVE_THRESHOLD}; human review needed.",
            }

    # Default: human review
    return {
        "approval_status": "pending_human_review",
        "reason": "Default routing: human review for safety.",
    }
