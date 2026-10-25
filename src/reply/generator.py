"""
Reply generation with LLM (optional) + fallback templates.

Requirements: FR-6.1 through FR-6.8, NFR-3.4, NFR-4.1-NFR-4.5, NFR-5.1-NFR-5.3, design Section 4.1-4.3
"""

import json
import logging

from src.config import OPENAI_API_KEY, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a customer service representative for a multi-location business in India.

Your role:
- Acknowledge the customer's feedback sincerely.
- Address specific aspects mentioned (e.g., service, price, delivery).
- Offer solutions grounded ONLY in the provided policy.
- Never promise anything not explicitly stated in the policy.
- Match the tone and language of the review where possible.
- Keep responses concise (2-3 sentences max).

Brand Tone Guide:
{TONE_GUIDE}

Operational Policy:
{POLICY_FILE}

Output Format (JSON):
{{
  "reply": "Your response text here.",
  "aspects_addressed": ["service", "price"],
  "policy_references": ["refund_eligibility", "escalation"],
  "confidence": 0.85,
  "requires_human_review": false,
  "reason": "Positive sentiment, high confidence, no escalation needed."
}}

GROUNDING RULE: Use ONLY the provided policy. If the policy does not address a customer concern,
write: "Our team will follow up with you within 24 hours to address this."
Do NOT invent solutions, promises, or guarantees not in the policy.
"""

FEW_SHOT_EXAMPLES = """
Example 1 - Negative review (non-crisis):
<REVIEW>Service was slow today. Waited 45 minutes for food.</REVIEW>
Response:
{"reply": "We are sorry to hear about the wait. Our team is working to improve service times. We would like to make it right - please mention this review when you next visit for a 50% discount per our compensation policy.", "aspects_addressed": ["service", "wait_time"], "policy_references": ["compensation"], "confidence": 0.88, "requires_human_review": false, "reason": "Clear negative feedback, policy-backed resolution offered."}

Example 2 - Positive review:
<REVIEW>Excellent service! Staff was very helpful and attentive.</REVIEW>
Response:
{"reply": "Thank you so much for the kind words! We are thrilled you enjoyed our service. We look forward to serving you again soon.", "aspects_addressed": ["service", "staff"], "policy_references": [], "confidence": 0.95, "requires_human_review": false, "reason": "Positive sentiment, generic thank-you appropriate."}

Example 3 - Crisis review:
<REVIEW>Found a piece of glass in my food. This is dangerous!</REVIEW>
Response:
{"reply": "We are deeply sorry to hear about this incident. Your safety is our top priority. This has been escalated to our management team who will contact you immediately.", "aspects_addressed": ["product_quality"], "policy_references": ["escalation"], "confidence": 1.0, "requires_human_review": true, "reason": "Crisis flag: safety_hazard. Escalated to manager."}
"""

INJECTION_PATTERNS = ["ignore", "override", "system prompt", "ignore above", "disregard", "forget"]


def _check_injection(text: str) -> bool:
    """Check for prompt injection patterns in review text."""
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in text_lower:
            logger.warning(f"Potential injection detected: pattern '{pattern}' in review text")
            return True
    return False


def _build_prompt(review_text: str, tone_guide: str, policy: str) -> str:
    """Build the full prompt for LLM."""
    system = SYSTEM_PROMPT.format(TONE_GUIDE=tone_guide, POLICY_FILE=policy)
    prompt = f"{system}\n\n{FEW_SHOT_EXAMPLES}\n\n"
    prompt += f"Now analyse this review and generate a response:\n"
    prompt += f"<REVIEW>\n{review_text}\n</REVIEW>\n\n"
    prompt += f"Respond with valid JSON only."
    return prompt


def _generate_llm_reply(review_text: str, tone_guide: str, policy: str) -> dict:
    """Generate reply using LLM API."""
    if not OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SECONDS)

        prompt = _build_prompt(review_text, tone_guide, policy)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful customer service assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )

        content = response.choices[0].message.content.strip()
        # Parse JSON
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        return result

    except Exception as e:
        logger.error(f"LLM reply generation failed: {e}")
        return None


def _fallback_reply_generator(
    review_text: str,
    sentiment: str,
    aspects: list,
    crisis_flag: bool,
    language: str = "english",
    tone_guide: str = "",
    policy: str = "",
) -> dict:
    """
    Template-based fallback when LLM is unavailable or fails.

    Requirements: FR-6.4, FR-6.5, FR-6.8, NFR-5.2, NFR-5.3
    """
    # Crisis: always escalate
    if crisis_flag:
        if language == "hinglish":
            return {
                "reply": "Hum apko is experience ke liye maaf chahte hain. Aapki chinta ko hum seriously le rahe hain. Hamari team aapko 24 ghante mein contact karegi.",
                "aspects_addressed": [a["aspect"] for a in aspects],
                "policy_references": ["escalation"],
                "confidence": 0.85,
                "requires_human_review": True,
                "reason": "Crisis detected; escalated to human review.",
            }
        return {
            "reply": "We sincerely apologize for your experience. Your concern has been escalated to our management team, who will contact you within 24 hours.",
            "aspects_addressed": [a["aspect"] for a in aspects],
            "policy_references": ["escalation"],
            "confidence": 0.85,
            "requires_human_review": True,
            "reason": "Crisis detected; escalated to human review.",
        }

    aspect_names = [a["aspect"] for a in aspects] if aspects else ["service"]
    first_aspect = aspect_names[0]

    # Positive
    if sentiment == "positive":
        if language == "hinglish":
            return {
                "reply": f"Shukriya aapka! Aapka feedback padh ke hum bahut khush hue. {first_aspect} ke liye aapki appreciation ka shukriya. Hum aapki phir se sevya karenge!",
                "aspects_addressed": aspect_names,
                "policy_references": [],
                "confidence": 0.90,
                "requires_human_review": False,
                "reason": "Positive sentiment, auto-approved.",
            }
        return {
            "reply": f"Thank you for your feedback! We are glad you enjoyed our {first_aspect}. We look forward to serving you again soon.",
            "aspects_addressed": aspect_names,
            "policy_references": [],
            "confidence": 0.90,
            "requires_human_review": False,
            "reason": "Positive sentiment, auto-approved.",
        }

    # Negative (non-crisis)
    if sentiment == "negative":
        has_service = any("service" in a or "staff" in a for a in aspect_names)
        if language == "hinglish":
            return {
                "reply": f"Hum apke is experience se dukh hain. {first_aspect} ke issue ko hum seriously le rahe hain. Hamari team isko fix karegi. Shukriya aapki patience ka.",
                "aspects_addressed": aspect_names,
                "policy_references": ["refund_eligibility"] if has_service else [],
                "confidence": 0.75,
                "requires_human_review": True,
                "reason": "Negative sentiment; human review recommended.",
            }
        return {
            "reply": f"We apologize for the issue with our {first_aspect}. Our team is working to improve this. Please reach out to us so we can make it right per our policy.",
            "aspects_addressed": aspect_names,
            "policy_references": ["refund_eligibility"] if has_service else [],
            "confidence": 0.75,
            "requires_human_review": True,
            "reason": "Negative sentiment; human review recommended.",
        }

    # Neutral
    if language == "hinglish":
        return {
            "reply": f"Aapke feedback ke liye shukriya. {first_aspect} ke baare mein aapki rai hum note kar lenge. Kuch aur help chahiye to batayein.",
            "aspects_addressed": aspect_names,
            "policy_references": [],
            "confidence": 0.70,
            "requires_human_review": True,
            "reason": "Neutral sentiment; human review for quality.",
        }
    return {
        "reply": f"Thank you for your feedback. We will use your comments about {first_aspect} to improve. Please reach out if we can help further.",
        "aspects_addressed": aspect_names,
        "policy_references": [],
        "confidence": 0.70,
        "requires_human_review": True,
        "reason": "Neutral sentiment; human review for quality.",
    }


class ReplyGenerator:
    """Reply generation with LLM + fallback."""

    def __init__(self, tone_guide: str = "", policy: str = ""):
        self.tone_guide = tone_guide or "Warm, empathetic, professional"
        self.policy = policy or "Default policy: Escalate all crises to manager."
        self.llm_available = bool(OPENAI_API_KEY)

    def generate(
        self,
        review_text: str,
        masked_text: str,
        sentiment: str,
        aspects: list,
        crisis_flag: bool,
        language: str = "english",
    ) -> dict:
        """
        Generate a reply for a review.

        Args:
            review_text: Original review text
            masked_text: PII-masked text (used for LLM)
            sentiment: Overall sentiment
            aspects: List of aspect dicts
            crisis_flag: Whether this is a crisis review
            language: Detected language

        Returns:
            Dict: {reply, aspects_addressed, policy_references, confidence, requires_human_review, reason}
        """
        # Check for injection attempts
        injection_detected = _check_injection(review_text)
        if injection_detected:
            logger.warning("Injection detected in review; using safe fallback")
            return _fallback_reply_generator(
                masked_text, sentiment, aspects, crisis_flag,
                language, self.tone_guide, self.policy
            )

        # Try LLM first (skip for crisis reviews)
        if self.llm_available and not crisis_flag:
            llm_result = _generate_llm_reply(masked_text, self.tone_guide, self.policy)
            if llm_result and "reply" in llm_result:
                return llm_result

        # Fallback template-based
        return _fallback_reply_generator(
            masked_text, sentiment, aspects, crisis_flag,
            language, self.tone_guide, self.policy
        )
