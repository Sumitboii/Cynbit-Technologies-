"""
Main pipeline: orchestrates the full Revio v2 analysis pipeline.

Wires together: preprocessing -> aspect extraction -> sentiment -> severity ->
crisis detection -> reply generation -> approval routing.

Requirements: FR-9.1, FR-9.2, design Section 5.1
"""

import time
import logging

from src.config import RANDOM_SEED, LLM_ENABLED
from src.preprocessing.language_detection import detect_language
from src.preprocessing.normaliser import normalise
from src.preprocessing.pii_masker import mask_pii
from src.models.aspect_extraction import AspectExtractor
from src.models.crisis import CrisisDetector
from src.models.baseline import BaselineSentimentModel
from src.reply.generator import ReplyGenerator
from src.reply.approval_gate import route_reply

logger = logging.getLogger(__name__)


class RevioPipeline:
    """Full Revio v2 analysis pipeline."""

    def __init__(self, use_embeddings: bool = True, all_reviews: list = None):
        """
        Initialise the pipeline.

        Args:
            use_embeddings: If True, use embedding model; else use baseline only
            all_reviews: All reviews for location spike detection
        """
        self.use_embeddings = use_embeddings
        self.all_reviews = all_reviews or []

        # Components
        self.aspect_extractor = AspectExtractor()
        self.crisis_detector = CrisisDetector()
        self.reply_generator = ReplyGenerator()

        # Sentiment model (baseline for now; embeddings optional)
        self.sentiment_model = None
        self._embeddings_model = None
        self._sentiment_head = None
        self._severity_head = None
        self._models_checked = False

    def _ensure_models(self):
        """Lazy-load models (only checks once)."""
        if self._models_checked:
            return
        self._models_checked = True
        try:
            from src.models.baseline import BASELINE_MODEL_PATH
            if BASELINE_MODEL_PATH.exists():
                self.sentiment_model = BaselineSentimentModel.load()
            else:
                logger.warning("Baseline model not found; using keyword-based sentiment")
        except Exception as e:
            logger.warning(f"Could not load baseline model: {e}")

    def _keyword_sentiment(self, text: str, rating: int = 3) -> tuple:
        """Simple keyword-based sentiment as fallback."""
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "accha",
                         "badhiya", "zabardast", "mast", "sahi", "best", "love", "happy",
                         "impressed", "recommend", "perfect", "outstanding", "shukriya"]
        negative_words = ["bad", "worst", "terrible", "awful", "poor", "horrible",
                          "bekar", "kharab", "baad", "bura", "dirty", "rude", "slow",
                          "disappointed", "waste", "late", "stale", "broken", "poisoned",
                          "scam", "fraud", "dangerous", "unsafe"]

        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        # Also consider rating
        if rating <= 2:
            neg_count += 2
        elif rating >= 4:
            pos_count += 2

        if pos_count > neg_count:
            return ("positive", min(0.95, 0.60 + pos_count * 0.10))
        elif neg_count > pos_count:
            return ("negative", min(0.95, 0.60 + neg_count * 0.10))
        else:
            return ("neutral", 0.60)

    def _determine_severity(self, sentiment: str, rating: int, crisis_flag: bool, crisis_categories: list) -> str:
        """Determine severity from sentiment, rating, and crisis."""
        if crisis_flag:
            if any(c in ("safety", "health", "legal") for c in crisis_categories):
                return "critical"
            return "high"
        if sentiment == "negative":
            return "high" if rating <= 2 else "medium"
        elif sentiment == "neutral":
            return "medium"
        else:
            return "low"

    def analyze(self, review: dict) -> dict:
        """
        Run full analysis pipeline on a single review.

        Args:
            review: Dict with keys: id, location, platform, rating, text, date

        Returns:
            Full analysis result dict
        """
        start_time = time.time()

        self._ensure_models()

        text = review.get("text", "")
        rating = int(review.get("rating", 3))
        location = review.get("location", "")

        # Step 1: Language detection
        language, script, lang_conf = detect_language(text)

        # Step 2: Normalisation
        normalised_text = normalise(text, language)

        # Step 3: PII masking
        masked_text = mask_pii(normalised_text)

        # Step 4: Sentiment classification
        if self.sentiment_model:
            sentiment, s_conf = self.sentiment_model.predict(normalised_text)
        else:
            sentiment, s_conf = self._keyword_sentiment(normalised_text, rating)

        # Step 5: Aspect extraction
        aspects = self.aspect_extractor.extract(normalised_text)

        # Step 6: Crisis detection
        crisis_result = self.crisis_detector.detect(
            text=normalised_text,
            sentiment=sentiment,
            confidence=s_conf,
            location=location,
            all_reviews=self.all_reviews,
        )

        crisis_flag = crisis_result["crisis_flag"]
        crisis_categories = crisis_result["crisis_categories"]

        # Step 7: Severity
        severity = self._determine_severity(sentiment, rating, crisis_flag, crisis_categories)

        # Step 8: Reply generation
        reply_result = self.reply_generator.generate(
            review_text=text,
            masked_text=masked_text,
            sentiment=sentiment,
            aspects=aspects,
            crisis_flag=crisis_flag,
            language=language,
        )

        # Step 9: Approval routing
        routing = route_reply(
            crisis_flag=crisis_flag,
            sentiment=sentiment,
            confidence=reply_result.get("confidence", 0.5),
            severity=severity,
            requires_human_review=reply_result.get("requires_human_review", False),
        )

        latency_ms = (time.time() - start_time) * 1000

        return {
            "id": review.get("id", ""),
            "language": language,
            "script": script,
            "language_confidence": lang_conf,
            "normalised_text": normalised_text,
            "overall_sentiment": sentiment,
            "sentiment_confidence": round(s_conf, 3),
            "aspects": aspects,
            "severity": severity,
            "crisis_flag": crisis_flag,
            "crisis_categories": crisis_categories,
            "crisis_confidence": crisis_result["crisis_confidence"],
            "crisis_evidence": crisis_result["evidence"],
            "draft_reply": reply_result.get("reply", ""),
            "reply_confidence": reply_result.get("confidence", 0.0),
            "aspects_addressed": reply_result.get("aspects_addressed", []),
            "policy_references": reply_result.get("policy_references", []),
            "requires_human_review": routing["approval_status"] == "pending_human_review",
            "approval_status": routing["approval_status"],
            "approval_reason": routing["reason"],
            "llm_enabled": LLM_ENABLED,
            "latency_ms": round(latency_ms, 1),
        }
