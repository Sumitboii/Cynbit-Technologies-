"""
FastAPI /analyze endpoint for Revio v2.

Requirements: FR-9.1 through FR-9.7, NFR-1.2, NFR-5.1, NFR-5.2, design Section 5.1
"""

import time
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from src.pipeline import RevioPipeline
from src.config import LLM_ENABLED

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Revio v2 - Reputation Intelligence Engine",
    description="Hinglish-aware aspect-based sentiment analysis, crisis detection, and grounded reply drafting.",
    version="1.0.0",
)

# Global pipeline instance (lazy-initialised)
_pipeline = None


def get_pipeline() -> RevioPipeline:
    """Get or create the pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising Revio v2 pipeline...")
        _pipeline = RevioPipeline()
        logger.info("Pipeline ready.")
    return _pipeline


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AspectSentiment(BaseModel):
    aspect: str
    sentiment: str
    confidence: float
    evidence: str = ""


class ReviewInput(BaseModel):
    id: str = Field(..., description="Unique review ID")
    location: str = Field(..., description="Location name")
    platform: str = Field(..., description="Platform (Google, Facebook, etc.)")
    rating: int = Field(..., ge=1, le=5, description="Star rating 1-5")
    text: str = Field(..., min_length=1, description="Review text")
    date: str = Field(..., description="Date in YYYY-MM-DD format")


class ReviewOutput(BaseModel):
    id: str
    language: str
    script: str = "latin"
    language_confidence: float = 0.0
    normalised_text: str
    overall_sentiment: str
    sentiment_confidence: float
    aspects: List[AspectSentiment]
    severity: str
    crisis_flag: bool
    crisis_categories: List[str]
    crisis_confidence: float = 0.0
    draft_reply: Optional[str]
    reply_confidence: Optional[float]
    aspects_addressed: List[str] = []
    policy_references: List[str] = []
    requires_human_review: bool
    approval_status: str
    approval_reason: str = ""
    llm_enabled: bool = False
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    llm_enabled: bool
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        llm_enabled=LLM_ENABLED,
        version="1.0.0",
    )


@app.post("/analyze", response_model=ReviewOutput)
async def analyze_review(review: ReviewInput):
    """
    Analyse a single review end-to-end.

    Returns:
        Full analysis: sentiment, aspects, crisis, reply, approval status
    """
    try:
        pipeline = get_pipeline()
        result = pipeline.analyze(review.model_dump())
        return ReviewOutput(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Revio v2 - Reputation Intelligence Engine",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
