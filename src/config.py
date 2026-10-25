"""
Central configuration for Revio v2 - Reputation Intelligence Engine.

All constants, thresholds, and shared dictionaries live here so the entire
system is reproducible with a single seed.

Requirements: NFR-2.1, NFR-2.5, FR-5.6
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "docs" / "results"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Locations and Platforms (synthetic data)
# ---------------------------------------------------------------------------
LOCATIONS = [
    "Mumbai North",
    "Mumbai South",
    "Bangalore Downtown",
    "Bangalore Tech Park",
    "Delhi CP",
    "Jaipur Old City",
]

PLATFORMS = ["Google Reviews", "Facebook", "Instagram", "Zomato", "Yelp"]

# ---------------------------------------------------------------------------
# Aspects and keywords
# ---------------------------------------------------------------------------
ASPECTS = [
    "service",
    "price",
    "cleanliness",
    "staff",
    "delivery",
    "product_quality",
    "wait_time",
    "ambiance",
]

ASPECT_KEYWORDS = {
    "service": ["service", "experience", "handling", "care", "attention"],
    "price": ["price", "cost", "expensive", "cheap", "worth", "value", "bill"],
    "cleanliness": ["clean", "hygiene", "dirty", "neat", "fresh", "stale", "unclean"],
    "staff": ["staff", "waiter", "owner", "employee", "behavior", "rude", "friendly", "helpful"],
    "delivery": ["delivery", "shipping", "arrived", "delay", "packaging", "courier"],
    "product_quality": ["quality", "taste", "freshness", "defect", "broken", "food", "product"],
    "wait_time": ["wait", "delay", "rush", "quick", "slow", "fast", "late"],
    "ambiance": ["ambiance", "atmosphere", "vibe", "decor", "noise", "music"],
}

# ---------------------------------------------------------------------------
# Crisis keywords (rule-based detection)
# ---------------------------------------------------------------------------
CRISIS_KEYWORDS = {
    "safety": ["poisoned", "broken glass", "injury", "accident", "unsafe", "hazard", "dangerous"],
    "legal": ["lawsuit", "sue", "court", "illegal", "police", "advocate", "legal action"],
    "health": ["food poisoning", "allergy", "virus", "disease", "hygiene violation", "sick", "vomit"],
    "fraud": ["fake", "counterfeit", "stolen", "duplicate", "scam", "cheated", "fraud"],
    "viral_risk": ["will post", "social media", "everyone", "viral", "going public", "expose"],
}

# ---------------------------------------------------------------------------
# Common romanised Hindi words (for Hinglish detection)
# ---------------------------------------------------------------------------
HINGLISH_WORDS = {
    "bahut", "accha", "acha", "acche", "bilkul", "zyada", "thik", "thi", "tha",
    "bhaiya", "shukriya", "nahi", "nahin", "haan", "kya", "ho", "gaya", "kiya",
    "acha", "bekar", "mast", "zabardast", "bhai", "didu", "sahi", "galat",
    "paisa", "khana", "khana", "service", "thoda", "bahut", "kharab", "accha",
}

# ---------------------------------------------------------------------------
# Common Indian names (for PII masking)
# ---------------------------------------------------------------------------
COMMON_INDIAN_NAMES = [
    "Rajesh", "Priya", "Amit", "Ananya", "Rahul", "Sneha", "Vikram",
    "Pooja", "Arjun", "Kavya", "Rohan", "Divya", "Karan", "Nisha",
    "Aditya", "Riya", "Saurabh", "Megha", "Deepak", "Anjali",
    "Sanjay", "Rekha", "Manoj", "Sunita", "Ramesh", "Lata",
]

# ---------------------------------------------------------------------------
# Confidence thresholds (approval gate)
# ---------------------------------------------------------------------------
AUTO_APPROVE_THRESHOLD = 0.80      # positive/neutral auto-approve if confidence >= this
NEGATIVE_AUTO_APPROVE_THRESHOLD = 0.75  # negative auto-approve if confidence >= this
CRISIS_KEYWORD_CONFIDENCE = 0.95
LOCATION_SPIKE_CONFIDENCE = 0.85
SENTIMENT_UNCERTAINTY_THRESHOLD = 0.70
SENTIMENT_UNCERTAINTY_CONFIDENCE = 0.50

# ---------------------------------------------------------------------------
# Location spike detection
# ---------------------------------------------------------------------------
SPIKE_Z_SCORE_THRESHOLD = 2.0
SPIKE_MIN_NEGATIVES = 3
SPIKE_WINDOW_DAYS = 7

# ---------------------------------------------------------------------------
# Crisis specificity (false-positive guard)
# ---------------------------------------------------------------------------
CRISIS_SPECIFICITY_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# LLM configuration (optional)
# ---------------------------------------------------------------------------
LLM_TIMEOUT_SECONDS = 5
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_ENABLED = bool(OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
BASELINE_MODEL_PATH = MODELS_DIR / "baseline.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CLASSIFIER_HEADS_PATH = MODELS_DIR / "classifier_heads.pt"

# ---------------------------------------------------------------------------
# Data file paths
# ---------------------------------------------------------------------------
REVIEWS_CSV = DATA_DIR / "reviews.csv"
REVIEWS_LABELLED_CSV = DATA_DIR / "reviews_labelled.csv"
REVIEWS_TRAIN_CSV = DATA_DIR / "reviews_train.csv"
REVIEWS_VAL_CSV = DATA_DIR / "reviews_val.csv"
REVIEWS_TEST_CSV = DATA_DIR / "reviews_test.csv"
GOLD_SET_CSV = DATA_DIR / "reviews_hand_labelled_gold_set.csv"
TONE_GUIDE_CSV = DATA_DIR / "tone_guide.csv"
POLICY_CSV = DATA_DIR / "policy.csv"
LOCATION_METADATA_CSV = DATA_DIR / "location_metadata.csv"
