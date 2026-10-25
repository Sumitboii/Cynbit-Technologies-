"""
Synthetic Review Generator for Revio v2.

Generates 2,000-4,000 realistic reviews with controlled parameters:
- Mix of English / Hindi / Hinglish
- Multi-aspect sentences
- Typos, emojis, abbreviations
- Injected crisis cases
- Location-level negative spikes
- Seeded and labelled (aspect, sentiment, severity)

Requirements: FR-1.1, FR-1.6, design Section 2.1
"""

import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import random
import re
import os
from datetime import datetime, timedelta
from pathlib import Path

from src.config import (
    RANDOM_SEED,
    LOCATIONS,
    PLATFORMS,
    ASPECTS,
    ASPECT_KEYWORDS,
    CRISIS_KEYWORDS,
    DATA_DIR,
)

# Ensure data dir exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Review templates by sentiment and language
# ---------------------------------------------------------------------------

POSITIVE_TEMPLATES = {
    "english": [
        "Great {aspect_phrase}! Really impressed.",
        "Loved the {aspect_phrase}, will definitely come back.",
        "Excellent {aspect_phrase}, highly recommend this place.",
        "The {aspect_phrase} was outstanding. Five stars!",
        "Wonderful {aspect_phrase}, exceeded my expectations.",
        "Amazing {aspect_phrase}! Best in the area.",
        "Really good {aspect_phrase}, happy with my visit.",
        "The {aspect_phrase} made my day. Thank you team!",
    ],
    "hinglish": [
        "Bahut accha {aspect_phrase} tha! Definitely aaoonga dubara.",
        "Mast {aspect_phrase}! Bilkul zabardast experience.",
        "{aspect_phrase} bahut accha tha, shukriya team!",
        "Zabardast {aspect_phrase}, highly recommend karta hoon.",
        "{aspect_phrase} was sahi! Bilkul perfect tha.",
        "Accha {aspect_phrase}, happy hoon main. Will come back!",
    ],
    "hindi": [
        "Bahut accha {aspect_phrase} tha! Paka dubara aaunga.",
        "{aspect_phrase} bahut badhiya tha, dhanyavad team!",
        "Excellent {aspect_phrase}, pakka recommend karunga.",
    ],
}

NEGATIVE_TEMPLATES = {
    "english": [
        "Terrible {aspect_phrase}, very disappointed.",
        "The {aspect_phrase} was awful, would not recommend.",
        "Worst {aspect_phrase} I have ever experienced.",
        "Very poor {aspect_phrase}, not worth the money.",
        "Disappointed with the {aspect_phrase}, needs improvement.",
        "The {aspect_phrase} was bad, will not return.",
        "Horrible {aspect_phrase}, waste of time and money.",
        "Not happy with the {aspect_phrase}, expected much better.",
    ],
    "hinglish": [
        "Bilkul bekar {aspect_phrase}, bahut disappointed hoon.",
        "{aspect_phrase} kharab tha, bilkul not recommend.",
        "Worst {aspect_phrase} ever, paisa barbaad.",
        "{aspect_phrase} bahut kharab, ghar jaake socha nahi tha.",
        "Bekar {aspect_phrase}, dobara nahi aaunga.",
        "{aspect_phrase} tha hi nahi accha, waste of money.",
    ],
    "hindi": [
        "Bilkul bekar {aspect_phrase}, bahut dukhi hoon.",
        "{aspect_phrase} bahut kharab tha, paise barbaad.",
        "Worst {aspect_phrase}, dobara nahi aaunga.",
    ],
}

NEUTRAL_TEMPLATES = {
    "english": [
        "The {aspect_phrase} was okay, nothing special.",
        "Average {aspect_phrase}, could be better.",
        "Decent {aspect_phrase}, but room for improvement.",
        "The {aspect_phrase} was fine, met expectations.",
        "Normal {aspect_phrase}, neither good nor bad.",
        "The {aspect_phrase} was average, not much to say.",
    ],
    "hinglish": [
        "{aspect_phrase} thik tha, kuch special nahi.",
        "Average {aspect_phrase} tha, better ho sakta tha.",
        "{aspect_phrase} theek tha, neither good nor bad.",
        "Normal {aspect_phrase}, nothing much to say.",
    ],
    "hindi": [
        "{aspect_phrase} theek tha, kuch khaas nahi.",
        "Average {aspect_phrase} tha, better ho sakta tha.",
    ],
}

# Multi-aspect connector phrases
MULTI_ASPECT_CONNECTORS = [
    " but ",
    " although ",
    " however ",
    " on the other hand ",
    " while ",
    " par ",  # Hinglish "but"
]

# Emojis for injection
POSITIVE_EMOJIS = ["😊", "👍", "❤️", "😍", "🎉", "⭐", "🌟", "👏"]
NEGATIVE_EMOJIS = ["😡", "👎", "😠", "😞", "💔", "🤮", "⚠️", "🚫"]
NEUTRAL_EMOJIS = ["😐", "🤷", "💭", "📝"]

# Common typos
TYPOS = {
    "the": ["teh", "th"],
    "good": ["gud", "god"],
    "great": ["gr8", "grate"],
    "service": ["servise", "servc"],
    "staff": ["staf", "stafff"],
    "bad": ["bd", "baad"],
    "food": ["fod", "fud"],
    "quality": ["qlty", "qualty"],
}


def _inject_typo(text: str) -> str:
    """Inject a single typo into the text."""
    words = text.split()
    for i, word in enumerate(words):
        clean = re.sub(r"[^a-zA-Z]", "", word.lower())
        if clean in TYPOS and random.random() < 0.3:
            words[i] = word.lower().replace(clean, random.choice(TYPOS[clean]))
            break
    return " ".join(words)


def _inject_emoji(text: str, sentiment: str) -> str:
    """Append an emoji matching the sentiment."""
    if sentiment == "positive":
        return text + " " + random.choice(POSITIVE_EMOJIS)
    elif sentiment == "negative":
        return text + " " + random.choice(NEGATIVE_EMOJIS)
    else:
        return text + " " + random.choice(NEUTRAL_EMOJIS)


def _build_aspect_phrase(aspect: str, sentiment: str, language: str) -> str:
    """Build a natural-sounding phrase for an aspect + sentiment + language."""
    keywords = ASPECT_KEYWORDS.get(aspect, [aspect])

    if sentiment == "positive":
        adjectives_en = ["great", "excellent", "amazing", "wonderful", "good", "outstanding"]
        adjectives_hi = ["accha", "badhiya", "zabardast", "mast"]
        adj_list = adjectives_hi if language == "hinglish" else adjectives_en
        adj = random.choice(adj_list)
        keyword = random.choice(keywords)
        return f"{keyword} was {adj}"
    elif sentiment == "negative":
        adjectives_en = ["terrible", "awful", "poor", "bad", "worst", "horrible"]
        adjectives_hi = ["bekar", "kharab", "baad", "bura"]
        adj_list = adjectives_hi if language == "hinglish" else adjectives_en
        adj = random.choice(adj_list)
        keyword = random.choice(keywords)
        return f"{keyword} was {adj}"
    else:
        adj = random.choice(["okay", "average", "fine", "decent", "normal"])
        keyword = random.choice(keywords)
        return f"{keyword} was {adj}"


def _build_multi_aspect_review(
    aspects_sentiments: list,
    language: str,
) -> str:
    """Build a review mentioning multiple aspects with potentially different sentiments."""
    parts = []
    for aspect, sent in aspects_sentiments:
        phrase = _build_aspect_phrase(aspect, sent, language)
        parts.append(phrase)

    connector = random.choice(MULTI_ASPECT_CONNECTORS)
    # Capitalize first letter of each part
    review = parts[0].capitalize()
    for p in parts[1:]:
        p = p if connector.strip() == "par" else p
        review += connector + p

    return review + "."


def _rating_for_sentiment(sentiment: str) -> int:
    """Map sentiment to a 1-5 star rating with noise."""
    if sentiment == "positive":
        return min(5, max(4, random.randint(4, 5) + random.choice([-1, 0, 0, 0])))
    elif sentiment == "negative":
        return max(1, min(2, random.randint(1, 2) + random.choice([0, 0, 0, 1])))
    else:
        return random.randint(3, 3)


def _severity_for_sentiment_and_crisis(sentiment: str, crisis: bool, crisis_type: str) -> str:
    """Assign severity level."""
    if crisis:
        if crisis_type in ("safety", "health", "legal"):
            return "critical"
        else:
            return "high"
    if sentiment == "negative":
        return "high"
    elif sentiment == "neutral":
        return "medium"
    else:
        return "low"


def generate_reviews(count: int = 3000, seed: int = RANDOM_SEED) -> list:
    """
    Generate synthetic reviews with ground-truth labels.

    Args:
        count: Number of reviews to generate (2000-4000 recommended)
        seed: Random seed for reproducibility

    Returns:
        List of dicts with keys: id, location, platform, rating, text, date,
        language, sentiment, aspects (list of {aspect, sentiment}),
        crisis_flag, crisis_category, severity
    """
    random.seed(seed)

    reviews = []
    # Choose 1-2 locations for negative spikes
    spike_locations = random.sample(LOCATIONS, k=min(2, len(LOCATIONS)))

    for i in range(count):
        # Pick location (spike locations get more negatives)
        if random.random() < 0.3 and spike_locations:
            location = random.choice(spike_locations)
        else:
            location = random.choice(LOCATIONS)

        platform = random.choice(PLATFORMS)

        # Sentiment distribution: 60% positive, 20% negative, 20% neutral
        # Spike locations get 40% negative
        if location in spike_locations and random.random() < 0.4:
            sentiment = "negative"
        else:
            r = random.random()
            if r < 0.6:
                sentiment = "positive"
            elif r < 0.8:
                sentiment = "negative"
            else:
                sentiment = "neutral"

        # Language: 70% English, 15% Hinglish, 15% Hindi
        r = random.random()
        if r < 0.70:
            language = "english"
        elif r < 0.85:
            language = "hinglish"
        else:
            language = "hindi"

        # Select 1-3 aspects
        n_aspects = random.choices([1, 2, 3], weights=[50, 30, 20])[0]
        selected_aspects = random.sample(ASPECTS, k=min(n_aspects, len(ASPECTS)))

        # For multi-aspect, sometimes mix sentiments
        aspects_with_sentiments = []
        for j, aspect in enumerate(selected_aspects):
            if j == 0:
                a_sent = sentiment
            else:
                # 70% same as overall, 30% different (mixed review)
                if random.random() < 0.7:
                    a_sent = sentiment
                else:
                    a_sent = random.choice(["positive", "negative", "neutral"])

            aspects_with_sentiments.append((aspect, a_sent))

        # Build review text
        if len(aspects_with_sentiments) == 1:
            aspect, a_sent = aspects_with_sentiments[0]
            templates = {
                "positive": POSITIVE_TEMPLATES,
                "negative": NEGATIVE_TEMPLATES,
                "neutral": NEUTRAL_TEMPLATES,
            }[a_sent]
            aspect_phrase = _build_aspect_phrase(aspect, a_sent, language)
            template = random.choice(templates.get(language, templates["english"]))
            text = template.format(aspect_phrase=aspect_phrase)
        else:
            text = _build_multi_aspect_review(aspects_with_sentiments, language)

        # Crisis injection: 5% of negative reviews
        crisis_flag = False
        crisis_category = ""
        if sentiment == "negative" and random.random() < 0.05:
            crisis_category = random.choice(list(CRISIS_KEYWORDS.keys()))
            crisis_keywords = CRISIS_KEYWORDS[crisis_category]
            crisis_kw = random.choice(crisis_keywords)
            text += f" {crisis_kw}!"
            crisis_flag = True

        # Inject noise
        # Typos: 5% of reviews
        if random.random() < 0.05:
            text = _inject_typo(text)

        # Emojis: 15% of reviews
        if random.random() < 0.15:
            text = _inject_emoji(text, sentiment)

        # Date: random within last 90 days
        days_ago = random.randint(0, 89)
        date = (datetime(2026, 10, 23) - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        # Rating
        rating = _rating_for_sentiment(sentiment)
        if crisis_flag:
            rating = 1

        # Severity
        severity = _severity_for_sentiment_and_crisis(sentiment, crisis_flag, crisis_category)

        review = {
            "id": f"rev_{i+1:05d}",
            "location": location,
            "platform": platform,
            "rating": rating,
            "text": text,
            "date": date,
            "language": language,
            "sentiment": sentiment,
            "aspects": [{"aspect": a, "sentiment": s} for a, s in aspects_with_sentiments],
            "crisis_flag": crisis_flag,
            "crisis_category": crisis_category,
            "severity": severity,
        }
        reviews.append(review)

    return reviews


def _aspects_to_str(aspects: list) -> str:
    """Serialize aspects list to a string for CSV."""
    parts = []
    for a in aspects:
        parts.append(f"{a['aspect']}:{a['sentiment']}")
    return "|".join(parts)


def _str_to_aspects(s: str) -> list:
    """Deserialize aspects from string."""
    if not s:
        return []
    aspects = []
    for part in s.split("|"):
        if ":" in part:
            aspect, sentiment = part.split(":", 1)
            aspects.append({"aspect": aspect, "sentiment": sentiment})
    return aspects


def save_reviews(reviews: list, path: Path, labelled: bool = True):
    """Save reviews to CSV."""
    fieldnames = ["id", "location", "platform", "rating", "text", "date"]
    if labelled:
        fieldnames += ["language", "sentiment", "aspects", "crisis_flag", "crisis_category", "severity"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reviews:
            row = {k: v for k, v in r.items() if k in fieldnames}
            if "aspects" in row:
                row["aspects"] = _aspects_to_str(r["aspects"])
            if "crisis_flag" in row:
                row["crisis_flag"] = str(r["crisis_flag"]).lower()
            writer.writerow(row)


def load_reviews(path: Path) -> list:
    """Load reviews from CSV."""
    reviews = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "aspects" in row and row["aspects"]:
                row["aspects"] = _str_to_aspects(row["aspects"])
            else:
                row["aspects"] = []
            if "crisis_flag" in row:
                row["crisis_flag"] = row["crisis_flag"].lower() == "true"
            if "rating" in row:
                row["rating"] = int(row["rating"])
            reviews.append(row)
    return reviews


def split_stratified(reviews: list, train: float = 0.7, val: float = 0.15, test: float = 0.15):
    """Stratified split by sentiment and crisis status."""
    random.seed(RANDOM_SEED)

    # Group by (sentiment, crisis_flag)
    groups = {}
    for r in reviews:
        key = (r["sentiment"], r["crisis_flag"])
        groups.setdefault(key, []).append(r)

    train_set, val_set, test_set = [], [], []
    for key, group in groups.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train)
        n_val = int(n * val)
        train_set.extend(group[:n_train])
        val_set.extend(group[n_train:n_train + n_val])
        test_set.extend(group[n_train + n_val:])

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    return train_set, val_set, test_set


def create_gold_set(reviews: list, n: int = 100) -> list:
    """Create a hand-labelled gold set with edge cases."""
    random.seed(RANDOM_SEED)

    gold = []

    # Select edge cases: sarcasm, multi-aspect, Hinglish, Hindi, emojis, crisis
    hinglish = [r for r in reviews if r["language"] == "hinglish"]
    hindi = [r for r in reviews if r["language"] == "hindi"]
    crisis = [r for r in reviews if r["crisis_flag"]]
    multi_aspect = [r for r in reviews if len(r["aspects"]) >= 2]
    emojis = [r for r in reviews if any(e in r["text"] for e in POSITIVE_EMOJIS + NEGATIVE_EMOJIS + NEUTRAL_EMOJIS)]

    # Sample from each category
    for subset, k in [(hinglish, 25), (hindi, 15), (crisis, 20), (multi_aspect, 25), (emojis, 15)]:
        gold.extend(random.sample(subset, min(k, len(subset))))

    # Deduplicate and trim to n
    seen = set()
    unique_gold = []
    for r in gold:
        if r["id"] not in seen and len(unique_gold) < n:
            seen.add(r["id"])
            unique_gold.append(r)

    return unique_gold


def generate_tone_guide(path: Path):
    """Generate the brand tone guide CSV."""
    rows = [
        {"scenario": "negative_service", "approved_response_pattern": "Acknowledge {aspect}, apologise, explain root cause if simple, offer resolution per policy.", "tone": "Warm, empathetic, professional", "language": "english"},
        {"scenario": "negative_price", "approved_response_pattern": "Acknowledge price concern, explain value, no promises on discounts.", "tone": "Respectful, business-like", "language": "english"},
        {"scenario": "positive_general", "approved_response_pattern": "Thank customer warmly, mention specific aspect, invite return.", "tone": "Genuine, uplifting", "language": "english"},
        {"scenario": "crisis_safety", "approved_response_pattern": "Immediately escalate to management. Do not promise anything. Acknowledge concern, assure investigation.", "tone": "Formal, urgent, reassuring", "language": "english"},
        {"scenario": "crisis_legal", "approved_response_pattern": "Do not admit fault. Do not promise resolution. Escalate immediately.", "tone": "Formal, neutral", "language": "english"},
        {"scenario": "hinglish_positive", "approved_response_pattern": "Shukriya! Aapke feedback se hum bahut khush hain. Aisi hi service denge.", "tone": "Warm, colloquial", "language": "hinglish"},
        {"scenario": "hinglish_negative", "approved_response_pattern": "Hum apka experience samajhte hain. Team ko inform karenge aur jald fix karenge.", "tone": "Empathetic, colloquial", "language": "hinglish"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "approved_response_pattern", "tone", "language"])
        writer.writeheader()
        writer.writerows(rows)


def generate_policy(path: Path):
    """Generate the policy file CSV."""
    rows = [
        {"service_category": "refund_eligibility", "policy_statement": "Refund eligible for: (1) service failures with rating <=2, (2) product quality defects, (3) delivery delays >48h.", "constraint": "Do not promise refund unless reason matches above."},
        {"service_category": "compensation", "policy_statement": "If refund approved: full refund. If partial fault: 50% refund or store credit. First-time customers: 20% discount on next purchase.", "constraint": "No refund > amount paid."},
        {"service_category": "escalation", "policy_statement": "Any crisis review (safety, legal, health, fraud) escalates to manager within 24h.", "constraint": "Never auto-respond to crisis."},
        {"service_category": "language", "policy_statement": "Replies in language of review where possible. English fallback if needed.", "constraint": "Match review language."},
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["service_category", "policy_statement", "constraint"])
        writer.writeheader()
        writer.writerows(rows)


def generate_location_metadata(path: Path):
    """Generate location metadata CSV."""
    rows = [
        {"name": loc, "city": loc.split()[0], "type": random.choice(["Restaurant", "Retail", "E-commerce"]), "region": "India"}
        for loc in LOCATIONS
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "city", "type", "region"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Generate all synthetic data."""
    print("Generating synthetic reviews...")
    reviews = generate_reviews(count=3000, seed=RANDOM_SEED)
    print(f"  Generated {len(reviews)} reviews")

    # Save unlabelled (for ingestion demo)
    save_reviews(reviews, DATA_DIR / "reviews.csv", labelled=False)
    print(f"  Saved reviews.csv (unlabelled)")

    # Save labelled
    save_reviews(reviews, DATA_DIR / "reviews_labelled.csv", labelled=True)
    print(f"  Saved reviews_labelled.csv")

    # Stratified split
    train_set, val_set, test_set = split_stratified(reviews)
    save_reviews(train_set, DATA_DIR / "reviews_train.csv", labelled=True)
    save_reviews(val_set, DATA_DIR / "reviews_val.csv", labelled=True)
    save_reviews(test_set, DATA_DIR / "reviews_test.csv", labelled=True)
    print(f"  Split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")

    # Gold set
    gold = create_gold_set(reviews, n=100)
    save_reviews(gold, DATA_DIR / "reviews_hand_labelled_gold_set.csv", labelled=True)
    print(f"  Gold set: {len(gold)} reviews")

    # Tone guide and policy
    generate_tone_guide(DATA_DIR / "tone_guide.csv")
    generate_policy(DATA_DIR / "policy.csv")
    generate_location_metadata(DATA_DIR / "location_metadata.csv")
    print("  Saved tone_guide.csv, policy.csv, location_metadata.csv")

    print("\nData generation complete!")
    print(f"  Crisis reviews: {sum(1 for r in reviews if r['crisis_flag'])}")
    print(f"  Languages: EN={sum(1 for r in reviews if r['language']=='english')}, "
          f"HI={sum(1 for r in reviews if r['language']=='hinglish')}, "
          f"HD={sum(1 for r in reviews if r['language']=='hindi')}")


if __name__ == "__main__":
    main()

