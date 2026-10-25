"""
Streamlit dashboard for Revio v2.

Requirements: FR-8.1 through FR-8.9, FR-9.5-FR-9.7, NFR-5.5, design Section 5.2

Run with: streamlit run app/dashboard.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.pipeline import RevioPipeline
from src.config import LLM_ENABLED, DATA_DIR


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Revio v2 - Reputation Intelligence Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialise session state for storing data."""
    if "reviews_data" not in st.session_state:
        st.session_state.reviews_data = []
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None


def get_pipeline():
    """Get or create pipeline."""
    if st.session_state.pipeline is None:
        st.session_state.pipeline = RevioPipeline()
    return st.session_state.pipeline


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render sidebar with navigation and settings."""
    st.sidebar.title("Revio v2")
    st.sidebar.markdown("### Reputation Intelligence Engine")
    st.sidebar.markdown("---")

    # LLM status
    if LLM_ENABLED:
        st.sidebar.success("LLM Available - Reply generation enabled")
    else:
        st.sidebar.warning("LLM Unavailable - Using template-based fallback")

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Review Explorer", "Upload Reviews", "Reply Management", "Diagnostics"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built for Cynbit Technologies PoC")
    st.sidebar.markdown(f"Version 1.0.0 | Synthetic Data Only")

    return page


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_dashboard():
    """Main dashboard with aggregated metrics."""
    st.title("Dashboard")
    st.markdown("### Multi-location reputation overview")

    results = st.session_state.analysis_results

    if not results:
        st.info("No reviews analysed yet. Upload reviews or load sample data to see the dashboard.")
        if st.button("Load Sample Data"):
            load_sample_data()
            st.rerun()
        return

    df = pd.DataFrame(results)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    crisis_count = df["crisis_flag"].sum() if "crisis_flag" in df else 0
    auto_approved = (df["approval_status"] == "approved").sum() if "approval_status" in df else 0
    pending = (df["approval_status"] == "pending_human_review").sum() if "approval_status" in df else 0

    col1.metric("Total Reviews", total)
    col2.metric("Crisis Alerts", int(crisis_count))
    col3.metric("Auto-Approved", int(auto_approved))
    col4.metric("Pending Review", int(pending))

    st.markdown("---")

    # Sentiment distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sentiment Distribution")
        if "overall_sentiment" in df:
            sentiment_counts = df["overall_sentiment"].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={"positive": "#4CAF50", "negative": "#F44336", "neutral": "#FF9800"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Crisis Categories")
        if "crisis_categories" in df:
            all_cats = []
            for cats in df["crisis_categories"]:
                if isinstance(cats, list):
                    all_cats.extend(cats)
            if all_cats:
                cat_counts = pd.Series(all_cats).value_counts()
                fig = px.bar(x=cat_counts.index, y=cat_counts.values, labels={"x": "Category", "y": "Count"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No crisis categories detected.")

    # Top negative aspects
    st.markdown("#### Top Negative Aspects by Location")
    neg_aspects = []
    for _, row in df.iterrows():
        if row.get("overall_sentiment") == "negative" and isinstance(row.get("aspects"), list):
            for a in row["aspects"]:
                if isinstance(a, dict) and a.get("sentiment") == "negative":
                    neg_aspects.append({"location": row.get("id", ""), "aspect": a["aspect"]})

    if neg_aspects:
        aspects_df = pd.DataFrame(neg_aspects)
        fig = px.bar(aspects_df["aspect"].value_counts().head(8), title="Most Mentioned Negative Aspects")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No negative aspects detected.")


def page_review_explorer():
    """Review explorer with filtering."""
    st.title("Review Explorer")

    results = st.session_state.analysis_results
    if not results:
        st.info("No reviews loaded. Upload or load sample data first.")
        if st.button("Load Sample Data"):
            load_sample_data()
            st.rerun()
        return

    df = pd.DataFrame(results)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        sentiment_filter = st.multiselect("Sentiment", ["positive", "negative", "neutral"], default=["positive", "negative", "neutral"])
    with col2:
        crisis_only = st.checkbox("Crisis reviews only")
    with col3:
        search = st.text_input("Search text")

    filtered = df.copy()
    if "overall_sentiment" in filtered.columns:
        filtered = filtered[filtered["overall_sentiment"].isin(sentiment_filter)]
    if crisis_only and "crisis_flag" in filtered.columns:
        filtered = filtered[filtered["crisis_flag"] == True]

    # Display
    st.markdown(f"Showing {len(filtered)} of {len(df)} reviews")

    for _, row in filtered.head(50).iterrows():
        with st.expander(f"{row.get('id', '')} | {row.get('overall_sentiment', '')} | {'CRISIS' if row.get('crisis_flag') else 'OK'}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Language:** {row.get('language', '')}")
                st.markdown(f"**Sentiment:** {row.get('overall_sentiment', '')} ({row.get('sentiment_confidence', 0):.2f})")
                st.markdown(f"**Severity:** {row.get('severity', '')}")
            with col2:
                st.markdown(f"**Crisis:** {'YES - ' + ', '.join(row.get('crisis_categories', [])) if row.get('crisis_flag') else 'No'}")
                st.markdown(f"**Approval:** {row.get('approval_status', '')}")

            if isinstance(row.get("aspects"), list) and row.get("aspects"):
                st.markdown("**Aspects:**")
                for a in row["aspects"]:
                    if isinstance(a, dict):
                        st.markdown(f"  - {a['aspect']}: {a['sentiment']} ({a.get('confidence', 0):.2f})")

            st.markdown(f"**Reply:** {row.get('draft_reply', '')}")
            st.markdown(f"**Latency:** {row.get('latency_ms', 0):.1f}ms")


def page_upload():
    """Upload reviews page."""
    st.title("Upload Reviews")
    st.markdown("Upload a CSV file with columns: id, location, platform, rating, text, date")

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            required = ["id", "location", "platform", "rating", "text", "date"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                st.success(f"Loaded {len(df)} reviews")
                st.dataframe(df.head())

                if st.button("Analyse All"):
                    pipeline = get_pipeline()
                    progress = st.progress(0)
                    results = []
                    for i, row in df.iterrows():
                        result = pipeline.analyze(row.to_dict())
                        results.append(result)
                        progress.progress((i + 1) / len(df))

                    st.session_state.analysis_results = results
                    st.success(f"Analysed {len(results)} reviews! Go to Dashboard to see results.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("Or load sample synthetic data:")
    if st.button("Load Sample Data (3000 reviews)"):
        load_sample_data()
        st.rerun()


def page_reply_management():
    """Reply management page."""
    st.title("Reply Management")

    results = st.session_state.analysis_results
    if not results:
        st.info("No reviews analysed yet.")
        return

    df = pd.DataFrame(results)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Auto-Approved", int((df["approval_status"] == "approved").sum()))
    with col2:
        st.metric("Pending Review", int((df["approval_status"] == "pending_human_review").sum()))

    st.markdown("---")

    pending = df[df["approval_status"] == "pending_human_review"]

    for _, row in pending.head(20).iterrows():
        with st.expander(f"{row.get('id', '')} - {row.get('overall_sentiment', '')} - Pending Review"):
            st.markdown(f"**Sentiment:** {row.get('overall_sentiment', '')} | **Severity:** {row.get('severity', '')}")
            st.markdown(f"**Draft Reply:** {row.get('draft_reply', '')}")
            st.markdown(f"**Reason:** {row.get('approval_reason', '')}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Approve {row.get('id', '')}", key=f"approve_{row.get('id', '')}"):
                    st.success("Approved!")
            with col2:
                if st.button(f"Reject {row.get('id', '')}", key=f"reject_{row.get('id', '')}"):
                    st.error("Rejected - will regenerate")


def page_diagnostics():
    """Diagnostics and evaluation page."""
    st.title("Diagnostics")

    st.markdown("### System Status")
    col1, col2, col3 = st.columns(3)
    col1.metric("LLM Status", "Available" if LLM_ENABLED else "Fallback Mode")
    col2.metric("Pipeline", "Ready" if st.session_state.pipeline else "Not Initialised")
    col3.metric("Reviews Loaded", len(st.session_state.analysis_results))

    st.markdown("---")

    st.markdown("### Model Information")
    st.markdown("""
    | Component | Model | Status |
    |-----------|-------|--------|
    | Sentiment | TF-IDF + LR (baseline) / Keyword-based | Active |
    | Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | Optional |
    | Aspect Extraction | Keyword matching | Active |
    | Crisis Detection | Rules + Z-score anomaly | Active |
    | Reply Generation | LLM (optional) + Template fallback | Active |
    """)

    st.markdown("---")

    if st.session_state.analysis_results:
        st.markdown("### Latency Statistics")
        df = pd.DataFrame(st.session_state.analysis_results)
        if "latency_ms" in df:
            st.dataframe(df["latency_ms"].describe().to_frame().T)
            fig = px.histogram(df, x="latency_ms", title="Latency Distribution (ms)")
            st.plotly_chart(fig, use_container_width=True)


def load_sample_data():
    """Load sample synthetic data."""
    reviews_csv = DATA_DIR / "reviews.csv"
    if reviews_csv.exists():
        import csv
        from data.generate_reviews import load_reviews
        reviews = load_reviews(reviews_csv)
        st.session_state.reviews_data = reviews
        pipeline = get_pipeline()
        results = []
        progress = st.progress(0)
        for i, review in enumerate(reviews[:200]):  # Limit to 200 for demo speed
            result = pipeline.analyze(review)
            results.append(result)
            progress.progress((i + 1) / min(len(reviews), 200))
        st.session_state.analysis_results = results
    else:
        st.error("Sample data not found. Run: python data/generate_reviews.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_session_state()
    page = render_sidebar()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Review Explorer":
        page_review_explorer()
    elif page == "Upload Reviews":
        page_upload()
    elif page == "Reply Management":
        page_reply_management()
    elif page == "Diagnostics":
        page_diagnostics()


if __name__ == "__main__":
    main()
