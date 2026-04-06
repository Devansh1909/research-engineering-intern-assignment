"""
app.py — NarrativeScope: An Investigative Narrative Analysis Platform

Main Streamlit application. NOT a generic tabbed dashboard — uses a vertical
narrative flow where insights cascade from a central search/query, creating
an investigative experience rather than disconnected chart panels.

Author: Devansh
"""

import os
# Must be set before any huggingface/sentence-transformers imports
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import sys
sys.modules['tensorflow'] = None
sys.modules['tensorflow.compat.v2.compiler.tensorrt'] = None

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import logging
import time  # TASK 3: needed for rate-limit cooldown display in sidebar
import networkx as nx

# Local modules
from data_loader import load_data, init_duckdb, query_filtered_data, get_summary_stats, normalize_dataframe
from search_engine import (
    load_embedding_model, compute_embeddings, build_faiss_index,
    semantic_search, validate_query, ZERO_OVERLAP_EXAMPLES
)
from network_analysis import (
    build_cooccurrence_network, compute_centrality_metrics,
    detect_communities, simulate_node_removal, generate_pyvis_html
)
from clustering import (
    compute_umap_embeddings, cluster_topics, extract_cluster_labels,
    get_cluster_stats, create_embedding_visualization, create_datamapplot_html
)
from genai_summarizer import (
    configure_gemini, summarize_timeseries, summarize_topics,
    summarize_network, summarize_search_results
)
from realtime import fetch_live_reddit_data, save_live_data
from events import fetch_wikipedia_events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="NarrativeScope — Digital Narrative Investigation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Override default Streamlit styling for investigative aesthetic */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(108, 99, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .hero-header h1 {
        background: linear-gradient(90deg, #6C63FF, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .hero-header p {
        color: #8892b0;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, rgba(108, 99, 255, 0.1), transparent);
        padding: 0.8rem 1.2rem;
        border-left: 3px solid #6C63FF;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem 0;
    }
    
    .section-header h2 {
        color: #E0E0E0;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Insight cards */
    .insight-card {
        background: #1A1D29;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #2A2D39;
        margin: 0.5rem 0;
        transition: border-color 0.3s;
    }
    
    .insight-card:hover {
        border-color: #6C63FF;
    }
    
    /* Summary boxes for GenAI summaries */
    .genai-summary {
        background: rgba(108, 99, 255, 0.08);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0 1.5rem 0;
        font-size: 0.95rem;
        color: #B0B8C8;
        line-height: 1.6;
    }
    
    .genai-summary::before {
        content: "✨ AI Summary";
        display: block;
        font-size: 0.75rem;
        font-weight: 600;
        color: #6C63FF;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Stat boxes */
    .stat-box {
        background: linear-gradient(145deg, #1A1D29, #22253A);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #2A2D39;
    }
    
    .stat-box .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6C63FF;
    }
    
    .stat-box .stat-label {
        font-size: 0.8rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Search results */
    .search-result {
        background: #1A1D29;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #4ECDC4;
    }
    
    .search-result .similarity-badge {
        background: rgba(78, 205, 196, 0.15);
        color: #4ECDC4;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Suggested queries */
    .suggested-query {
        display: inline-block;
        background: rgba(108, 99, 255, 0.1);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 20px;
        padding: 0.3rem 1rem;
        margin: 0.2rem;
        font-size: 0.85rem;
        color: #8892b0;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .suggested-query:hover {
        background: rgba(108, 99, 255, 0.2);
        color: #6C63FF;
    }
    
    /* Warning/Info banners */
    .edge-case-banner {
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid rgba(255, 107, 107, 0.2);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #FF6B6B;
        font-size: 0.9rem;
    }
    
    .info-banner {
        background: rgba(78, 205, 196, 0.1);
        border: 1px solid rgba(78, 205, 196, 0.2);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #4ECDC4;
        font-size: 0.9rem;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0E1117; }
    ::-webkit-scrollbar-thumb { background: #2A2D39; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #6C63FF; }
    
    /* Streamlit metric override */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #6C63FF;
    }

    /* ── Narrative Health Score card ─────────────────── */
    .health-score-card {
        background: linear-gradient(145deg, #12141f, #1e2135);
        border: 1px solid rgba(108, 99, 255, 0.35);
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        margin: 1rem 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 2rem;
        flex-wrap: wrap;
        box-shadow: 0 4px 24px rgba(108,99,255,0.12);
    }
    .health-score-number {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
    }
    .health-score-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8892b0;
        margin-top: 0.2rem;
    }
    .health-dim { display: flex; flex-direction: column; gap: 0.3rem; }
    .health-dim-bar {
        height: 6px;
        border-radius: 3px;
        background: #2A2D39;
        overflow: hidden;
        width: 140px;
    }
    .health-dim-fill { height: 100%; border-radius: 3px; }
    .health-dim-label {
        font-size: 0.75rem;
        color: #8892b0;
        display: flex;
        justify-content: space-between;
        width: 140px;
    }

    /* ── Anomaly alert card ──────────────────────────── */
    .anomaly-card {
        background: rgba(255, 107, 107, 0.07);
        border: 1px solid rgba(255, 107, 107, 0.25);
        border-left: 4px solid #FF6B6B;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.4rem 0;
    }
    .anomaly-detail {
        color: #B0B8C8;
        font-size: 0.84rem;
        margin-top: 0.3rem;
    }

    /* ── ELI5 help card ─────────────────────────────── */
    .help-card {
        background: rgba(78, 205, 196, 0.05);
        border: 1px solid rgba(78, 205, 196, 0.15);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.3rem 0;
        font-size: 0.88rem;
        color: #9AAFC5;
        line-height: 1.65;
    }
    .help-card-title {
        color: #4ECDC4;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .help-card table { width: 100%; border-collapse: collapse; margin-top: 0.6rem; }
    .help-card th { color: #4ECDC4; font-size: 0.78rem; text-align: left;
                    padding: 0.25rem 0.5rem; border-bottom: 1px solid #2A2D39; }
    .help-card td { font-size: 0.82rem; padding: 0.25rem 0.5rem;
                    border-bottom: 1px solid rgba(42,45,57,0.5); color: #B0B8C8; }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ─────────────────────────────────────────────────
# TASK 2 & 3 FIX: ALL session_state keys must be initialized here — BEFORE any
# widget is instantiated. Writing to a session_state key that shares its name
# with a widget key (after the widget is created) raises a Streamlit error.

if 'search_query' not in st.session_state:
    st.session_state.search_query = ''
if 'main_search' not in st.session_state:
    st.session_state.main_search = ''
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'live_df' not in st.session_state:
    st.session_state.live_df = pd.DataFrame()
if 'live_embeddings' not in st.session_state:
    st.session_state.live_embeddings = None

# TASK 3 FIX: cooldown timestamp init — prevents re-fetch on page reload.
if '_last_reddit_fetch_ts' not in st.session_state:
    st.session_state['_last_reddit_fetch_ts'] = 0


# ─── UI Helper: ELI5 Help Cards ───────────────────────────────────────────────
def help_card(title: str, body: str):
    """Render a collapsible teal-tinted ELI5 explanation panel beneath a section header."""
    with st.expander(f"ℹ️ {title}", expanded=False):
        st.markdown(
            f'<div class="help-card"><div class="help-card-title">How this works</div>{body}</div>',
            unsafe_allow_html=True,
        )


# ─── Narrative Health Score ────────────────────────────────────────────────────
def compute_health_score(results_df: pd.DataFrame) -> dict:
    """
    Compute a composite Narrative Health Score (0-100) for a set of search results.

    Dimensions
    ----------
    breadth     : how many unique subreddits are discussing this (more = healthier)
    velocity    : relative activity in recent vs early period (growing = high)
    distribution: how spread out across authors (not dominated by one account)
    """
    if results_df is None or results_df.empty:
        return {"score": 0, "breadth": 0, "velocity": 0, "distribution": 0,
                "label": "No Data", "color": "#555"}

    n = len(results_df)

    # Breadth: unique subreddits normalised (cap at 1)
    breadth = min(1.0, results_df['subreddit'].nunique() / max(1, n / 5))

    # Velocity: compare last-third vs first-third of posts sorted by date
    if 'datetime' in results_df.columns and n >= 6:
        sorted_df = results_df.sort_values('datetime')
        third = max(1, n // 3)
        early_count = len(sorted_df.iloc[:third])
        late_count  = len(sorted_df.iloc[-third:])
        velocity = min(1.0, (late_count / max(early_count, 1)))
    else:
        velocity = 0.5

    # Distribution: Herfindahl-Hirschman Index inverted (lower HHI = more distributed = better)
    author_shares = results_df['author'].value_counts(normalize=True)
    hhi = float((author_shares ** 2).sum())
    distribution = 1.0 - hhi

    # Composite (weighted)
    raw   = 0.35 * breadth + 0.30 * min(velocity, 1.0) + 0.35 * distribution
    score = int(round(raw * 100))

    if score >= 70:
        label, color = "Healthy",      "#4ECDC4"
    elif score >= 45:
        label, color = "Moderate",     "#FFD166"
    else:
        label, color = "Concentrated", "#FF6B6B"

    return {
        "score": score, "label": label, "color": color,
        "breadth":       round(breadth * 100),
        "velocity":      round(min(velocity, 1.0) * 100),
        "distribution":  round(distribution * 100),
    }


def render_health_score(hs: dict, query: str):
    """Render the Narrative Health Score as a styled HTML card."""
    def bar(pct, color):
        return (
            f'<div class="health-dim-bar">'
            f'<div class="health-dim-fill" style="width:{pct}%;background:{color};"></div>'
            f'</div>'
        )

    dims_html = ""
    for name, key, col in [
        ("Breadth",      "breadth",      "#6C63FF"),
        ("Velocity",     "velocity",     "#4ECDC4"),
        ("Distribution", "distribution", "#FFD166"),
    ]:
        pct = hs.get(key, 0)
        dims_html += (
            f'<div class="health-dim">'
            f'  <div class="health-dim-label"><span>{name}</span><span>{pct}%</span></div>'
            f'  {bar(pct, col)}'
            f'</div>'
        )

    html = f"""
    <div class="health-score-card">
        <div>
            <div class="health-score-number" style="color:{hs['color']};">{hs['score']}</div>
            <div class="health-score-label">Narrative Health Score</div>
            <div style="color:{hs['color']};font-size:0.8rem;font-weight:600;margin-top:0.3rem;">{hs['label']}</div>
        </div>
        <div style="display:flex;flex-direction:column;gap:0.6rem;">{dims_html}</div>
        <div style="color:#8892b0;font-size:0.82rem;flex:1;min-width:180px;">
            Snapshot of the narrative around <b>'{query}'</b>.<br>
            <b>Breadth</b>: discussed across many communities?<br>
            <b>Velocity</b>: growing or dying out?<br>
            <b>Distribution</b>: many voices or just a few?
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ─── Anomaly Detector ──────────────────────────────────────────────────────────
def detect_anomalies(check_df: pd.DataFrame) -> list:
    """
    Scan for suspicious posting patterns indicating coordinated inauthentic behaviour.

    Checks
    ------
    1. Cross-subreddit spammers  : same author in >5 subreddits
    2. Statistical spike         : days with Z-score >2.5 above 7-day rolling mean
    3. Near-duplicate propagation: identical text prefix from 3+ different authors
    """
    flags = []
    if check_df is None or check_df.empty:
        return flags

    # ── 1: cross-subreddit spammers ──────────────────────────────────────────
    if 'subreddit' in check_df.columns and 'author' in check_df.columns:
        clean = check_df[~check_df['author'].isin(['[deleted]', 'unknown', 'AutoModerator'])]
        author_sub = clean.groupby('author')['subreddit'].nunique()
        spammers   = author_sub[author_sub > 5].sort_values(ascending=False).head(5)
        for author, sub_count in spammers.items():
            post_count = int((clean['author'] == author).sum())
            flags.append({
                'type':     '🔁 Cross-Subreddit Propagation',
                'title':    f"u/{author} posted in {sub_count} different subreddits",
                'detail':   (
                    f"{post_count} total posts. Spreading the same topic across many communities "
                    f"can indicate an amplification campaign — or simply a very active user."
                ),
                'severity': 'high' if sub_count > 10 else 'medium',
            })

    # ── 2: temporal spike (Z-score) ──────────────────────────────────────────
    if 'datetime' in check_df.columns:
        daily = check_df.groupby(check_df['datetime'].dt.date).size().reset_index(name='count')
        if len(daily) >= 7:
            daily['roll_mean'] = daily['count'].rolling(7, min_periods=3, center=True).mean()
            daily['roll_std']  = daily['count'].rolling(7, min_periods=3, center=True).std().fillna(1.0)
            daily['z']         = (daily['count'] - daily['roll_mean']) / daily['roll_std'].replace(0, 1)
            spikes = daily[daily['z'] > 2.5]
            for _, row in spikes.head(3).iterrows():
                flags.append({
                    'type':     '📈 Unusual Activity Spike',
                    'title':    f"Spike of {int(row['count'])} posts on {row.iloc[0]}",
                    'detail':   (
                        f"Z-score {row['z']:.1f}σ above the rolling 7-day baseline of "
                        f"{row['roll_mean']:.1f} posts/day. "
                        f"Investigate what real-world event triggered this surge."
                    ),
                    'severity': 'high' if row['z'] > 4 else 'medium',
                })

    # ── 3: near-duplicate propagation ────────────────────────────────────────
    if 'text' in check_df.columns and len(check_df) > 10:
        d2 = check_df[~check_df['author'].isin(['[deleted]', 'unknown'])].copy()
        d2['prefix'] = d2['text'].str[:80].str.lower().str.strip()
        pfx_stats = d2.groupby('prefix').agg(
            authors=('author', 'nunique'),
            posts=('text', 'count')
        )
        dupes = pfx_stats[(pfx_stats['authors'] > 2) & (pfx_stats['posts'] > 3)]
        if not dupes.empty:
            worst   = dupes.sort_values('posts', ascending=False).iloc[0]
            snippet = dupes.index[0][:60] + '...'
            flags.append({
                'type':     '📋 Near-Duplicate Propagation',
                'title':    f"Template text shared by {int(worst['authors'])} accounts ({int(worst['posts'])} posts)",
                'detail':   (
                    f'Starting with: "{snippet}" — '
                    f"Multiple accounts sharing near-identical text is a strong coordination signal."
                ),
                'severity': 'high',
            })

    return flags


st.markdown("""
<div class="hero-header">
    <h1>🔍 NarrativeScope</h1>
    <p>An investigative platform for tracing how digital narratives emerge, spread, and evolve across online communities. 
    Start by searching for a topic, or explore the dataset below.</p>
</div>
""", unsafe_allow_html=True)


# ─── Load Data ────────────────────────────────────────────────────────────────
# Separate caching: st.cache_data for serializable data, st.cache_resource for objects
@st.cache_data(show_spinner="Loading dataset...")
def load_and_embed():
    """Load data and compute embeddings (cached across sessions)."""
    df = load_data()
    if df.empty:
        return None, None
    model = load_embedding_model()
    embeddings = compute_embeddings(model, df['text'].tolist())
    return df, embeddings


# Load everything
with st.spinner("Initializing NarrativeScope..."):
    result = load_and_embed()
    if result[0] is None:
        df, embeddings = None, None
    else:
        df, embeddings = result
    
    # Merge live data from session state
    if df is not None and 'live_df' in st.session_state and not st.session_state.live_df.empty:
        df = pd.concat([df, st.session_state.live_df], ignore_index=True)
        # Drop strict duplicates (ignoring index)
        df = df.drop_duplicates(subset=['text', 'author', 'datetime'], keep='last').reset_index(drop=True)
        
        if st.session_state.live_embeddings is not None and embeddings is not None:
            # Note: the full recompute via drop_duplicates above means indices might shift.
            # To be 100% physically aligned, it's safer to just let Data_loader's DuckDB and the FAISS FAISS index build on the concatenated DF directly.
            pass
            
            # Since drop_duplicates alters rows, let's just recompute embeddings for the combined df if live data exists 
            # to avoid index-mismatch with FAISS. (Or we can assume negligible duplicates and just vstack).
            # To be robust, let's vstack ONLY if we skip drop_duplicates, but since we drop duplicates:
            embeddings = compute_embeddings(load_embedding_model(), df['text'].tolist())

    if df is not None and not df.empty:
        con = init_duckdb(df)
        model = load_embedding_model()
        index = build_faiss_index(embeddings)
    else:
        con, model, index = None, None, None

if df is None or df.empty:
    st.error("⚠️ No data loaded. Please add your data files to the `data/` directory.")
    st.info("""
    **Expected data format:** JSONL files with Reddit post data.
    
    Place your `.jsonl` files in the `data/` folder and restart the app.
    
    Expected fields: `author`, `subreddit`, `body` or `title`, `created_utc`, `score`
    """)
    st.stop()


# ─── Sidebar: Configuration & Filters ────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Investigation Controls")
    
    # API Key for GenAI summaries
    gemini_key = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        help="For AI-generated summaries. Get a free key at https://aistudio.google.com/apikey"
    )
    if gemini_key:
        configure_gemini(gemini_key)
    
    st.divider()
    
    # Live Data Connection
    st.markdown("#### ⚡ Real-Time Connection")
    with st.expander("Fetch Live Reddit Data", expanded=False):
        # TASK 1 FIX: Show whether credentials are already set via environment variables
        # so users know they don't need to paste them in the UI every session.
        _env_id_set     = bool(os.environ.get("REDDIT_CLIENT_ID", "").strip())
        _env_secret_set = bool(os.environ.get("REDDIT_CLIENT_SECRET", "").strip())
        if _env_id_set and _env_secret_set:
            st.success("✅ Reddit credentials loaded from environment variables.")
        else:
            st.caption(
                "Tip: Set `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` env vars "
                "to avoid entering credentials every session."
            )

        reddit_client_id = st.text_input(
            "Reddit Client ID" + (" (env override active)" if _env_id_set else ""),
            type="password",
            key="reddit_client_id_input"
        )
        reddit_secret = st.text_input(
            "Reddit Secret" + (" (env override active)" if _env_secret_set else ""),
            type="password",
            key="reddit_secret_input"
        )
        reddit_sub   = st.text_input("Subreddit (e.g., all, news)", value="all", key="reddit_sub_input")
        reddit_query = st.text_input("Search Keyword (optional)", key="reddit_query_input")

        # TASK 3 FIX: Show cooldown countdown so users don't spam the button.
        _last_fetch = st.session_state.get("_last_reddit_fetch_ts", 0)
        _cooldown_remaining = max(0, 30 - int(time.time() - _last_fetch)) if _last_fetch else 0
        if _cooldown_remaining > 0:
            st.info(f"⏳ Next fetch available in {_cooldown_remaining}s (rate limit guard)")

        if st.button("Fetch Latest Posts", key="fetch_reddit_btn", disabled=_cooldown_remaining > 0):
            # TASK 1: Credentials can come from env vars (no UI input required)
            _has_ui_creds = bool(reddit_client_id and reddit_secret)
            _has_env_creds = _env_id_set and _env_secret_set
            if not _has_ui_creds and not _has_env_creds:
                st.error("Please provide both Client ID and Secret (or set env vars).")
            else:
                with st.spinner("Fetching live data from Reddit..."):
                    new_posts, err = fetch_live_reddit_data(
                        client_id=reddit_client_id or "",
                        client_secret=reddit_secret or "",
                        user_agent="NarrativeScope/1.0",
                        query=reddit_query or "",
                        subreddit_name=reddit_sub or "all",
                        limit=50
                    )

                    if err:
                        st.error(err)
                    elif not new_posts:
                        st.info("No new posts found for that query/subreddit.")
                    else:
                        st.success(f"✅ Fetched {len(new_posts)} live posts!")
                        save_live_data(new_posts)

                        # Process and append to session state
                        new_df = normalize_dataframe(pd.DataFrame(new_posts))
                        st.session_state.live_df = pd.concat(
                            [st.session_state.live_df, new_df], ignore_index=True
                        )
                        # TASK 3: Flag for embedding recompute on the next run.
                        st.session_state.live_embeddings = True
                        st.rerun()

    st.divider()
    
    # Date range filter
    st.markdown("#### 📅 Date Range")
    if 'datetime' in df.columns:
        min_date = df['datetime'].min().date()
        max_date = df['datetime'].max().date()
        date_range = st.date_input(
            "Filter by date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    else:
        date_range = None
    
    st.divider()
    
    # Subreddit filter
    st.markdown("#### 🏘️ Community Filter")
    subreddits = ['All'] + sorted(df['subreddit'].unique().tolist())
    selected_sub = st.selectbox("Subreddit", subreddits)
    
    st.divider()
    
    # Clustering parameters
    st.markdown("#### 🎯 Clustering Parameters")
    suggested_min_cluster = max(5, int(np.sqrt(len(df))))
    min_cluster_size = st.slider(
        "Min Cluster Size",
        min_value=5,
        max_value=max(100, len(df) // 5),
        value=min(suggested_min_cluster, max(100, len(df) // 5)),
        step=5,
        help=f"Minimum posts to form a topic. Suggested: {suggested_min_cluster} (√n heuristic)"
    )
    
    st.divider()

    # Dataset stats
    st.markdown("#### 📊 Dataset Overview")
    stats = get_summary_stats(con)
    if stats:
        live_count = len(st.session_state.live_df) if not st.session_state.live_df.empty else 0
        st.metric("Total Posts", f"{stats.get('total_posts', 0):,}", delta=f"{live_count} Live" if live_count > 0 else None, delta_color="normal")
        st.metric("Unique Authors", f"{stats.get('unique_authors', 0):,}")
        st.metric("Subreddits", f"{stats.get('unique_subreddits', 0):,}")

    st.divider()

    # ── Export Investigation Panel ────────────────────────────────────────────
    st.markdown("#### 📤 Export Investigation")
    st.caption("Download a Markdown summary of the current investigation.")

    def _build_export_report() -> str:
        # NOTE: This runs at sidebar render time, BEFORE filtered_df is defined.
        # Always use `df` (the full dataset) here — it's guaranteed to exist.
        _export_df = df if df is not None else pd.DataFrame()
        lines = ["# NarrativeScope — Investigation Report"]
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} (local)*\n")
        q = st.session_state.get('main_search', '') or '*(no query — full dataset view)*'
        lines.append(f"## Search Query\n> {q}\n")
        lines.append("## Dataset Summary")
        if stats:
            lines.append(f"- Total posts: **{stats.get('total_posts', 'N/A'):,}**")
            lines.append(f"- Unique authors: **{stats.get('unique_authors', 'N/A'):,}**")
            lines.append(f"- Subreddits covered: **{stats.get('unique_subreddits', 'N/A'):,}**")
        lines.append("")
        if not _export_df.empty and 'subreddit' in _export_df.columns:
            top_subs = _export_df['subreddit'].value_counts().head(10)
            lines.append("## Top Subreddits in Current View")
            for sub, cnt in top_subs.items():
                lines.append(f"- r/{sub}: {cnt:,} posts")
            lines.append("")
        if not _export_df.empty and 'author' in _export_df.columns:
            top_authors = _export_df['author'].value_counts().head(5)
            lines.append("## Most Active Authors")
            for author, cnt in top_authors.items():
                if author not in ['[deleted]', 'unknown']:
                    lines.append(f"- u/{author}: {cnt:,} posts")
            lines.append("")
        lines.append("## Investigation Notes")
        lines.append("*Add your notes here.*\n")
        lines.append("---")
        lines.append("*Report generated by [NarrativeScope](https://github.com/devansh)*")
        return "\n".join(lines)

    st.download_button(
        label="⬇️ Download Report (.md)",
        data=_build_export_report(),
        file_name=f"narrativescope_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        use_container_width=True,
        key="export_btn",
    )




# ─── Apply Filters ────────────────────────────────────────────────────────────
filtered_df = df.copy()
if date_range and len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['datetime'].dt.date >= date_range[0]) &
        (filtered_df['datetime'].dt.date <= date_range[1])
    ]
if selected_sub != 'All':
    filtered_df = filtered_df[filtered_df['subreddit'] == selected_sub]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1: SEMANTIC SEARCH & RAG CHATBOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="section-header">
    <h2>🔎 Narrative Search — Ask a Question, Pull a Thread</h2>
</div>
""", unsafe_allow_html=True)

help_card("How does Narrative Search work?", """
<b>In plain English:</b> Normal search looks for your exact words. This is <em>semantic search</em>
— it finds posts that <em>mean the same thing</em>, even with completely different words.<br><br>
<b>Example:</b> Type <em>"government manipulation of public opinion"</em> → finds posts about
<em>"astroturfing campaigns"</em> and <em>"sockpuppet accounts"</em> — zero word overlap, same concept.<br><br>
<b>How?</b> Every post is encoded into 384 numbers (an <em>embedding</em>) that capture its meaning.
Your query gets encoded the same way, and we find posts whose numbers are closest using FAISS
(an ultra-fast vector index). Model: <code>all-MiniLM-L6-v2</code> — 80 sentences/sec on CPU.<br><br>
<table>
  <tr><th>Score</th><th>What it means</th></tr>
  <tr><td>0.85+</td><td>Essentially the same idea, different words</td></tr>
  <tr><td>0.60–0.85</td><td>Strongly related — overlapping framing</td></tr>
  <tr><td>0.40–0.60</td><td>Related — thematically similar</td></tr>
  <tr><td>0.25–0.40</td><td>Weakly related — borderline</td></tr>
  <tr><td>&lt;0.25</td><td>Discarded as irrelevant (threshold)</td></tr>
</table>
""")

def update_search(query):
    st.session_state.main_search = query

col_search, col_examples = st.columns([3, 1])


with col_search:
    search_query = st.text_input(
        "What narrative are you investigating?",
        placeholder="e.g., 'government manipulation of public opinion' or 'distrust in media coverage'",
        key="main_search"
    )

with col_examples:
    st.markdown("**Try these:**")
    for ex in ZERO_OVERLAP_EXAMPLES[:3]:
        st.button(f"🔗 {ex['query'][:35]}...", key=f"ex_{ex['query'][:10]}", on_click=update_search, args=(ex['query'],))

# Process search
if search_query is not None:
    if not search_query.strip():
        st.info("💡 Enter a query above to semantically search the dataset, or explore the full narrative timeline below.")
        analysis_df = filtered_df
    else:
        results, query_info = semantic_search(search_query, model, index, df, embeddings)
        
        # Handle edge cases
        if query_info['status'] == 'empty':
            st.markdown(f'<div class="edge-case-banner">⚠️ {query_info["message"]}</div>', unsafe_allow_html=True)
        elif query_info['status'] == 'short':
            st.markdown(f'<div class="edge-case-banner">⚠️ {query_info["message"]}</div>', unsafe_allow_html=True)
        elif query_info['status'] == 'non_english':
            st.markdown(f'<div class="info-banner">🌐 {query_info["message"]}</div>', unsafe_allow_html=True)
        elif query_info['status'] == 'no_results':
            st.markdown(f'<div class="edge-case-banner">🔍 {query_info["message"]}</div>', unsafe_allow_html=True)
        
        if not results.empty:
            st.markdown(f"**{query_info['num_results']} results** found in {query_info['search_time_ms']}ms (avg similarity: {query_info.get('avg_similarity', 'N/A')})")
            
            # GenAI summary of results
            summary = summarize_search_results(results, search_query)
            st.markdown(f'<div class="genai-summary">{summary}</div>', unsafe_allow_html=True)
            
            # Display results
            for _, row in results.head(10).iterrows():
                sim_score = row.get('similarity_score', 0)
                text_preview = row['text'][:300] + '...' if len(row['text']) > 300 else row['text']
                
                st.markdown(f"""
                <div class="search-result">
                    <span class="similarity-badge">Similarity: {sim_score:.3f}</span>
                    &nbsp; <b>r/{row.get('subreddit', 'unknown')}</b> by <i>u/{row.get('author', 'unknown')}</i>
                    &nbsp; | &nbsp; Score: {row.get('score', 0)} &nbsp; | &nbsp; {row.get('datetime', 'N/A')}
                    <p style="color: #B0B8C8; margin-top: 0.5rem; font-size: 0.9rem;">{text_preview}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Follow-up query suggestions
            if 'suggested_queries' in query_info and query_info['suggested_queries']:
                st.markdown("**Explore further:**")
                cols = st.columns(len(query_info['suggested_queries']))
                for idx, sq in enumerate(query_info['suggested_queries']):
                    with cols[idx]:
                        st.button(f"🔗 {sq}", key=f"sq_{idx}", use_container_width=True, on_click=update_search, args=(sq,))
        
        # Update filtered_df to search results for downstream analysis
        if not results.empty:
            analysis_df = results
        else:
            analysis_df = filtered_df
    


# ── Narrative Health Score card (appears after search results) ────────────────
if search_query and not analysis_df.empty:
    hs = compute_health_score(analysis_df)
    render_health_score(hs, search_query)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2: TIME-SERIES ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="section-header">
    <h2>📈 Temporal Patterns — When Does the Narrative Surge?</h2>
</div>
""", unsafe_allow_html=True)

help_card("How does the time-series analysis work?", """
<b>In plain English:</b> We count how many posts appeared each day, then draw a bar chart.
The teal line is a <em>7-day moving average</em> — it smooths out day-to-day noise so you can see the real trend.<br><br>
<b>What to look for:</b><ul>
<li>A sudden spike → something triggered a flood of posts (check Section 6 for real-world events)</li>
<li>A gradual rise → the topic is gaining organic traction</li>
<li>A flat line → stable background chatter, nothing unusual</li>
</ul>
<b>The subreddit chart below</b> shows which communities were active <em>when</em>, revealing whether a narrative
jumped from one community to another (a classic amplification pattern).
""")

if not analysis_df.empty and 'datetime' in analysis_df.columns:
    # Daily post counts
    daily_counts = analysis_df.groupby(analysis_df['datetime'].dt.date).agg(
        post_count=('text', 'count'),
        unique_authors=('author', 'nunique'),
        avg_score=('score', 'mean')
    ).reset_index()
    daily_counts.columns = ['date', 'post_count', 'unique_authors', 'avg_score']
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    
    # 7-day moving average
    daily_counts['ma_7d'] = daily_counts['post_count'].rolling(7, min_periods=1).mean()
    
    # Time-series plot
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Bar(
        x=daily_counts['date'],
        y=daily_counts['post_count'],
        name='Daily Posts',
        marker_color='rgba(108, 99, 255, 0.4)',
        hovertemplate='%{x}<br>Posts: %{y}<extra></extra>'
    ))
    fig_ts.add_trace(go.Scatter(
        x=daily_counts['date'],
        y=daily_counts['ma_7d'],
        name='7-Day Moving Average',
        line=dict(color='#4ECDC4', width=2),
        hovertemplate='%{x}<br>7d Avg: %{y:.1f}<extra></extra>'
    ))
    
    # Add Wikipedia events if live mode is active and we have a query
    if not st.session_state.live_df.empty and search_query:
        with st.spinner("Fetching contextual Wikipedia events..."):
            wiki_events = fetch_wikipedia_events(search_query, max_events=3)
            if wiki_events and len(daily_counts) > 0:
                # Spread events across the most recent dates
                last_dates = daily_counts['date'].tail(len(wiki_events)).tolist()
                event_x = []
                event_y = []
                event_texts = []
                event_hovers = []
                
                for i, event in enumerate(wiki_events):
                    dt = last_dates[-i-1] if i < len(last_dates) else last_dates[-1]
                    # Get the y value (moving average or post count) at that date to place marker
                    y_val = daily_counts[daily_counts['date'] == dt]['post_count'].max()
                    if pd.isna(y_val):
                        y_val = 0
                        
                    event_x.append(dt)
                    event_y.append(y_val + (y_val * 0.15) + i) # slightly above the bar
                    event_texts.append("📘 " + event['title'][:15] + "..")
                    event_hovers.append(f"<b>{event['title']}</b><br>{event['summary']}<br><a href='{event['url']}'>Wikipedia</a><extra></extra>")
                
                fig_ts.add_trace(go.Scatter(
                    x=event_x,
                    y=event_y,
                    mode='markers+text',
                    name='Real-World Events (Wiki)',
                    marker=dict(symbol='star', size=14, color='#FFD166'),
                    text=event_texts,
                    textposition="top center",
                    hovertemplate=event_hovers,
                    textfont=dict(color='#FFD166')
                ))
    
    fig_ts.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1A1D29',
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Number of Posts",
    )
    
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # GenAI summary beneath the chart
    ts_summary = summarize_timeseries(daily_counts)
    st.markdown(f'<div class="genai-summary">{ts_summary}</div>', unsafe_allow_html=True)
    
    # Subreddit breakdown over time
    if 'subreddit' in analysis_df.columns:
        st.markdown("**Community Activity Over Time**")
        top_subs = analysis_df['subreddit'].value_counts().head(8).index.tolist()
        sub_daily = analysis_df[analysis_df['subreddit'].isin(top_subs)].groupby(
            [analysis_df['datetime'].dt.date, 'subreddit']
        ).size().reset_index(name='count')
        sub_daily.columns = ['date', 'subreddit', 'count']
        
        fig_sub = px.area(
            sub_daily, x='date', y='count', color='subreddit',
            template='plotly_dark',
            labels={'count': 'Posts', 'date': 'Date'},
        )
        fig_sub.update_layout(
            paper_bgcolor='#0E1117',
            plot_bgcolor='#1A1D29',
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_sub, use_container_width=True)
else:
    st.info("No temporal data available for the current selection.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3: TOPIC CLUSTERING & EMBEDDING VISUALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="section-header">
    <h2>🧩 Topic Landscape — What Are People Talking About?</h2>
</div>
""", unsafe_allow_html=True)

help_card("How does topic clustering work?", """
<b>In plain English:</b> Imagine dumping all posts into a room. Posts about similar things
naturally clump together. This section finds those clumps automatically.<br><br>
<b>The pipeline:</b><ol>
<li><b>Embeddings</b> — each post becomes 384 numbers capturing its meaning</li>
<li><b>UMAP</b> — squishes 384 dimensions into 2D for the scatter plot (preserving which posts are "neighbours")</li>
<li><b>HDBSCAN</b> — finds dense clusters without needing you to say how many topics exist</li>
</ol>
<b>Why HDBSCAN?</b> Unlike K-Means, it doesn't force every post into a cluster — outliers become
"noise" (grey dots). This is <em>honest</em>: not every post belongs to a neat topic.<br><br>
<b>Min Cluster Size slider:</b> Lower → more fine-grained topics. Higher → fewer, broader topics.
Default is √n (square root of total posts).
""")

# Initialize cluster tracking variables for downstream components
labels = None
analysis_df_reset = pd.DataFrame()
cluster_labels_dict = {}

if not analysis_df.empty:
    with st.spinner("Clustering topics..."):
        # Get embeddings for analysis subset
        analysis_indices = analysis_df.index.tolist()
        if embeddings is not None and len(analysis_indices) > 0:
            # Use pre-computed embeddings where possible, recompute for filtered subsets
            valid_indices = [i for i in analysis_indices if i < len(embeddings)]
            if valid_indices:
                analysis_embeddings = embeddings[valid_indices]
                analysis_df_reset = analysis_df.loc[valid_indices].reset_index(drop=True)
            else:
                analysis_embeddings = compute_embeddings(model, analysis_df['text'].tolist())
                analysis_df_reset = analysis_df.reset_index(drop=True)
            
            # UMAP reduction
            umap_2d = compute_umap_embeddings(analysis_embeddings, for_visualization=True)
            umap_cluster = compute_umap_embeddings(analysis_embeddings, for_visualization=False)
            
            # HDBSCAN clustering
            labels = cluster_topics(umap_cluster, min_cluster_size=min_cluster_size)
            
            # Extract cluster labels
            cluster_labels_dict = extract_cluster_labels(analysis_df_reset, labels)
            cluster_stats = get_cluster_stats(analysis_df_reset, labels)
            
            # Cluster summary
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Topics Found", n_clusters)
            with col2:
                st.metric("Clustered Posts", f"{(labels != -1).sum():,}")
            with col3:
                st.metric("Unclustered (Noise)", f"{n_noise:,}")
            
            # Edge case messages
            if n_clusters == 0:
                st.warning("⚠️ No clusters formed. All posts were classified as noise. Try **decreasing** the minimum cluster size.")
            elif n_clusters == 1:
                st.info("ℹ️ Only one cluster formed — the dataset may be very homogeneous, or the min cluster size is too high.")
            elif n_noise / len(labels) > 0.6:
                st.info(f"ℹ️ {n_noise/len(labels)*100:.0f}% of posts are unclustered. Consider decreasing the min cluster size for more granular topics.")
            
            # Embedding visualization
            # Try Datamapplot first, fall back to Plotly
            datamapplot_html = create_datamapplot_html(
                umap_2d, labels, cluster_labels_dict, analysis_df_reset['text'].tolist()
            ) if len(labels) > 0 else None
            
            if datamapplot_html:
                st.components.v1.html(datamapplot_html, height=700, scrolling=True)
            else:
                fig_embed = create_embedding_visualization(
                    umap_2d, labels, analysis_df_reset, cluster_labels_dict
                )
                st.plotly_chart(fig_embed, use_container_width=True)
            
            # GenAI summary
            topic_summary = summarize_topics(cluster_stats, cluster_labels_dict)
            st.markdown(f'<div class="genai-summary">{topic_summary}</div>', unsafe_allow_html=True)
            
            # Cluster details table
            if not cluster_stats.empty:
                st.markdown("**Topic Details**")
                display_stats = cluster_stats.copy()
                display_stats['topic'] = display_stats['cluster'].map(
                    lambda x: cluster_labels_dict.get(x, f"Cluster {x}")
                )
                st.dataframe(
                    display_stats[['topic', 'post_count', 'unique_authors', 'unique_subreddits', 'date_range']],
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning("Embedding data not available for clustering.")
else:
    st.info("No data available for clustering with the current filters.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: NETWORK ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="section-header">
    <h2>🕸️ Influence Network — Who Amplifies the Narrative?</h2>
</div>
""", unsafe_allow_html=True)

help_card("How does the influence network work?", """
<b>In plain English:</b> If two Reddit accounts both posted in the same subreddit, we draw a line
between them. The more subreddits they share, the thicker the line. The result is a map of
<em>who hangs out with whom</em>.<br><br>
<b>Key metrics:</b>
<table>
  <tr><th>Metric</th><th>Meaning</th></tr>
  <tr><td>PageRank</td><td>Importance — accounts connected to other important accounts score high</td></tr>
  <tr><td>Betweenness</td><td>Bridge — accounts that connect otherwise-separate groups</td></tr>
  <tr><td>Degree</td><td>Connectivity — how many direct neighbours</td></tr>
</table><br>
<b>Colours</b> = Louvain communities (groups of tightly connected accounts).<br>
<b>Robustness test:</b> Remove an account and see if the network falls apart — if it does,
that account was a critical amplifier.
""")

if not analysis_df.empty and 'author' in analysis_df.columns:
    with st.spinner("Building network graph..."):
        G = build_cooccurrence_network(analysis_df, groupby_col='subreddit', node_col='author')
        
        if G.number_of_nodes() > 0:
            centrality_df = compute_centrality_metrics(G)
            partition = detect_communities(G)
            n_communities = len(set(partition.values())) if partition else 0
            
            # Network stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accounts", G.number_of_nodes())
            with col2:
                st.metric("Connections", G.number_of_edges())
            with col3:
                st.metric("Communities", n_communities)
            with col4:
                st.metric("Density", f"{nx.density(G):.4f}" if G.number_of_nodes() > 1 else "N/A")
            
            # Network visualization
            network_html = generate_pyvis_html(G, partition, centrality_df)
            st.components.v1.html(network_html, height=620, scrolling=False)
            
            # GenAI summary
            net_summary = summarize_network(centrality_df, n_communities)
            st.markdown(f'<div class="genai-summary">{net_summary}</div>', unsafe_allow_html=True)
            
            # Top influencers table
            if not centrality_df.empty:
                st.markdown("**Top Influential Accounts**")
                display_centrality = centrality_df.head(15).copy()
                display_centrality['pagerank'] = display_centrality['pagerank'].round(6)
                display_centrality['betweenness'] = display_centrality['betweenness'].round(6)
                display_centrality['influence_score'] = display_centrality['influence_score'].round(4)
                
                st.dataframe(
                    display_centrality[['account', 'influence_score', 'pagerank', 'betweenness', 'degree', 'post_count']],
                    use_container_width=True,
                    hide_index=True,
                )
            
            # Node removal simulation
            st.markdown("**🧪 Robustness Test: What Happens If a Key Account Is Removed?**")
            if not centrality_df.empty:
                removable_nodes = centrality_df.head(20)['account'].tolist()
                node_to_remove = st.selectbox(
                    "Select an account to remove",
                    removable_nodes,
                    help="See how the network fragments when a key amplifier is removed"
                )
                
                if st.button("Simulate Removal", key="remove_node"):
                    removal_results = simulate_node_removal(G, node_to_remove)
                    
                    if 'error' not in removal_results:
                        col_before, col_after = st.columns(2)
                        with col_before:
                            st.markdown("**Before Removal**")
                            st.write(f"- Nodes: {removal_results['before']['nodes']}")
                            st.write(f"- Edges: {removal_results['before']['edges']}")
                            st.write(f"- Components: {removal_results['before']['components']}")
                        with col_after:
                            st.markdown("**After Removal**")
                            st.write(f"- Nodes: {removal_results['after']['nodes']}")
                            st.write(f"- Edges: {removal_results['after']['edges']}")
                            st.write(f"- Components: {removal_results['after']['components']}")
                        
                        if removal_results['fragmentation'] > 0:
                            st.warning(f"⚡ Removing **{node_to_remove}** fragmented the network into {removal_results['fragmentation']} additional components, disconnecting {removal_results['after']['isolated_nodes_removed']} accounts entirely.")
                        else:
                            st.info(f"Network remained connected after removing {node_to_remove}. This suggests the network is resilient — no single account is a bottleneck.")
                        
                        # Show the modified graph
                        if removal_results['after']['nodes'] > 0:
                            G_after = removal_results['graph_after']
                            partition_after = detect_communities(G_after)
                            centrality_after = compute_centrality_metrics(G_after)
                            html_after = generate_pyvis_html(G_after, partition_after, centrality_after, height="400px")
                            st.components.v1.html(html_after, height=420, scrolling=False)
                    else:
                        st.error(removal_results['error'])
        else:
            st.info("Not enough connected accounts to build a meaningful network. Try broadening your filters.")
else:
    st.info("No author data available for network analysis.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4b: ANOMALY DETECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="section-header">
    <h2>⚠️ Anomaly Detector — Coordination Signals</h2>
</div>
""", unsafe_allow_html=True)

help_card("How does the anomaly detector work?", """
<b>In plain English:</b> This section automatically scans the data for patterns that
<em>could</em> indicate coordinated inauthentic behaviour — bots, spam rings, or astroturfing.
<br><br>
<b>Three checks run automatically:</b>
<table>
  <tr><th>Check</th><th>What it flags</th></tr>
  <tr><td>🔁 Cross-Subreddit</td><td>An account posting in &gt;5 different subreddits (unusual spread pattern)</td></tr>
  <tr><td>📈 Activity Spike</td><td>A day with Z-score &gt;2.5σ above the rolling mean (statistically rare surge)</td></tr>
  <tr><td>📋 Near-Duplicate</td><td>3+ different accounts posting near-identical text (coordination signal)</td></tr>
</table><br>
<b>Important:</b> These are <em>signals</em>, not proof. An active user might legitimately post
across many subreddits. Always investigate further before drawing conclusions.
""")

if not analysis_df.empty:
    anomaly_flags = detect_anomalies(analysis_df)
    if anomaly_flags:
        st.markdown(f"**{len(anomaly_flags)} potential anomalies detected**")
        for flag in anomaly_flags:
            severity_color = '#FF6B6B' if flag['severity'] == 'high' else '#FFD166'
            st.markdown(f"""
            <div class="anomaly-card">
                <div style="color:{severity_color};font-weight:600;font-size:0.95rem;">
                    {flag['type']}: {flag['title']}
                </div>
                <div class="anomaly-detail">{flag['detail']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-banner">✅ No obvious coordination signals detected in the current data.
        This doesn't rule out sophisticated activity — it means the three automated checks came back clean.</div>
        """, unsafe_allow_html=True)
else:
    st.info("No data available for anomaly detection.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: THREAD PULLER (WOW FEATURE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="section-header">
    <h2>🧵 Thread Puller — Trace a Narrative's Journey</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="insight-card">
    <p style="color: #8892b0;">Select a topic cluster above to trace its narrative lifecycle:
    when it emerged, how it spread across communities, and whether the framing shifted over time.</p>
</div>
""", unsafe_allow_html=True)

if not analysis_df.empty and labels is not None and len(labels) > 0:
    # Let user select a cluster to investigate
    valid_clusters = sorted([c for c in set(labels) if c != -1])
    
    if valid_clusters:
        selected_cluster_name = st.selectbox(
            "Select a topic to trace",
            [cluster_labels_dict.get(c, f"Cluster {c}") for c in valid_clusters],
            key="thread_puller_cluster"
        )
        
        # Find the cluster ID from the name
        selected_cluster_id = valid_clusters[0]
        for c in valid_clusters:
            if cluster_labels_dict.get(c, f"Cluster {c}") == selected_cluster_name:
                selected_cluster_id = c
                break
        
        # Get posts in this cluster
        cluster_mask = labels == selected_cluster_id
        cluster_posts = analysis_df_reset[cluster_mask].copy()
        
        if not cluster_posts.empty:
            # Thread timeline
            st.markdown("**📅 Narrative Timeline**")
            thread_daily = cluster_posts.groupby(cluster_posts['datetime'].dt.date).size().reset_index(name='posts')
            thread_daily.columns = ['date', 'posts']
            
            fig_thread = go.Figure()
            fig_thread.add_trace(go.Scatter(
                x=thread_daily['date'],
                y=thread_daily['posts'],
                fill='tozeroy',
                fillcolor='rgba(108, 99, 255, 0.2)',
                line=dict(color='#6C63FF', width=2),
                mode='lines',
                hovertemplate='%{x}<br>Posts: %{y}<extra></extra>'
            ))
            fig_thread.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0E1117',
                plot_bgcolor='#1A1D29',
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Date",
                yaxis_title="Posts in Thread",
            )
            st.plotly_chart(fig_thread, use_container_width=True)
            
            # Community crossover
            st.markdown("**🏘️ Community Crossover**")
            sub_breakdown = cluster_posts['subreddit'].value_counts().head(10)
            fig_crossover = px.bar(
                x=sub_breakdown.values,
                y=sub_breakdown.index,
                orientation='h',
                labels={'x': 'Posts', 'y': 'Subreddit'},
                template='plotly_dark',
            )
            fig_crossover.update_traces(marker_color='#4ECDC4')
            fig_crossover.update_layout(
                paper_bgcolor='#0E1117',
                plot_bgcolor='#1A1D29',
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_crossover, use_container_width=True)
            
            # Key voices in this thread
            st.markdown("**🎙️ Key Voices in This Thread**")
            top_voices = cluster_posts.groupby('author').agg(
                posts=('text', 'count'),
                avg_score=('score', 'mean')
            ).sort_values('posts', ascending=False).head(10)
            st.dataframe(top_voices, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6: EVENT CORRELATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="section-header">
    <h2>🌍 Real-World Event Correlation</h2>
</div>
""", unsafe_allow_html=True)

wiki_topic = st.text_input(
    "Enter a topic to find related real-world events",
    placeholder="e.g., 'United States elections 2024' or 'climate change policy'",
    key="wiki_search"
)

if wiki_topic:
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(
            user_agent='NarrativeScope/1.0 (https://github.com/devansh; research)',
            language='en'
        )
        page = wiki.page(wiki_topic)
        
        if page.exists():
            st.markdown(f"**📝 Wikipedia: {page.title}**")
            # Show first 500 chars as context
            summary_text = page.summary[:500] + "..." if len(page.summary) > 500 else page.summary
            st.markdown(f'<div class="insight-card"><p style="color: #B0B8C8;">{summary_text}</p></div>', unsafe_allow_html=True)
            st.markdown(f"[Read full article →]({page.fullurl})")
        else:
            st.info(f"No Wikipedia article found for '{wiki_topic}'. Try a different search term.")
    except ImportError:
        st.info("Wikipedia API not installed. Install it with: `pip install wikipedia-api`")
    except Exception as e:
        st.warning(f"Could not fetch Wikipedia data: {e}")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align: center; color: #555; padding: 1rem;">
    <p><b>NarrativeScope</b> — Built by Devansh for SimPPL Research Engineering Internship</p>
    <p style="font-size: 0.8rem;">
        Powered by sentence-transformers, HDBSCAN, UMAP, NetworkX, DuckDB, and Google Gemini
    </p>
</div>
""", unsafe_allow_html=True)


