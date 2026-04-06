"""
search_engine.py — Semantic search with FAISS + sentence-transformers.

Implements embedding-based similarity search that works with zero keyword overlap.
Includes edge case handling for empty queries, short queries, and non-English input.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import sys
sys.modules['tensorflow'] = None
sys.modules['tensorflow.compat.v2.compiler.tensorrt'] = None

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from langdetect import detect, LangDetectException
import logging
import time

logger = logging.getLogger(__name__)

# Model: all-MiniLM-L6-v2 — 384-dim embeddings, ~80 sentences/sec on CPU
# Chosen for speed/quality balance: good semantic understanding, fast enough for free-tier deployment
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MIN_QUERY_LENGTH = 3
SIMILARITY_THRESHOLD = 0.25  # Minimum cosine similarity for a result to be "relevant"


@st.cache_resource(show_spinner="Loading semantic search model...")
def load_embedding_model():
    """Load the sentence-transformers model (cached across sessions)."""
    model = SentenceTransformer(MODEL_NAME)
    return model


@st.cache_data(show_spinner="Building search index (this takes a moment on first run)...")
def compute_embeddings(_model, texts: list) -> np.ndarray:
    """
    Compute embeddings for all texts in the dataset.
    Uses batched encoding for efficiency.
    """
    if not texts:
        return np.array([]).reshape(0, EMBEDDING_DIM)
    
    # Truncate very long texts to first 512 chars (model max is 256 tokens ≈ ~400 chars)
    truncated_texts = [t[:512] if len(t) > 512 else t for t in texts]
    
    embeddings = _model.encode(
        truncated_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize so inner product = cosine similarity
        convert_to_numpy=True
    )
    
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index for fast similarity search.
    Uses IndexFlatIP (inner product) on normalized vectors = cosine similarity.
    """
    if embeddings.shape[0] == 0:
        return None
    
    # Ensure float32 for FAISS
    embeddings = embeddings.astype(np.float32)
    
    # IndexFlatIP for exact inner product search (cosine sim on normalized vectors)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    
    logger.info(f"Built FAISS index with {index.ntotal} vectors")
    return index


def detect_language(text: str) -> tuple:
    """
    Detect the language of input text.
    Returns (language_code, is_english).
    """
    try:
        lang = detect(text)
        return lang, lang == 'en'
    except LangDetectException:
        return 'unknown', True  # Default to treating as English


def validate_query(query: str) -> tuple:
    """
    Validate a search query and return (is_valid, message).
    Handles: empty queries, very short queries, non-English input.
    """
    if not query or not query.strip():
        return False, "empty", "Please enter a search query. Try something like: 'misinformation campaigns' or 'election discourse'"
    
    query = query.strip()
    
    if len(query) < MIN_QUERY_LENGTH:
        return False, "short", f"Query is too short (minimum {MIN_QUERY_LENGTH} characters). Try a more descriptive query for better results."
    
    # Check for non-English input
    lang, is_english = detect_language(query)
    if not is_english and lang != 'unknown':
        # We still process it — MiniLM handles some multilingual input
        return True, "non_english", f"Detected language: {lang}. Results may be less accurate for non-English queries, but I'll do my best."
    
    return True, "ok", ""


def semantic_search(query: str, model, index: faiss.Index, 
                     df: pd.DataFrame, embeddings: np.ndarray,
                     top_k: int = 20) -> tuple:
    """
    Perform semantic search and return ranked results.
    
    Returns:
        (results_df, query_info_dict)
    """
    # Validate query
    is_valid, status, message = validate_query(query)
    
    if not is_valid:
        return pd.DataFrame(), {
            'status': status,
            'message': message,
            'query': query,
            'num_results': 0
        }
    
    if index is None or index.ntotal == 0:
        return pd.DataFrame(), {
            'status': 'error',
            'message': 'Search index is not initialized. Please reload the data.',
            'query': query,
            'num_results': 0
        }
    
    start_time = time.time()
    
    # Encode query
    query_embedding = model.encode(
        [query.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)
    
    # Search FAISS
    actual_k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, actual_k)
    
    scores = scores[0]
    indices = indices[0]
    
    # Filter by similarity threshold
    mask = scores >= SIMILARITY_THRESHOLD
    filtered_scores = scores[mask]
    filtered_indices = indices[mask]
    
    search_time = time.time() - start_time
    
    if len(filtered_indices) == 0:
        return pd.DataFrame(), {
            'status': 'no_results',
            'message': f'No posts found with sufficient semantic similarity (threshold: {SIMILARITY_THRESHOLD}). Try a different phrasing or broader query.',
            'query': query,
            'num_results': 0,
            'search_time_ms': round(search_time * 1000, 1),
            'suggested_queries': generate_suggested_queries(query)
        }
    
    # Build results DataFrame
    results = df.iloc[filtered_indices].copy()
    results['similarity_score'] = filtered_scores
    results['rank'] = range(1, len(results) + 1)
    results = results.sort_values('similarity_score', ascending=False)
    
    lang, is_english = detect_language(query)
    
    return results, {
        'status': 'success' if status == 'ok' else status,
        'message': message if status == 'non_english' else f'Found {len(results)} semantically relevant posts',
        'query': query,
        'num_results': len(results),
        'search_time_ms': round(search_time * 1000, 1),
        'avg_similarity': round(float(filtered_scores.mean()), 3),
        'suggested_queries': generate_suggested_queries(query),
        'language': lang
    }


def generate_suggested_queries(query: str) -> list:
    """
    Generate 2-3 related follow-up queries based on the original query.
    Uses query expansion heuristics (not AI-generated, for speed).
    """
    query_lower = query.lower().strip()
    
    # Base suggestions based on common investigation patterns
    suggestions = []
    
    # Pattern 1: If query is about a topic, suggest related angles
    topic_expansions = {
        'election': ['voter suppression tactics', 'campaign misinformation', 'electoral integrity debates'],
        'misinformation': ['fact-checking community response', 'media literacy discussions', 'source credibility analysis'],
        'climate': ['environmental policy debate', 'renewable energy discourse', 'climate denial narratives'],
        'vaccine': ['public health messaging', 'medical misinformation spread', 'pharmaceutical trust discussion'],
        'immigration': ['border policy perspectives', 'refugee discourse analysis', 'labor market impact narratives'],
        'ai': ['artificial intelligence ethics debate', 'automation job displacement fears', 'AI regulation perspectives'],
        'war': ['conflict media framing', 'peace negotiation discourse', 'humanitarian crisis narratives'],
        'economy': ['inflation impact discussions', 'wealth inequality narratives', 'economic policy polarization'],
        'police': ['criminal justice reform debate', 'community safety discourse', 'accountability narratives'],
        'protest': ['civil disobedience perspectives', 'social movement amplification', 'counter-protest narratives'],
    }
    
    for keyword, expansions in topic_expansions.items():
        if keyword in query_lower:
            suggestions.extend(expansions[:2])
            break
    
    # Pattern 2: Suggest different framing
    if len(suggestions) < 2:
        suggestions.append(f"Who are the most influential voices discussing {query}?")
    if len(suggestions) < 3:
        suggestions.append(f"How has the narrative around {query} changed over time?")
    
    return suggestions[:3]


# Pre-defined examples of zero-keyword-overlap semantic search
ZERO_OVERLAP_EXAMPLES = [
    {
        'query': 'government manipulation of public opinion',
        'expected_match_concept': 'astroturfing, sockpuppet accounts, coordinated inauthentic behavior',
        'why': 'Semantic embedding captures the concept of institutional actors shaping discourse, even though none of these words appear in the query.'
    },
    {
        'query': 'people losing trust in mainstream journalism',
        'expected_match_concept': 'media bias, fake news accusations, alternative news sources',
        'why': 'The embedding space maps "losing trust in journalism" near posts about media criticism, even with completely different vocabulary.'
    },
    {
        'query': 'spreading false health information online',
        'expected_match_concept': 'anti-vax content, unverified medical claims, wellness misinformation',
        'why': 'The model understands "false health information" is semantically equivalent to misinformation about medicine/health, matching related discussions.'
    }
]
