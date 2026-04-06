"""
clustering.py — Topic clustering with HDBSCAN + UMAP + Datamapplot visualization.

Implements tunable topic clustering with robust handling at parameter extremes.
HDBSCAN chosen over K-means because:
1. Doesn't assume spherical clusters (text embeddings form irregular shapes)
2. Naturally handles noise (not every post belongs to a coherent topic)
3. min_cluster_size is more interpretable than k: "how many posts make a topic?"
"""

import numpy as np
import pandas as pd
import streamlit as st
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
import logging
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

# UMAP parameters
# n_neighbors=15: balances local vs global structure preservation
# min_dist=0.1: allows tight clusters while keeping some spread for visualization
# n_components: 2 for viz, 15 for clustering (higher dim preserves more structure for HDBSCAN)
UMAP_VIZ_PARAMS = {
    'n_components': 2,
    'n_neighbors': 15,
    'min_dist': 0.1,
    'metric': 'cosine',
    'random_state': 42,
    'low_memory': True,
}

UMAP_CLUSTER_PARAMS = {
    'n_components': 15,
    'n_neighbors': 15,
    'min_dist': 0.0,
    'metric': 'cosine',
    'random_state': 42,
    'low_memory': True,
}


@st.cache_data(show_spinner="Reducing dimensions with UMAP...")
def compute_umap_embeddings(embeddings_array: np.ndarray, for_visualization: bool = True) -> np.ndarray:
    """
    Compute UMAP dimensionality reduction.
    
    Two modes:
    - for_visualization=True: 2D for scatter plots
    - for_visualization=False: 15D for HDBSCAN input (higher dim preserves more structure)
    """
    if embeddings_array.shape[0] == 0:
        return np.array([])
    
    params = UMAP_VIZ_PARAMS if for_visualization else UMAP_CLUSTER_PARAMS
    
    # For small datasets, adjust n_neighbors
    n_samples = embeddings_array.shape[0]
    if n_samples < params['n_neighbors'] * 2:
        params = params.copy()
        params['n_neighbors'] = max(2, n_samples // 3)
    
    reducer = umap.UMAP(**params)
    reduced = reducer.fit_transform(embeddings_array)
    
    return reduced


def cluster_topics(embeddings_for_clustering: np.ndarray, 
                   min_cluster_size: int = 15,
                   min_samples: int = 5) -> np.ndarray:
    """
    Cluster posts using HDBSCAN.
    
    Parameters:
    - min_cluster_size: minimum number of posts to form a topic cluster (tunable by user)
    - min_samples: minimum density neighborhood size (kept fixed for simplicity)
    
    Returns cluster labels (-1 = noise/unclustered)
    """
    if embeddings_for_clustering.shape[0] == 0:
        return np.array([])
    
    # Adjust min_samples if needed
    actual_min_samples = min(min_samples, min_cluster_size)
    
    # Handle extreme values gracefully
    n_samples = embeddings_for_clustering.shape[0]
    if min_cluster_size >= n_samples:
        logger.warning(f"min_cluster_size ({min_cluster_size}) >= dataset size ({n_samples}). All points will be noise.")
        return np.full(n_samples, -1)
    
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, min_cluster_size),
            min_samples=max(1, actual_min_samples),
            metric='euclidean',
            cluster_selection_method='eom',  # Excess of Mass — better for varying density clusters
            prediction_data=False,
        )
        labels = clusterer.fit_predict(embeddings_for_clustering)
    except Exception as e:
        logger.error(f"HDBSCAN clustering failed: {e}")
        return np.full(embeddings_for_clustering.shape[0], -1)
    
    # Log cluster stats
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels)*100:.1f}%)")
    
    return labels


def extract_cluster_labels(df: pd.DataFrame, labels: np.ndarray, top_n_words: int = 5) -> dict:
    """
    Extract human-readable labels for each cluster using TF-IDF.
    
    For each cluster, find the most distinctive terms compared to other clusters.
    """
    if len(labels) == 0 or df.empty:
        return {}
    
    cluster_labels = {}
    unique_clusters = sorted(set(labels))
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_labels[-1] = "Unclustered / Noise"
            continue
        
        cluster_mask = labels == cluster_id
        cluster_texts = df.loc[cluster_mask, 'text'].tolist()
        
        if not cluster_texts:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
            continue
        
        try:
            # Use TF-IDF to find distinctive terms
            tfidf = TfidfVectorizer(
                max_features=100, 
                stop_words='english',
                max_df=0.9,
                min_df=1,
                ngram_range=(1, 2)
            )
            tfidf_matrix = tfidf.fit_transform(cluster_texts)
            
            # Get top terms by average TF-IDF score
            feature_names = tfidf.get_feature_names_out()
            avg_scores = tfidf_matrix.mean(axis=0).A1
            top_indices = avg_scores.argsort()[-top_n_words:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            cluster_labels[cluster_id] = ', '.join(top_terms)
        except Exception:
            cluster_labels[cluster_id] = f"Topic {cluster_id}"
    
    return cluster_labels


def get_cluster_stats(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Get statistics for each cluster."""
    if len(labels) == 0:
        return pd.DataFrame()
    
    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels
    
    stats = df_with_labels.groupby('cluster').agg(
        post_count=('text', 'count'),
        avg_score=('score', 'mean'),
        unique_authors=('author', 'nunique'),
        unique_subreddits=('subreddit', 'nunique'),
        date_range=('datetime', lambda x: f"{x.min().strftime('%Y-%m-%d')} to {x.max().strftime('%Y-%m-%d')}" if len(x) > 0 else "N/A")
    ).reset_index()
    
    stats = stats.sort_values('post_count', ascending=False)
    return stats


def create_embedding_visualization(umap_2d: np.ndarray, labels: np.ndarray, 
                                     df: pd.DataFrame, cluster_labels: dict) -> go.Figure:
    """
    Create an interactive Plotly scatter plot of the topic embeddings.
    
    This serves as the primary topic visualization when Datamapplot is unavailable.
    Each point is a post, colored by cluster, with hover info.
    """
    if umap_2d.shape[0] == 0:
        return go.Figure()
    
    plot_df = pd.DataFrame({
        'x': umap_2d[:, 0],
        'y': umap_2d[:, 1],
        'cluster': labels,
        'cluster_name': [cluster_labels.get(l, f"Cluster {l}") for l in labels],
        'text_preview': [t[:100] + '...' if len(t) > 100 else t for t in df['text'].values[:len(labels)]],
        'author': df['author'].values[:len(labels)] if 'author' in df.columns else ['unknown'] * len(labels),
        'subreddit': df['subreddit'].values[:len(labels)] if 'subreddit' in df.columns else ['unknown'] * len(labels),
    })
    
    # Color noise points differently
    plot_df['is_noise'] = plot_df['cluster'] == -1
    
    fig = px.scatter(
        plot_df[~plot_df['is_noise']],
        x='x', y='y',
        color='cluster_name',
        hover_data=['text_preview', 'author', 'subreddit'],
        title='Topic Embedding Space',
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 'cluster_name': 'Topic'},
        opacity=0.7,
    )
    
    # Add noise points in grey
    noise = plot_df[plot_df['is_noise']]
    if not noise.empty:
        fig.add_trace(go.Scatter(
            x=noise['x'], y=noise['y'],
            mode='markers',
            marker=dict(color='#555555', size=3, opacity=0.3),
            name='Unclustered',
            hoverinfo='text',
            text=noise['text_preview'],
        ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1A1D29',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig


def create_datamapplot_html(umap_2d: np.ndarray, labels: np.ndarray, 
                              cluster_labels: dict, texts: list) -> str:
    """
    Create a Datamapplot interactive visualization.
    
    Datamapplot generates beautiful, publication-quality topic maps with 
    automatic cluster labeling — more visually distinctive than basic scatter plots.
    """
    try:
        import datamapplot
        
        # Create label array for datamapplot
        label_array = np.array([
            cluster_labels.get(l, "Unclustered") if l != -1 else "Unclustered" 
            for l in labels
        ])
        
        # Generate the plot
        fig, ax = datamapplot.create_plot(
            umap_2d,
            label_array,
            title="NarrativeScope — Topic Landscape",
            sub_title="Posts clustered by semantic similarity",
            darkmode=True,
            label_font_size=10,
            point_size=5,
            noise_label="Unclustered",
            figsize=(14, 10),
        )
        
        # Save to HTML-compatible format
        import io
        import base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                    facecolor='#0E1117', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        html = f'<img src="data:image/png;base64,{img_base64}" style="width:100%;border-radius:8px;" />'
        return html
        
    except ImportError:
        logger.info("Datamapplot not available, falling back to Plotly visualization")
        return None
    except Exception as e:
        logger.warning(f"Datamapplot rendering failed: {e}, falling back to Plotly")
        return None
