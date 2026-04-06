"""
network_analysis.py — Network graph construction, centrality analysis, and visualization.

Builds co-occurrence networks from Reddit data, computes PageRank and betweenness centrality,
detects communities using Louvain, and generates interactive PyVis visualizations.
"""

import networkx as nx
from community import community_louvain
import pandas as pd
import numpy as np
from pyvis.network import Network
import streamlit as st
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

# Graph construction parameters
MIN_EDGE_WEIGHT = 2  # Minimum co-occurrences to create an edge
MAX_NODES = 200  # Cap for visualization performance
PAGERANK_ALPHA = 0.85  # Damping factor (standard value, well-studied)


def build_cooccurrence_network(df: pd.DataFrame, 
                                 groupby_col: str = 'subreddit',
                                 node_col: str = 'author',
                                 min_edge_weight: int = MIN_EDGE_WEIGHT) -> nx.Graph:
    """
    Build a co-occurrence network where nodes are authors and edges connect
    authors who post in the same subreddit (or topic cluster).
    
    Edge weight = number of shared context co-occurrences.
    
    Why co-occurrence over reply networks:
    - Reddit JSONL data often lacks reply-to relationships
    - Co-occurrence captures coordinated behavior patterns
    - More robust to missing data fields
    """
    if df.empty or node_col not in df.columns:
        return nx.Graph()
    
    # Filter out deleted/unknown authors
    filtered = df[
        (df[node_col] != '[deleted]') & 
        (df[node_col] != 'unknown') &
        (df[node_col] != 'AutoModerator')
    ].copy()
    
    if filtered.empty:
        return nx.Graph()
    
    G = nx.Graph()
    
    # Group by context (subreddit) and find co-occurring authors
    for group_name, group_df in filtered.groupby(groupby_col):
        authors = group_df[node_col].unique()
        
        # Skip very small or very large groups (noise or too generic)
        if len(authors) < 2 or len(authors) > 500:
            continue
        
        # Add nodes with metadata
        for author in authors:
            if not G.has_node(author):
                author_data = group_df[group_df[node_col] == author]
                G.add_node(author, 
                          post_count=len(author_data),
                          avg_score=float(author_data['score'].mean()) if 'score' in author_data else 0,
                          subreddits=set())
            G.nodes[author]['subreddits'].add(group_name)
        
        # Create edges between co-occurring authors (efficiently)
        # Only connect pairs, not all permutations (too expensive for large groups)
        author_list = list(authors)
        for i in range(min(len(author_list), 50)):  # Cap to prevent O(n²) explosion
            for j in range(i + 1, min(len(author_list), 50)):
                a1, a2 = author_list[i], author_list[j]
                if G.has_edge(a1, a2):
                    G[a1][a2]['weight'] += 1
                else:
                    G.add_edge(a1, a2, weight=1, context=group_name)
    
    # Remove weak edges
    weak_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_edge_weight]
    G.remove_edges_from(weak_edges)
    
    # Remove isolated nodes after edge pruning
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    
    # Cap graph size for visualization
    if G.number_of_nodes() > MAX_NODES:
        # Keep top nodes by degree
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:MAX_NODES]
        top_node_set = {n for n, _ in top_nodes}
        G = G.subgraph(top_node_set).copy()
    
    logger.info(f"Built network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_centrality_metrics(G: nx.Graph) -> pd.DataFrame:
    """
    Compute multiple centrality metrics for network analysis.
    
    PageRank: recursive influence — an account is influential if other influential 
    accounts co-occur with it. More nuanced than simple degree centrality.
    
    Betweenness: identifies bridge accounts connecting different communities.
    These are often key narrative propagators between subreddits.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    
    metrics = {}
    
    # PageRank — primary influence metric
    # alpha=0.85 is the standard damping factor used in the original paper
    try:
        pagerank = nx.pagerank(G, alpha=PAGERANK_ALPHA, weight='weight')
        metrics['pagerank'] = pagerank
    except nx.NetworkXError as e:
        logger.warning(f"PageRank computation failed: {e}")
        metrics['pagerank'] = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}
    
    # Betweenness centrality — bridge detection
    try:
        betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
        metrics['betweenness'] = betweenness
    except Exception as e:
        logger.warning(f"Betweenness computation failed: {e}")
        metrics['betweenness'] = {n: 0.0 for n in G.nodes()}
    
    # Degree centrality — simple connectivity measure
    degree_cent = nx.degree_centrality(G)
    metrics['degree_centrality'] = degree_cent
    
    # Build results DataFrame
    results = pd.DataFrame({
        'account': list(G.nodes()),
        'pagerank': [metrics['pagerank'].get(n, 0) for n in G.nodes()],
        'betweenness': [metrics['betweenness'].get(n, 0) for n in G.nodes()],
        'degree_centrality': [degree_cent.get(n, 0) for n in G.nodes()],
        'degree': [G.degree(n) for n in G.nodes()],
        'post_count': [G.nodes[n].get('post_count', 0) for n in G.nodes()],
    })
    
    # Composite influence score (weighted combination)
    results['influence_score'] = (
        0.5 * results['pagerank'] / (results['pagerank'].max() or 1) +
        0.3 * results['betweenness'] / (results['betweenness'].max() or 1) +
        0.2 * results['degree_centrality'] / (results['degree_centrality'].max() or 1)
    )
    
    results = results.sort_values('influence_score', ascending=False)
    return results


def detect_communities(G: nx.Graph) -> dict:
    """
    Detect communities using Louvain method.
    
    Why Louvain: Fast, deterministic, works well on co-occurrence networks,
    and produces interpretable communities that correspond to discourse groups.
    """
    if G.number_of_nodes() == 0:
        return {}
    
    try:
        partition = community_louvain.best_partition(G, resolution=1.0, random_state=42)
        
        # Log community stats
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        logger.info(f"Detected {len(communities)} communities")
        return partition
    except Exception as e:
        logger.warning(f"Community detection failed: {e}")
        return {n: 0 for n in G.nodes()}


def simulate_node_removal(G: nx.Graph, node_to_remove: str) -> dict:
    """
    Simulate removing a highly connected node to test network robustness.
    Returns stats about the network before and after removal.
    
    This demonstrates: "What happens if a key amplifier is removed?"
    Handles disconnected components gracefully.
    """
    if node_to_remove not in G.nodes():
        return {'error': f'Node "{node_to_remove}" not found in network'}
    
    # Before removal stats
    before = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'components': nx.number_connected_components(G),
        'density': round(nx.density(G), 4),
    }
    
    # Remove node
    G_after = G.copy()
    G_after.remove_node(node_to_remove)
    
    # Remove any resulting isolates
    isolates = list(nx.isolates(G_after))
    G_after.remove_nodes_from(isolates)
    
    # After removal stats
    after = {
        'nodes': G_after.number_of_nodes(),
        'edges': G_after.number_of_edges(),
        'components': nx.number_connected_components(G_after) if G_after.number_of_nodes() > 0 else 0,
        'density': round(nx.density(G_after), 4) if G_after.number_of_nodes() > 1 else 0,
        'isolated_nodes_removed': len(isolates),
    }
    
    return {
        'before': before,
        'after': after,
        'removed_node': node_to_remove,
        'fragmentation': after['components'] - before['components'],
        'edges_lost': before['edges'] - after['edges'],
        'graph_after': G_after
    }


def generate_pyvis_html(G: nx.Graph, partition: dict = None, 
                         centrality_df: pd.DataFrame = None,
                         height: str = "600px") -> str:
    """
    Generate an interactive PyVis network visualization.
    
    Nodes sized by PageRank, colored by community.
    Edges weighted by co-occurrence strength.
    """
    if G.number_of_nodes() == 0:
        return "<p style='color: #999; text-align: center;'>No network data to display.</p>"
    
    net = Network(
        height=height, 
        width="100%",
        bgcolor="#0E1117",
        font_color="#E0E0E0",
        directed=False,
        notebook=False
    )
    
    # Color palette for communities
    colors = [
        '#6C63FF', '#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3',
        '#F38181', '#AA96DA', '#FCBAD3', '#A8D8EA', '#FF9A76',
        '#C4E538', '#E056B4', '#74B9FF', '#FFA502', '#7BED9F'
    ]
    
    # Compute node sizes from centrality
    max_pagerank = 1.0
    if centrality_df is not None and not centrality_df.empty:
        max_pagerank = centrality_df['pagerank'].max() or 1.0
    
    for node in G.nodes():
        # Size based on PageRank
        if centrality_df is not None and not centrality_df.empty:
            pr = centrality_df[centrality_df['account'] == node]['pagerank'].values
            size = 10 + (pr[0] / max_pagerank * 40) if len(pr) > 0 else 10
        else:
            size = 10 + G.degree(node) * 2
        
        # Color based on community
        comm = partition.get(node, 0) if partition else 0
        color = colors[comm % len(colors)]
        
        # Tooltip
        post_count = G.nodes[node].get('post_count', 'N/A')
        subreddits = G.nodes[node].get('subreddits', set())
        sub_list = ', '.join(list(subreddits)[:3]) if subreddits else 'N/A'
        title = f"<b>{node}</b><br>Posts: {post_count}<br>Subreddits: {sub_list}<br>Connections: {G.degree(node)}"
        
        net.add_node(node, 
                     label=node if size > 15 else "",
                     size=size,
                     color=color,
                     title=title,
                     borderWidth=2,
                     borderWidthSelected=4)
    
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        net.add_edge(u, v, 
                     value=weight,
                     color={'color': '#555555', 'opacity': 0.5})
    
    # Physics configuration for layout
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08,
          "damping": 0.4
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": true,
          "iterations": 200
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "zoomView": true,
        "dragView": true
      }
    }
    """)
    
    # Generate HTML
    try:
        # Write to temp file and read back
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "network.html")
        net.save_graph(tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            html = f.read()
        os.remove(tmp_path)
        os.rmdir(tmp_dir)
        return html
    except Exception as e:
        logger.error(f"Error generating network visualization: {e}")
        return f"<p style='color: #ff6b6b;'>Error generating network: {e}</p>"
