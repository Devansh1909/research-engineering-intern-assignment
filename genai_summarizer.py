"""
genai_summarizer.py — Dynamic GenAI summaries using Google Gemini with rule-based fallback.

Generates plain-language summaries beneath each visualization so non-technical
audiences can understand trends without interpreting charts themselves.
Summaries are generated dynamically from actual data, never hardcoded.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.info("google-generativeai not installed. Using rule-based summaries.")


def configure_gemini(api_key: str = None):
    """Configure the Gemini API with the provided key."""
    if not GEMINI_AVAILABLE:
        return False
    
    key = api_key or os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    if not key:
        return False
    
    try:
        genai.configure(api_key=key)
        return True
    except Exception as e:
        logger.warning(f"Failed to configure Gemini: {e}")
        return False


def _call_gemini(prompt: str, max_tokens: int = 300) -> str:
    """Call Gemini API with error handling."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=max_tokens,
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")
        return None


def summarize_timeseries(daily_data: pd.DataFrame, context: str = "posts") -> str:
    """
    Generate a plain-language summary of time-series data.
    
    The summary describes: overall trend, peak/valley dates, growth rate,
    and notable patterns — all derived from the actual data.
    """
    if daily_data.empty or 'post_count' not in daily_data.columns:
        return "No time-series data available to summarize."
    
    # Extract key statistics from the actual data
    total_posts = int(daily_data['post_count'].sum())
    peak_date = daily_data.loc[daily_data['post_count'].idxmax(), 'date']
    peak_count = int(daily_data['post_count'].max())
    valley_date = daily_data.loc[daily_data['post_count'].idxmin(), 'date']
    valley_count = int(daily_data['post_count'].min())
    avg_daily = round(daily_data['post_count'].mean(), 1)
    date_range_start = daily_data['date'].min()
    date_range_end = daily_data['date'].max()
    
    # Compute trend direction
    if len(daily_data) >= 7:
        first_week_avg = daily_data.head(7)['post_count'].mean()
        last_week_avg = daily_data.tail(7)['post_count'].mean()
        if last_week_avg > first_week_avg * 1.2:
            trend = "increasing"
        elif last_week_avg < first_week_avg * 0.8:
            trend = "declining"
        else:
            trend = "relatively stable"
    else:
        trend = "too short to determine a trend"
    
    # Detect spikes (days > 2x average)
    spikes = daily_data[daily_data['post_count'] > avg_daily * 2]
    spike_info = ""
    if not spikes.empty:
        spike_dates = [str(d) for d in spikes['date'].tolist()[:3]]
        spike_info = f"Notable activity spikes occurred on: {', '.join(spike_dates)}."
    
    # Try Gemini first
    if GEMINI_AVAILABLE:
        prompt = f"""Summarize this time-series data about {context} in 2-3 sentences for a non-technical audience.
        
Data:
- Total {context}: {total_posts}
- Date range: {date_range_start} to {date_range_end}
- Peak: {peak_count} {context} on {peak_date}
- Lowest: {valley_count} {context} on {valley_date}
- Average daily: {avg_daily}
- Overall trend: {trend}
- Spikes: {spike_info if spike_info else 'None detected'}

Write a concise, informative summary. Focus on what the pattern means, not just the numbers. Do not use markdown formatting."""
        
        gemini_summary = _call_gemini(prompt)
        if gemini_summary:
            return gemini_summary
    
    # Rule-based fallback
    summary = f"Over the period from {date_range_start} to {date_range_end}, there were {total_posts:,} {context} with an average of {avg_daily} per day. "
    summary += f"Activity was {trend}, peaking at {peak_count} {context} on {peak_date}. "
    if spike_info:
        summary += spike_info
    
    return summary


def summarize_topics(cluster_stats: pd.DataFrame, cluster_labels: dict) -> str:
    """
    Generate a summary of discovered topic clusters.
    Describes the major themes found and their relative sizes.
    """
    if cluster_stats.empty:
        return "No topic clusters were identified in the current data."
    
    # Filter out noise cluster
    real_clusters = cluster_stats[cluster_stats['cluster'] != -1]
    
    if real_clusters.empty:
        return "All posts were classified as noise — try adjusting the minimum cluster size parameter."
    
    n_clusters = len(real_clusters)
    total_clustered = int(real_clusters['post_count'].sum())
    total_posts = int(cluster_stats['post_count'].sum())
    noise_pct = round((1 - total_clustered / total_posts) * 100, 1) if total_posts > 0 else 0
    
    # Top 3 clusters
    top_clusters = real_clusters.head(3)
    top_info = []
    for _, row in top_clusters.iterrows():
        label = cluster_labels.get(row['cluster'], f"Topic {row['cluster']}")
        top_info.append(f"'{label}' ({int(row['post_count'])} posts)")
    
    if GEMINI_AVAILABLE:
        prompt = f"""Summarize these topic clusters found in social media data in 2-3 sentences for a non-technical audience.

Data:
- Number of topic clusters: {n_clusters}
- Total clustered posts: {total_clustered} out of {total_posts} ({noise_pct}% unclustered)
- Top topics: {'; '.join(top_info)}

Explain what these clusters reveal about the main themes in the discourse. Do not use markdown formatting."""
        
        gemini_summary = _call_gemini(prompt)
        if gemini_summary:
            return gemini_summary
    
    # Rule-based fallback
    summary = f"The analysis identified {n_clusters} distinct topic clusters covering {total_clustered:,} out of {total_posts:,} posts ({noise_pct}% remained unclustered). "
    summary += f"The dominant topics are: {', '.join(top_info)}. "
    if noise_pct > 40:
        summary += "The high proportion of unclustered posts suggests significant topic diversity — try decreasing the minimum cluster size to capture more granular topics."
    
    return summary


def summarize_network(centrality_df: pd.DataFrame, n_communities: int, 
                       graph_stats: dict = None) -> str:
    """
    Generate a summary of the network analysis.
    Describes key influencers, community structure, and network properties.
    """
    if centrality_df.empty:
        return "No network data available to summarize."
    
    top_accounts = centrality_df.head(5)['account'].tolist()
    n_nodes = len(centrality_df)
    avg_degree = round(centrality_df['degree'].mean(), 1) if 'degree' in centrality_df else 'N/A'
    
    top_by_pagerank = centrality_df.head(3)[['account', 'pagerank']].to_dict('records')
    top_by_betweenness = centrality_df.sort_values('betweenness', ascending=False).head(3)[['account', 'betweenness']].to_dict('records')
    
    if GEMINI_AVAILABLE:
        prompt = f"""Summarize this social media network analysis in 2-3 sentences for a non-technical audience.

Data:
- Network has {n_nodes} accounts and {n_communities} communities
- Average connections per account: {avg_degree}
- Most influential accounts (by PageRank): {', '.join([f"{a['account']} ({a['pagerank']:.4f})" for a in top_by_pagerank])}
- Key bridge accounts (by betweenness centrality): {', '.join([f"{a['account']} ({a['betweenness']:.4f})" for a in top_by_betweenness])}

Explain what this means in terms of who drives the conversation and how information flows. Do not use markdown formatting."""
        
        gemini_summary = _call_gemini(prompt)
        if gemini_summary:
            return gemini_summary
    
    # Rule-based fallback
    summary = f"The network consists of {n_nodes} accounts organized into {n_communities} distinct communities. "
    summary += f"The most influential accounts (by PageRank) are {', '.join(top_accounts[:3])}, "
    summary += f"who appear to drive significant portions of the discourse. "
    summary += f"On average, each account is connected to {avg_degree} others."
    
    return summary


def summarize_search_results(results_df: pd.DataFrame, query: str) -> str:
    """Generate a summary of semantic search results."""
    if results_df.empty:
        return f"No results found for '{query}'."
    
    n_results = len(results_df)
    avg_sim = round(results_df['similarity_score'].mean(), 3) if 'similarity_score' in results_df else 'N/A'
    top_subreddits = results_df['subreddit'].value_counts().head(3).to_dict() if 'subreddit' in results_df else {}
    date_range = f"{results_df['datetime'].min().strftime('%Y-%m-%d')} to {results_df['datetime'].max().strftime('%Y-%m-%d')}" if 'datetime' in results_df else 'N/A'
    
    if GEMINI_AVAILABLE:
        prompt = f"""Summarize these semantic search results in 1-2 sentences.

Query: "{query}"
Results: {n_results} posts found
Average relevance score: {avg_sim}
Top subreddits: {top_subreddits}
Date range: {date_range}

Describe what the results reveal about this topic. Do not use markdown formatting."""
        
        gemini_summary = _call_gemini(prompt)
        if gemini_summary:
            return gemini_summary
    
    sub_info = ', '.join([f"r/{k} ({v})" for k, v in list(top_subreddits.items())[:3]]) if top_subreddits else "various subreddits"
    return f"Found {n_results} semantically relevant posts for '{query}' (avg similarity: {avg_sim}), primarily from {sub_info}, spanning {date_range}."
