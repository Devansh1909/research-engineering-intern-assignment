"""
data_loader.py — Data ingestion and DuckDB setup for NarrativeScope.

Handles loading Reddit JSONL data into DuckDB for fast analytical queries.
Includes data cleaning, timestamp normalization, and schema validation.
"""

import os
import json
import duckdb
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected columns in the Reddit dataset (flexible — we handle missing ones)
EXPECTED_COLUMNS = {
    'author', 'subreddit', 'body', 'title', 'created_utc',
    'score', 'id', 'permalink', 'url', 'num_comments'
}

DATA_DIR = Path(__file__).parent / "data"


def find_data_files() -> list:
    """Find all JSONL/JSON/CSV data files in the data directory."""
    data_files = []
    if DATA_DIR.exists():
        for ext in ['*.jsonl', '*.json', '*.csv']:
            data_files.extend(DATA_DIR.glob(ext))
    # Also check project root
    root = Path(__file__).parent
    for ext in ['*.jsonl', '*.json']:
        data_files.extend(root.glob(ext))
    return data_files


def load_jsonl(filepath: str) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame with error handling for malformed lines."""
    records = []
    errors = 0
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                errors += 1
                if errors <= 5:
                    logger.warning(f"Skipping malformed line {i+1} in {filepath}")
    
    if errors > 0:
        logger.info(f"Total malformed lines skipped: {errors}")
    
    if not records:
        return pd.DataFrame()
    
    return pd.DataFrame(records)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the DataFrame to a consistent schema.
    Handles variations in column names across different Reddit data exports.
    """
    # Column name mapping for common variations
    column_aliases = {
        'selftext': 'body',
        'text': 'body',
        'content': 'body',
        'post_body': 'body',
        'user': 'author',
        'username': 'author',
        'author_name': 'author',
        'timestamp': 'created_utc',
        'created': 'created_utc',
        'date': 'created_utc',
        'created_at': 'created_utc',
        'subreddit_name': 'subreddit',
        'sub': 'subreddit',
        'community': 'subreddit',
        'upvotes': 'score',
        'ups': 'score',
        'karma': 'score',
        'post_id': 'id',
        'comment_id': 'id',
        'link': 'permalink',
        'post_url': 'url',
        'comment_count': 'num_comments',
        'replies': 'num_comments',
    }
    
    # Apply aliases
    for old_name, new_name in column_aliases.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Combine title and body into a 'text' column for analysis
    if 'title' in df.columns and 'body' in df.columns:
        df['text'] = df.apply(
            lambda row: f"{row.get('title', '')} {row.get('body', '')}".strip(),
            axis=1
        )
    elif 'title' in df.columns:
        df['text'] = df['title'].fillna('')
    elif 'body' in df.columns:
        df['text'] = df['body'].fillna('')
    else:
        df['text'] = ''
    
    # Clean text
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 0]  # Remove empty posts
    
    # Normalize timestamps
    if 'created_utc' in df.columns:
        df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
        df['datetime'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    else:
        # No timestamp found — use index as proxy
        logger.warning("No timestamp column found. Using sequential index.")
        df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='h')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['datetime'])
    df['date'] = df['datetime'].dt.date
    
    # Ensure numeric columns
    for col in ['score', 'num_comments']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            df[col] = 0
    
    # Ensure string columns
    for col in ['author', 'subreddit', 'id', 'permalink', 'url']:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('unknown')
        else:
            df[col] = 'unknown'
    
    # Extract hashtags (Reddit uses them less, but some posts include them)
    df['hashtags'] = df['text'].str.findall(r'#(\w+)')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} posts spanning {df['datetime'].min()} to {df['datetime'].max()}")
    return df


@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    """Main data loading function with caching."""
    data_files = find_data_files()
    
    if not data_files:
        st.error("No data files found! Please place your JSONL/CSV data files in the `data/` directory.")
        st.info("Expected format: JSONL with fields like author, subreddit, body/title, created_utc, score")
        return pd.DataFrame()
    
    all_dfs = []
    for filepath in data_files:
        logger.info(f"Loading {filepath}...")
        try:
            if filepath.suffix == '.csv':
                df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
            elif filepath.suffix == '.jsonl':
                df = load_jsonl(str(filepath))
            elif filepath.suffix == '.json':
                df = pd.read_json(str(filepath), lines=False)
            else:
                continue
            
            if not df.empty:
                df = normalize_dataframe(df)
                all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            continue
    
    if not all_dfs:
        st.error("Could not load any data files. Check the format of your data files.")
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['text', 'author', 'datetime'], keep='first')
    combined = combined.sort_values('datetime').reset_index(drop=True)
    
    logger.info(f"Total dataset: {len(combined)} posts, {combined['subreddit'].nunique()} subreddits, {combined['author'].nunique()} authors")
    return combined


def init_duckdb(df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB with the loaded DataFrame for fast analytical queries."""
    con = duckdb.connect(':memory:')
    con.register('posts', df)
    
    # Create useful views
    con.execute("""
        CREATE OR REPLACE VIEW daily_counts AS
        SELECT 
            date,
            COUNT(*) as post_count,
            COUNT(DISTINCT author) as unique_authors,
            COUNT(DISTINCT subreddit) as unique_subreddits,
            AVG(score) as avg_score
        FROM posts
        GROUP BY date
        ORDER BY date
    """)
    
    con.execute("""
        CREATE OR REPLACE VIEW subreddit_stats AS
        SELECT 
            subreddit,
            COUNT(*) as post_count,
            COUNT(DISTINCT author) as unique_authors,
            AVG(score) as avg_score,
            MIN(datetime) as first_post,
            MAX(datetime) as last_post
        FROM posts
        GROUP BY subreddit
        ORDER BY post_count DESC
    """)
    
    con.execute("""
        CREATE OR REPLACE VIEW author_stats AS
        SELECT 
            author,
            COUNT(*) as post_count,
            COUNT(DISTINCT subreddit) as subreddit_count,
            AVG(score) as avg_score,
            SUM(score) as total_score
        FROM posts
        WHERE author != '[deleted]' AND author != 'unknown'
        GROUP BY author
        ORDER BY post_count DESC
    """)
    
    return con


def query_filtered_data(con: duckdb.DuckDBPyConnection, 
                         search_term: str = None,
                         subreddit: str = None,
                         date_range: tuple = None,
                         min_score: int = None) -> pd.DataFrame:
    """Query filtered data using DuckDB for fast analytical operations."""
    conditions = []
    
    if search_term and search_term.strip():
        # Escape single quotes in search term
        safe_term = search_term.replace("'", "''")
        conditions.append(f"LOWER(text) LIKE '%{safe_term.lower()}%'")
    
    if subreddit and subreddit != 'All':
        safe_sub = subreddit.replace("'", "''")
        conditions.append(f"subreddit = '{safe_sub}'")
    
    if date_range and len(date_range) == 2:
        conditions.append(f"date >= '{date_range[0]}' AND date <= '{date_range[1]}'")
    
    if min_score is not None:
        conditions.append(f"score >= {min_score}")
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    query = f"""
        SELECT * FROM posts
        WHERE {where_clause}
        ORDER BY datetime DESC
    """
    
    try:
        return con.execute(query).fetchdf()
    except Exception as e:
        logger.error(f"Query error: {e}")
        return pd.DataFrame()


def get_summary_stats(con: duckdb.DuckDBPyConnection) -> dict:
    """Get high-level summary statistics for the dataset."""
    try:
        stats = con.execute("""
            SELECT 
                COUNT(*) as total_posts,
                COUNT(DISTINCT author) as unique_authors,
                COUNT(DISTINCT subreddit) as unique_subreddits,
                MIN(datetime) as date_start,
                MAX(datetime) as date_end,
                AVG(score) as avg_score,
                MEDIAN(score) as median_score
            FROM posts
        """).fetchdf().iloc[0].to_dict()
        return stats
    except Exception as e:
        logger.error(f"Error getting summary stats: {e}")
        return {}


def append_live_data(con: duckdb.DuckDBPyConnection, original_df: pd.DataFrame, new_posts: list) -> pd.DataFrame:
    """
    Append live fetched posts to the existing dataframe and DuckDB connection.
    Returns the newly combined dataframe.
    """
    if not new_posts:
        return original_df

    # Convert to DataFrame
    new_df = pd.DataFrame(new_posts)
    
    # Normalize schema using the existing function
    new_df = normalize_dataframe(new_df)
    
    # Combine with original
    combined_df = pd.concat([original_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['text', 'author', 'datetime'], keep='last')
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    
    # Update DuckDB by replacing the view entirely is safest and fast for our scale
    con.register('posts', combined_df)
    
    logger.info(f"Appended {len(new_df)} live posts. Total is now {len(combined_df)}")
    return combined_df
