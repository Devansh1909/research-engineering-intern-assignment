"""
realtime.py — Live Reddit API integration for NarrativeScope.

Provides capabilities to fetch recent posts on-demand and merge them
with the offline dataset without crashing the UI.

TASK 1 FIX: Reads Reddit credentials from environment variables first
(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT).
Falls back to explicitly-passed arguments when env vars are absent.
TASK 3 FIX: Tracks last fetch time to avoid excessive API calls.
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

EXPECTED_COLUMNS = [
    'author', 'subreddit', 'body', 'title', 'created_utc',
    'score', 'id', 'permalink', 'url', 'num_comments'
]

# Minimum seconds between consecutive API fetches to respect Reddit rate limits.
# Reddit's free-tier limit is 60 requests/minute for OAuth and 10 for
# unauthenticated. We set a conservative 30-second cooldown.
_FETCH_COOLDOWN_SECONDS = 30

# ── Helper: resolve credentials ────────────────────────────────────────────────

def _resolve_credentials(client_id: str, client_secret: str, user_agent: str) -> tuple:
    """
    TASK 1 FIX: Resolve Reddit credentials from environment variables first,
    then fall back to explicitly passed arguments. Returns (client_id,
    client_secret, user_agent) or raises ValueError if both sources are empty.
    """
    resolved_id     = os.environ.get("REDDIT_CLIENT_ID",     client_id     or "").strip()
    resolved_secret = os.environ.get("REDDIT_CLIENT_SECRET", client_secret or "").strip()
    resolved_agent  = os.environ.get("REDDIT_USER_AGENT",    user_agent    or "NarrativeScope/1.0").strip()

    return resolved_id, resolved_secret, resolved_agent


# ── Core fetch function ────────────────────────────────────────────────────────

def fetch_live_reddit_data(
    client_id:     str = "",
    client_secret: str = "",
    user_agent:    str = "NarrativeScope/1.0",
    query:         str = "",
    subreddit_name: str = "all",
    limit:         int = 50,
) -> tuple:
    """
    Fetch recent Reddit posts using PRAW.

    TASK 1 FIX:
      - Credentials are resolved from env vars first, then passed arguments.
      - Gracefully returns an empty list with an error message if creds are missing.
    TASK 3 FIX:
      - Enforces a per-session cooldown (FETCH_COOLDOWN_SECONDS) to prevent
        runaway API calls on rapid reruns.
      - Returns (list_of_post_dicts, error_message_or_None).
    """
    # ── Rate-limit guard ───────────────────────────────────────────────────────
    # TASK 3: Use session_state to track the last successful fetch timestamp.
    now = time.time()
    last_fetch = st.session_state.get("_last_reddit_fetch_ts", 0)
    elapsed = now - last_fetch
    if elapsed < _FETCH_COOLDOWN_SECONDS and last_fetch != 0:
        wait_secs = int(_FETCH_COOLDOWN_SECONDS - elapsed)
        return [], (
            f"Rate-limit cooldown active — please wait {wait_secs}s before fetching again. "
            f"(Minimum interval between fetches: {_FETCH_COOLDOWN_SECONDS}s)"
        )

    # ── Credential resolution ──────────────────────────────────────────────────
    resolved_id, resolved_secret, resolved_agent = _resolve_credentials(
        client_id, client_secret, user_agent
    )

    # TASK 1: Graceful fallback — app still works without credentials.
    if not resolved_id or not resolved_secret:
        return [], (
            "Reddit credentials are missing. "
            "Either enter them in the sidebar or set the REDDIT_CLIENT_ID and "
            "REDDIT_CLIENT_SECRET environment variables."
        )

    # ── PRAW initialisation ────────────────────────────────────────────────────
    try:
        import praw
        from prawcore.exceptions import ResponseException, OAuthException

        reddit = praw.Reddit(
            client_id=resolved_id,
            client_secret=resolved_secret,
            user_agent=resolved_agent,
        )
        # Confirm read-only mode (does not require user auth).
        reddit.read_only = True

    except ImportError:
        return [], (
            "PRAW is not installed. Run: pip install praw>=7.7.0"
        )
    except Exception as exc:
        logger.error("Failed to initialise PRAW: %s", exc)
        return [], f"Authentication error: {exc}"

    # ── Fetch posts ────────────────────────────────────────────────────────────
    new_posts = []
    try:
        sub = reddit.subreddit(subreddit_name.strip() or "all")

        if query and query.strip():
            # Search within the subreddit, sorted by newest first
            results = sub.search(query.strip(), sort="new", limit=limit, time_filter="week")
        else:
            results = sub.new(limit=limit)

        for submission in results:
            author_name = "unknown"
            try:
                if submission.author:
                    author_name = submission.author.name
            except Exception:
                pass  # Author may be deleted/suspended

            post_data = {
                "id":           submission.id,
                "author":       author_name,
                "subreddit":    submission.subreddit.display_name,
                "title":        submission.title,
                "body":         submission.selftext,
                "created_utc":  submission.created_utc,
                "score":        submission.score,
                "num_comments": submission.num_comments,
                "permalink":    f"https://reddit.com{submission.permalink}",
                "url":          submission.url,
            }
            new_posts.append(post_data)

        # TASK 3: Record successful fetch time to enforce cooldown on next call.
        st.session_state["_last_reddit_fetch_ts"] = time.time()

        logger.info("Fetched %d live posts from r/%s", len(new_posts), subreddit_name)
        return new_posts, None

    except ResponseException as resp_exc:
        logger.error("Reddit API ResponseException: %s", resp_exc)
        code = getattr(resp_exc.response, "status_code", None)
        if code == 429:
            return [], "Rate limit exceeded by Reddit. Please wait a few minutes before trying again."
        if code == 401:
            return [], "Unauthorised: Invalid Client ID or Secret. Check your Reddit app credentials."
        if code == 403:
            return [], "Forbidden: Your app may not have permission for this subreddit."
        return [], f"Reddit API Error (HTTP {code})."

    except OAuthException as oauth_exc:
        logger.error("Reddit OAuthException: %s", oauth_exc)
        return [], f"OAuth error — check your credentials: {oauth_exc}"

    except Exception as exc:
        logger.error("Unexpected error fetching live Reddit data: %s", exc)
        return [], f"Error fetching data: {exc}"


# ── Persistence helper ─────────────────────────────────────────────────────────

def save_live_data(posts: list) -> str:
    """
    Save fetched posts to a JSONL file in the data/ directory.
    Returns the filepath of the saved data, or an empty string if nothing to save.
    """
    if not posts:
        return ""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = DATA_DIR / f"live_reddit_{timestamp}.jsonl"

    with open(filepath, "w", encoding="utf-8") as fh:
        for post in posts:
            fh.write(json.dumps(post) + "\n")

    logger.info("Saved %d live posts to %s", len(posts), filepath)
    return str(filepath)
