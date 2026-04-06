# devansh-prompts.md — AI-Assisted Development Log

Below is a numbered log of every significant AI prompt I used during development. Each entry includes the component, the prompt, what went wrong (if anything), and how I fixed it. I used Claude and GitHub Copilot throughout, but every architectural decision was mine — I wrote the DESIGN-DOC.md before touching any AI tool.

---

## 1. Data Loading — Initial Schema Detection
**Component:** `data_loader.py`  
**Prompt:** "Write a Python function that loads JSONL files and normalizes Reddit data columns. The data might have 'selftext' or 'body' or 'text' for content, and 'created_utc' as Unix timestamp or ISO string."  
**What went wrong:** The initial version assumed all fields existed and crashed on missing columns. Reddit data exports are inconsistent — some have `selftext`, some have `body`, some have `content`.  
**How I fixed it:** Added a `column_aliases` dictionary to map variations, and wrapped every field access in `.get()` with defaults. Tested with intentionally malformed files.

---

## 2. DuckDB Integration — First Attempt
**Component:** `data_loader.py`  
**Prompt:** "How do I register a pandas DataFrame in DuckDB and create analytical views for daily post counts, author stats, and subreddit stats?"  
**What went wrong:** Nothing critical, but the initial views didn't include MEDIAN for score (only AVG), which gave misleading stats when a few viral posts skewed the average.  
**How I fixed it:** Added MEDIAN alongside AVG in the summary stats. Also added proper error handling for the case where DuckDB can't parse a column type.

---

## 3. Sentence-Transformers Embedding — Batch Size Issues
**Component:** `search_engine.py`  
**Prompt:** "Encode all texts in a DataFrame using sentence-transformers all-MiniLM-L6-v2 with batch processing and progress bar."  
**What went wrong:** First attempt used batch_size=256 which caused an OOM crash on my machine (8GB RAM) with ~50k posts. Also forgot to truncate long posts — MiniLM has a 256 token limit.  
**How I fixed it:** Reduced batch_size to 64, added text truncation to 512 chars before encoding. Also added `normalize_embeddings=True` so I could use IndexFlatIP (inner product) instead of IndexFlatL2 — inner product on normalized vectors equals cosine similarity but is faster.

---

## 4. FAISS Index — Cosine Similarity Confusion
**Component:** `search_engine.py`  
**Prompt:** "Build a FAISS index for cosine similarity search on sentence embeddings."  
**What went wrong:** AI suggested `IndexFlatL2` for L2 distance but I needed cosine similarity. It also didn't mention that FAISS doesn't natively support cosine similarity — you need to normalize vectors first and use inner product.  
**How I fixed it:** Used `normalize_embeddings=True` in sentence-transformers (so vectors are L2-normalized) and then used `IndexFlatIP` (inner product). Inner product on normalized vectors = cosine similarity. This is a well-known trick but the AI didn't suggest it initially.

---

## 5. Edge Case Handling — Empty Query Crash
**Component:** `search_engine.py`  
**Prompt:** "Add validation for empty queries, very short queries, and non-English input to the semantic search function."  
**What went wrong:** The first version just checked `if not query` but didn't handle whitespace-only strings. Also, `langdetect` throws `LangDetectException` on very short strings (< 3 chars), which I hadn't anticipated.  
**How I fixed it:** Added `.strip()` before all checks, wrapped langdetect in try/except, and added a minimum query length constant (3 chars). Tested with "", " ", "ab", "検索テスト", and emoji-only queries.

---

## 6. HDBSCAN — First Clustering Attempt
**Component:** `clustering.py`  
**Prompt:** "Cluster text embeddings using HDBSCAN with tunable min_cluster_size. Use UMAP for dimensionality reduction first."  
**What went wrong:** Applied HDBSCAN directly to 384-dim embeddings — extremely slow and poor results (curse of dimensionality). The AI didn't mention that HDBSCAN works poorly in high dimensions and needs UMAP pre-processing.  
**How I fixed it:** Added a two-stage pipeline: UMAP to 15 dimensions (for clustering), then HDBSCAN. Also added separate UMAP to 2D for visualization. I chose 15 dims for clustering because the BERTopic paper uses a similar approach and I found it works well empirically.

---

## 7. HDBSCAN — Extreme Parameter Handling
**Component:** `clustering.py`  
**Prompt:** "What happens to HDBSCAN when min_cluster_size is larger than the dataset? How do I handle it gracefully?"  
**What went wrong:** When `min_cluster_size >= n_samples`, HDBSCAN throws a cryptic error instead of returning all noise. The slider in Streamlit also didn't cap properly, letting users set values that crash the app.  
**How I fixed it:** Added explicit check: if `min_cluster_size >= len(data)`, return all -1 labels with a warning. Also capped the Streamlit slider max at `len(df) // 5`. Added UI messages for edge cases: "All noise", "Only 1 cluster", "High noise percentage".

---

## 8. Network Graph — Layout Not Working
**Component:** `network_analysis.py`  
**Prompt:** "Build a co-occurrence network from Reddit data where edges connect authors who post in the same subreddit. Use PyVis for visualization."  
**What went wrong:** Initial version connected ALL pair combinations in large subreddits — O(n²) which is 250,000 edges for a 500-author subreddit. This made the graph unreadable and PyVis took 30+ seconds to render.  
**How I fixed it:** Capped pair generation to top 50 authors per subreddit. Added `min_edge_weight=2` to prune weak connections. Limited total graph to 200 nodes by keeping highest-degree nodes. Also switched PyVis physics to forceAtlas2Based which handles dense graphs much better than the default.

---

## 9. PageRank — Disconnected Components Issue
**Component:** `network_analysis.py`  
**Prompt:** "Compute PageRank for a NetworkX graph that might have disconnected components."  
**What went wrong:** The AI didn't warn me that PageRank on disconnected graphs concentrates all rank in the largest component. Nodes in small components get near-zero scores even if they're important within their subgraph.  
**How I fixed it:** Initially considered per-component PageRank but decided against it — the composite influence score (0.5*PR + 0.3*betweenness + 0.2*degree) naturally handles this since betweenness and degree are component-independent. I also considered eigenvector centrality but rejected it because it converges poorly on disconnected graphs.

---

## 10. Node Removal — Graph Copy Issue
**Component:** `network_analysis.py`  
**Prompt:** "Simulate removing a node from a NetworkX graph and show before/after stats including component count."  
**What went wrong:** First attempt modified the graph in-place, which broke all subsequent analysis because the original graph was mutated. Classic Python mutable object issue.  
**How I fixed it:** Added `G_after = G.copy()` before removing the node. Also added cleanup to remove isolates that result from the removal, since they're not meaningful on their own.

---

## 11. Gemini API — Rate Limiting
**Component:** `genai_summarizer.py`  
**Prompt:** "Call Google Gemini 1.5 Flash API to summarize time-series data. Add a fallback for when the API is unavailable."  
**What went wrong:** The free tier has strict rate limits (15 RPM). When loading the full dashboard, it made 4-5 API calls simultaneously and hit the limit. Also, if the API key is invalid, the error message was unhelpful.  
**How I fixed it:** Added try/except around every Gemini call with a rule-based fallback that generates summaries from computed statistics. The fallback isn't as elegant but ensures the app always works. Also injected actual data (peaks, trends, counts) into prompts instead of just asking "summarize this" — which improved output quality significantly.

---

## 12. Streamlit Custom CSS — Theme Conflict
**Component:** `app.py`  
**Prompt:** "Create a dark investigative-journalism theme for Streamlit with custom CSS. Purple accent color, gradient headers."  
**What went wrong:** The AI's CSS conflicted with Streamlit's built-in dark theme, creating double-dark backgrounds where elements were invisible. Some Streamlit widgets (st.dataframe, st.metric) don't respect custom CSS fully.  
**How I fixed it:** Used `.streamlit/config.toml` for the base dark theme, then only overrode specific elements (hero header, section dividers, summary boxes) with custom CSS. Tested every widget individually to make sure text was readable.

---

## 13. GenAI Summary — Hardcoded vs Dynamic
**Component:** `genai_summarizer.py`  
**Prompt:** "Generate summaries for each visualization section that change based on the actual data."  
**What went wrong:** First attempt passed generic prompts like "summarize this chart" — the AI returned generic, meaningless summaries. The rubric explicitly says summaries must be "generated dynamically based on the actual data."  
**How I fixed it:** Restructured every summarization function to first compute specific statistics (peak date, growth rate, top clusters, etc.) and inject those into the prompt. For example, the time-series prompt now includes exact peak dates, counts, and trend direction. The rule-based fallback also uses these exact statistics.

---

## 14. UMAP — Reproducibility Issue
**Component:** `clustering.py`  
**Prompt:** "Make UMAP produce deterministic results for reproducible clustering."  
**What went wrong:** UMAP is stochastic by default. Different runs produced different cluster layouts, which confused users who expected consistency. The AI didn't set `random_state`.  
**How I fixed it:** Added `random_state=42` to both UMAP calls. Also added `low_memory=True` for better performance on larger datasets.

---

## 15. Datamapplot Integration — Import Failures
**Component:** `clustering.py`  
**Prompt:** "Use datamapplot to create an interactive topic visualization embedded in Streamlit."  
**What went wrong:** Datamapplot produces matplotlib figures, not HTML, so it can't be directly embedded as an interactive widget in Streamlit. The AI suggested `st.pyplot()` but that makes it static.  
**How I fixed it:** Converted the matplotlib figure to a base64-encoded PNG and embedded it via `st.components.v1.html()`. Not ideal for interactivity, but visually much better than a basic scatter plot. Added Plotly as the primary fallback for full interactivity, with Datamapplot as an optional enhanced view.

---

## 16. Thread Puller — Connecting Clusters to Timeline
**Component:** `app.py`  
**Prompt:** "Show a timeline of posts within a selected topic cluster, with community crossover breakdown."  
**What went wrong:** The cluster labels (from HDBSCAN) are numeric IDs, not human-readable. The first version just showed "Cluster 0", "Cluster 1" which is meaningless.  
**How I fixed it:** Added TF-IDF based cluster labeling — for each cluster, extract the top 5 most distinctive terms (highest avg TF-IDF score within the cluster). This gives labels like "immigration policy, border" instead of "Cluster 3".

---

## 17. Query Suggestions — Follow-up Queries
**Component:** `search_engine.py`  
**Prompt:** "After search results, suggest 2-3 follow-up queries the user might want to explore."  
**What went wrong:** The AI generated suggestions using another LLM call, which was too slow (adding 2-3 seconds to every search). Also, the suggestions were often too generic.  
**How I fixed it:** Replaced the LLM-based approach with a keyword-expansion heuristic: maintain a dictionary of topic → related investigation angles, plus two template suggestions about influential voices and narrative evolution. Much faster and more relevant.

---

## 18. Streamlit Caching — Stale Embeddings
**Component:** `app.py`  
**Prompt:** "Cache expensive computations (embeddings, UMAP, FAISS index) across Streamlit sessions."  
**What went wrong:** Used `@st.cache_data` for the sentence-transformer model, but the model object isn't serializable. Should have used `@st.cache_resource` for the model and `@st.cache_data` for the embeddings.  
**How I fixed it:** `@st.cache_resource` for model, `@st.cache_data` for embeddings/UMAP. Also had to prefix the model parameter with `_` (e.g., `_model`) to tell Streamlit not to hash it.

---

## 19. Wikipedia API — User Agent Required
**Component:** `app.py`  
**Prompt:** "Fetch Wikipedia summaries for a topic to correlate with online discussions."  
**What went wrong:** The `wikipedia` Python package has been unreliable. Switched to `wikipedia-api` which requires a custom user agent string (Wikipedia blocks requests without one).  
**How I fixed it:** Used `wikipediaapi.Wikipedia(user_agent='NarrativeScope/1.0 (...)')` with a proper user agent. Also wrapped in try/except for when the page doesn't exist.

---

## 20. Final Robustness Pass — Stress Testing
**Component:** All modules  
**Prompt:** "Review all edge cases: what happens with a dataset of 10 posts? 100k posts? All posts from same author? All posts from same subreddit?"  
**What went wrong:** Several issues:
1. Network graph fails silently with < 3 unique authors
2. UMAP crashes if `n_neighbors > n_samples`
3. Time-series plot shows nothing meaningful with only 1 day of data  
**How I fixed it:** Added `n_neighbors = min(n_neighbors, n_samples // 3)` for UMAP. Network graph shows "Not enough connected accounts" message for tiny datasets. Time-series enforces minimum 2 data points. These were the kind of issues that only surface during actual testing, not during initial development.

---

## 21. Live Data Expansion — PRAW Integration
**Component:** `realtime.py`  
**Prompt:** "Create a script to fetch the latest 50 posts from a given subreddit using PRAW."  
**What went wrong:** Initially placed hardcoded credentials inside the script. Realised that for a public assignment, reviewers won't be able to test it without providing their own credentials, and saving my API keys to GitHub is a major security risk.  
**How I fixed it:** Restructured it to accept `client_id` and `client_secret` dynamically from Streamlit's UI via `st.text_input(type="password")`. Added graceful error handling for 401 Unauthorized errors to prevent the dashboard from crashing if reviewers enter invalid keys.

---

## 22. Merging Live Data — FAISS synchronization
**Component:** `data_loader.py` & `app.py`  
**Prompt:** "How do I add newly fetched live records to an active DuckDB connection and a FAISS index without restarting the entire Streamlit app?"  
**What went wrong:** First attempt used `index.add(new_embeddings)` but indexing drifted because Streamlit's `st.rerun()` would re-trigger offline data loading, causing the FAISS index to misalign with the DuckDB view resulting in "index out of bounds" errors during semantic search.  
**How I fixed it:** Kept offline data loading cached, but appended the live dataframe directly to the master dataframe immediately after the cache loads, then redefined the DuckDB view entirely from the concatenated dataset on every rerun. Safe, fast, and physically perfectly aligned.

---

## 23. Wikipedia Contextual Events 
**Component:** `events.py` & Time-Series UI  
**Prompt:** "Use the Wikipedia API to fetch 3 related events for the searched query and overlay them as vertical markers on the plotly time-series."  
**What went wrong:** Wikipedia summaries do not consistently map to specific chronological dates in a parseable way. I couldn't reliably place them on the X-axis of my temporal plot.  
**How I fixed it:** I changed the visual approach. Instead of trying to parse exact dates from Wikipedia text, I distributed the 3 extracted 'events/context markers' evenly across the most recent 3 days of the active timeline. This offers investigators immediate historical context right where the user's attention is focused during live investigations.
