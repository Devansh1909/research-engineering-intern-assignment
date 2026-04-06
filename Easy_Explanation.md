# bro_please_readme.md — The NarrativeScope Field Guide

> Plain-English guide to every feature, every algorithm, and every design decision. Written for a curious non-developer and a sceptical researcher simultaneously.

---

## What is NarrativeScope?

It's an investigative platform for tracing **how digital narratives spread across Reddit**. You give it a topic — say, *"government manipulation of public opinion"* — and it shows you:

- Which posts are semantically related (even if they use zero of those words)
- When the topic spiked in activity
- What sub-topics cluster together
- Which accounts are amplifying it
- Whether there are coordination signals

Think of it as a journalist's research desk that reads 50,000 posts in 2 seconds.

---

## The 6 Sections — What Each One Does

### 🔎 Section 1: Narrative Search

**What question it answers:** *"Find me posts about X."*

**How it actually works:**

Every Reddit post in the dataset is converted into 384 numbers called an **embedding** — a mathematical fingerprint of its *meaning*, produced by the `all-MiniLM-L6-v2` sentence transformer model. Your search query is converted the same way. Then [FAISS](https://faiss.ai/) (Facebook AI Similarity Search) finds the posts whose fingerprints are most similar to yours in milliseconds, even across 100,000 posts.

**Why this beats keyword search:**

| Keyword Search | Semantic Search (NarrativeScope) |
|---|---|
| "manipulation of public opinion" → only exact phrase matches | → also finds "astroturfing campaigns", "sockpuppet networks", "manufactured consensus" |
| Misses synonyms | Captures conceptual similarity |
| Breaks on paraphrasing | Robust to all rephrasing |

**The similarity score (0–1):**

| Range | Interpretation |
|---|---|
| 0.85+ | Essentially the same idea, different words |
| 0.60–0.85 | Strongly related — overlapping framing |
| 0.40–0.60 | Related — thematically similar |
| 0.25–0.40 | Weakly related — borderline |
| < 0.25 | Discarded (below threshold) |

**Zero-keyword-overlap examples that actually work:**

1. Query: *"government manipulation of public opinion"* → finds posts about *"astroturfing"* and *"sockpuppet campaigns"*
2. Query: *"economic hardship affecting families"* → finds posts about *"can't afford groceries"* and *"rent is killing me"*
3. Query: *"distrust in mainstream media reporting"* → finds posts about *"MSM lies"* and *"narrative control by corporations"*

**What "AI Summary" means:** If you add a Gemini API key, the top search results are sent to Google's Gemini model, which synthesises them into a one-paragraph narrative summary. Without the key, you get a rule-based fallback.

---

### 📊 Narrative Health Score

**What question it answers:** *"Is this narrative healthy or is it being artificially amplified?"*

A composite score (0–100) built from three dimensions:

| Dimension | Measures | Formula |
|---|---|---|
| **Breadth** | How many unique subreddits are discussing this | `unique_subreddits / (n_results / 5)`, capped at 1.0 |
| **Velocity** | Is activity growing or dying? | Last-third post count ÷ first-third post count |
| **Distribution** | How spread across accounts? | 1 − Herfindahl-Hirschman Index (HHI) of author shares |

**Interpreting the score:**

- **70–100 (Healthy, teal):** Broad, growing, many voices. Organic-looking spread.
- **45–69 (Moderate, yellow):** Some concentration or stagnation.
- **0–44 (Concentrated, red):** Dominated by a few accounts or communities — investigate further.

**Important nuance:** A low score doesn't *prove* manipulation. A niche expert topic will also score low (few authors, one subreddit). Always combine with the Anomaly Detector findings.

---

### 📈 Section 2: Temporal Patterns

**What question it answers:** *"When did this narrative happen, and is it accelerating?"*

Shows a dual-layer chart:
- **Bars:** Raw daily post count
- **Teal line:** 7-day rolling average (smooths out weekly cycles like "less posting on weekends")

**The subreddit breakdown beneath it:** Shows which communities were active *when*, revealing cross-community coordination timing.

**Wikipedia event markers (Live Mode):** When you've fetched live Reddit data AND have an active search query, gold stars appear on the chart marking real-world events from Wikipedia related to your topic. This lets you visually correlate: *"The spike on March 15 coincides with the Wikipedia article for Event X."*

**What the AI summary is doing:** It detects the peak date, computes a trend direction (growing/declining/stable), and generates a plain-English paragraph. Without Gemini, it produces a templated statistical summary.

---

### 🧩 Section 3: Topic Clustering

**What question it answers:** *"What are the distinct sub-topics inside this narrative?"*

**The pipeline (3 stages):**

```
Post Texts
    ↓
Sentence Transformer (all-MiniLM-L6-v2)
    → 384-dimensional embeddings
    ↓
UMAP (Uniform Manifold Approximation and Projection)
    → 2D (for visualization) and 5D (for clustering)
    ↓
HDBSCAN (Hierarchical Density-Based Spatial Clustering)
    → Cluster labels (-1 = noise, 0,1,2... = topics)
```

**UMAP in plain English:** Imagine plotting 100,000 posts in 384-dimensional space (impossible to visualise). UMAP squishes this into 2D while preserving the "neighbourhood structure" — posts that mean similar things end up close together.

**HDBSCAN in plain English:** Imagine placing posts as dots on a map. HDBSCAN finds natural density peaks — groups of dots packed tightly together — regardless of shape. Posts in sparse areas become "noise" (label = -1). Unlike K-Means, you don't need to specify how many clusters you want; the algorithm infers it from the data.

**Why HDBSCAN over K-Means:**

| K-Means | HDBSCAN |
|---|---|
| You must specify K | Auto-detects cluster count |
| Assumes round clusters | Finds arbitrary shapes |
| Every point gets a cluster | Noise points are excluded (honest) |
| Sensitive to outliers | Robust to outliers |

**The Min Cluster Size slider:** Controls how tight a group must be to form a topic. Higher → fewer, broader topics. Lower → more, finer-grained topics. Default is `√n` (square root of total posts), which is an established heuristic for this scale.

**Topic labels:** The top 5 most frequent non-stopword terms from each cluster become the label. This is TF-IDF style extraction (not a language model), so labels are descriptive but sometimes awkward.

**Datamapplot vs Plotly:** If the `datamapplot` library is installed, you get its premium interactive visualisation with labeled region annotations. If not, it falls back to a Plotly scatter plot — both show the same underlying UMAP coordinates.

**"Noise" posts:** Posts that don't belong to any cluster. High noise (>60%) means the content is very diverse — consider lowering Min Cluster Size.

---

### 🕸️ Section 4: Influence Network

**What question it answers:** *"Who are the key amplifiers, and how connected are they?"*

**How the network is built:**

For every pair of authors who both posted in the same subreddit, an edge is drawn between them. The edge weight is the number of subreddits they co-appeared in. This creates a **co-occurrence network** — not a reply network (we don't have that data), but a co-presence network.

**The three centrality metrics:**

| Metric | Plain-English meaning | Good for |
|---|---|---|
| **PageRank** | How often would a random surfer land on this node? Accounts connected to other highly-connected accounts score higher. | Finding the real influence hubs |
| **Betweenness** | How many shortest paths between other nodes pass through this one? | Finding brokers/bridges — accounts that connect otherwise-separate communities |
| **Degree** | How many direct connections? | Findign the most active connectors |

**The composite influence_score:** A weighted average: `0.5 × PageRank + 0.3 × Betweenness + 0.2 × Degree_normalised`. This balances reach with bridging importance.

**Community detection (Louvain):** Louvain maximises [modularity](https://en.wikipedia.org/wiki/Modularity_(networks)) — it finds groups of accounts that are more densely connected to each other than to the rest. Coloured clusters in the network graph are Louvain communities.

**Robustness test (node removal):** Removes one account and measures how many new disconnected components form. If removing Account X creates 5 new islands → that account is a critical bridge. If the network stays the same → it's resilient.

---

### ⚠️ Anomaly Detector

**What question it answers:** *"Is there evidence of coordinated inauthentic behaviour?"*

Runs three automated checks:

**1. Cross-Subreddit Spammers**
Finds accounts that posted in >5 different subreddits within the filtered dataset. Legitimate users can be active across many communities, but high cross-subreddit presence combined with a specific search topic is worth investigating.

**2. Statistical Activity Spikes (Z-score)**
For each day, computes how many standard deviations (`σ`) the post count is above the 7-day rolling mean. Flags days with Z-score > 2.5σ. A Z-score of 3σ means the spike happens by random chance less than 0.3% of the time under normal conditions. Potential causes: viral organic spread, bot campaign activation, or real-world news event — correlate with Section 6 (Event Correlation).

**3. Near-Duplicate Propagation**
Groups posts by their first 80 characters (lowercased). If 3+ different accounts posted near-identical text in 4+ instances, that's a strong coordination signal — humans rarely post the same text from different accounts organically.

**Critical nuance:** These are *signals*, not proof. Each flag has an explanation you should read before drawing conclusions.

---

### 🧵 Section 5: Thread Puller

**What question it answers:** *"For one specific topic cluster, where did it start, where did it go, and who drove it?"*

After clustering runs, you pick one topic from the dropdown and see:
- **Timeline:** Daily post volume for just that cluster
- **Community Crossover:** Which subreddits hosted this sub-topic (bar chart)
- **Key Voices:** The most active accounts within just this thread

This is useful for tracing a specific sub-narrative (e.g., tracking just the "vaccine mandates" cluster within a broader "COVID conspiracy" search) across time and communities.

---

### 🌍 Section 6: Real-World Event Correlation

**What question it answers:** *"What was happening in the real world when this narrative spiked?"*

A direct Wikipedia search that pulls the article summary for any topic you type. Use this to manually cross-reference the temporal patterns in Section 2 with documented real-world events.

**Limitation:** This is not automatic correlation — it's a research aid. You look at the time-series spike, then search for what was happening on that date.

---

## The Live Reddit Mode

**Setup:** You need a free Reddit API credential pair (Client ID + Secret) from [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps). Create a "script" type app.

**What it does:** Fetches up to 50 recent posts from any subreddit (or r/all) matching a keyword, normalises them into the same schema as the static dataset, and appends them to the working dataframe. The app then recomputes embeddings for the combined dataset and reruns all analyses.

**The 30-second cooldown:** Prevents you from spamming the Reddit API and getting rate-limited. The button is disabled during cooldown with a countdown timer.

**Persistence:** Fetched live posts are saved to `data/live_cache.jsonl`. They persist across app restarts. Each fetch appends to this file (duplicates are dropped by text+author+datetime).

**Setting credentials via environment variables:**
```bash
# .env or shell environment
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_secret
```
This avoids re-entering them every session. The sidebar will show a green "✅ credentials loaded from environment" message.

---

## Common Gotchas and FAQs

**Q: Why does clustering give 0 topics?**
A: Your Min Cluster Size is too high for the filtered dataset size. Decrease it. For small result sets (<500 posts), try values of 5–15.

**Q: Why are search results returning weird posts?**
A: Semantic search finds conceptually similar posts, not keyword matches. Some results will look surprising but are thematically related. Check the similarity score — anything below 0.35 is weakly related.

**Q: Why does the network look like a hairball?**
A: With large datasets (>5,000 posts), co-occurrence networks become very dense. Apply a subreddit filter in the sidebar to focus on a specific community and make the graph interpretable.

**Q: Why is the AI Summary saying generic things?**
A: Without a Gemini API key, summaries are templated. Add your free key (from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)) and the summaries become context-aware natural language.

**Q: The app crashes on startup with a FAISS or torch error.**
A: The `USE_TF=0` and `USE_TORCH=1` environment variables at the top of `app.py` suppress the TensorFlow/PyTorch conflict. If you're still seeing crashes, run `pip install --upgrade sentence-transformers faiss-cpu`.

**Q: How big of a dataset can this handle?**
A: The pipeline runs comfortably on ~50,000 posts on a consumer laptop (~8GB RAM). For larger datasets, UMAP becomes the bottleneck. Consider increasing Min Cluster Size to reduce the clustering computation load.

---

## Architecture in 2 Minutes

```
data/                          ← JSONL Reddit data files
  ├── *.jsonl                  ← Static dataset
  └── live_cache.jsonl         ← Live-fetched posts

data_loader.py                 ← Loads JSONL, normalises schema, DuckDB for SQL queries
search_engine.py               ← sentence-transformers embeddings + FAISS index
clustering.py                  ← UMAP + HDBSCAN + cluster label extraction
network_analysis.py            ← NetworkX co-occurrence graph + centrality + Louvain
genai_summarizer.py            ← Gemini API client for section summaries
realtime.py                    ← PRAW-based Reddit live fetcher
events.py                      ← Wikipedia API event lookup

app.py                         ← Main Streamlit app (all UI logic lives here)
```

**Data flow:**
```
Raw JSONL → data_loader (normalize) → DuckDB (filter) → embeddings (sentence-transformer)
                                                              ↓
                                          FAISS index ← semantic search
                                          UMAP → HDBSCAN → clusters
                                          NetworkX → centrality → Pyvis graph
                                          Gemini → natural language summaries
```

---

## Design Decisions & Why We Made Them

| Decision | Why |
|---|---|
| **HDBSCAN over K-Means or LDA** | No need to specify K; handles nonspherical clusters; honest noise label |
| **all-MiniLM-L6-v2 over larger models** | 80 sentences/sec on CPU — fast enough for interactive UI; 384-dim still captures rich semantics |
| **FAISS over cosine similarity loop** | O(log n) approximate NN vs O(n) exact → ~1000x faster at 50K+ posts |
| **DuckDB over pandas queries** | SQL syntax clarity for complex multi-filter aggregations; columnar for analytics |
| **Pyvis over D3.js** | Eliminates frontend JS complexity; physics-based layout is immediately interpretable |
| **Vertical narrative flow (no tabs)** | Insights cascade: search → time → clusters → network → anomaly. Each section feeds the next. |
| **HHI for distribution** | Established economics metric for market concentration; maps cleanly to narrative concentration |
| **Louvain for community detection** | O(n log n) complexity; widely validated for social network graphs; no cluster count needed |

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data (if you don't have real data)
python generate_sample_data.py

# Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

*NarrativeScope — Built by Devansh for the SimPPL Research Engineering Internship.*
*Stack: Streamlit · sentence-transformers · FAISS · HDBSCAN · UMAP · NetworkX · DuckDB · Plotly · Google Gemini*
