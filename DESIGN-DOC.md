# NarrativeScope — Design Document

**Author:** Devansh  
**Date:** March 2026  
**Status:** Final

---

## 1. What Am I Building and Why?

When I first read the SimPPL assignment, my instinct was to build a standard dashboard with a few charts and a search bar. But after looking at other forks, I noticed they all looked exactly the same. I wanted to build something that actually helps an investigator trace a story from start to finish.

I thought about how a real investigation works: you find a weird post, then try to figure out who else is talking about it, when it spiked, and if it's connected to actual news. That's why I built NarrativeScope to act more like a research tool than just a static dashboard.

## 2. Core Decisions

### 2.1 One Page vs. Tabs

A lot of dashboards split everything into tabs: "Time Series", "Network", "Clustering". I didn't want to do that. When you split the data up, you lose the context. I decided to make it one long scrolling page. You start by searching at the top, and as you scroll down, each section builds on the last one.

### 2.2 Using DuckDB Instead of Pandas

The data is in JSONL format. I started out using pandas, but the app got really slow when filtering data. I switched to DuckDB because:
- It processes SQL queries way faster than pandas group-by functions.
- It doesn't need to load the entire JSONL file into memory.
- Dealing with time-series stats was a lot easier with SQL window functions.

### 2.3 How the Search Works

**The problem:** If you search for "government manipulation", a regular keyword search won't find a post that says "astroturfing campaign" because they don't share any words.

**My fix:**
1. I used `sentence-transformers/all-MiniLM-L6-v2` to turn the posts into 384-dimensional math vectors (embeddings). I picked this model because it runs decently fast on a CPU without crashing.
2. I built a FAISS index to do the heavy lifting of searching the vectors.
3. When you type a query, it turns your text into a vector too, and FAISS finds the closest matching posts.

**Edge cases:**
- If you leave the box empty, it just shows a prompt.
- If you type a tiny word, it warns you the results might be off.
- If you type in Spanish or French, it detects the language and warns you, but it still tries to search.

### 2.4 The Network Graph

I had to figure out how to draw connections between users.

- **How lines are drawn:** If two users post in the same subreddit about the same topic, they get linked. The more they cross paths, the stronger the link.
- **Scoring influence:** I used **PageRank** because it rewards people linked to *other* important people, not just someone spamming posts. I also used **Betweenness Centrality** to find accounts acting as bridges between different groups.
- **Node Removal test:** I built a tool at the bottom of the section that lets you click a prominent user and "delete" them to see how the graph physically breaks apart.

### 2.5 Clustering with HDBSCAN

**Why not K-means?**
K-means forces every single post into a cluster. Text topics don't work like that—a lot of posts are just random thoughts. HDBSCAN is better because it groups the tight clusters together and throws everything else out as "noise". 

The user can control the cluster tightness with a slider. If you push the slider to 100, almost everything becomes noise, which the app handles safely.

To show the clusters visually, I used UMAP to crush the dimensions down to 2, and then fed them into Datamapplot.

### 2.6 AI Summaries

Instead of making you read the raw data, I hooked the app up to Gemini 1.5 Flash. I passed the actual numbers and chart data directly into the prompt so the AI can write a quick paragraph explaining the trend in plain English. I made sure to add a fallback so the app won't crash if the API key is missing.

### 2.7 Live Data Mode

I added a feature that hooks into the Reddit API using PRAW. You can fetch live posts about a topic, and the app will merge them with the existing data. It immediately updates the search index and re-draws the graphs so you can investigate things happening right now.

## 3. Tech Stack Specs

| Component | Tool / Algorithm | Settings | Library |
|-----------|----------------|----------------|---------|
| Search Engine | `all-MiniLM-L6-v2` | 384-dim, cosine similarity | `sentence-transformers`, `faiss-cpu` |
| Clustering | HDBSCAN | tunable `min_cluster_size` | `hdbscan` |
| 2D Mapping | UMAP | 2 components, `n_neighbors=15` | `umap-learn` |
| Graph scoring | PageRank + Betweenness | `alpha=0.85` | `networkx`, `community` |
| Text Summaries | Gemini 1.5 Flash | `temperature=0.3` | `google-generativeai` |
| Visualizing topics | Datamapplot | Auto-labeling | `datamapplot` |

## 4. Future Ideas

If I had more time, I would:
- Connect Twitter/X data to see how topics jump between platforms.
- Track how the network graph changes day-by-day instead of just showing one snapshot.
- Make the clustering run on a GPU for faster mapping.
