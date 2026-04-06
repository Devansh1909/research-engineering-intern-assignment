# NarrativeScope: A Quick User Guide

This guide is a quick visual walkthrough of NarrativeScope. I put this together to show you exactly how the dashboard works and what the different sections do.

---

## 1. Starting Point: Semantic Search

Most investigations don't start with perfect keywords. Sometimes you're looking for a general idea, but people are using completely different words to talk about it.

![Semantic Search](screenshots/search.png)

### How I set this up:
- Instead of exact keyword matching, the app uses a **sentence-transformers model (`all-MiniLM-L6-v2`)** to turn your search into a math vector (an embedding).
- It runs that vector through **FAISS** (Facebook AI Similarity Search) to find posts that are conceptually related.
- **For example:** If you search *"government manipulation"*, it will find posts talking about *"coordinated astroturfing campaigns"*, even if those exact words aren't in the post.
- **Edge cases:** If you submit an empty query or type in another language, the app won't crash. I added some simple UI banners that catch those inputs and warn you.

### AI Summaries:
If you look under the results, you'll see a teal box. The app takes the search results and sends them to the **Google Gemini API**, which returns a quick, plain-English summary of what people are actually talking about in the data.

---

## 2. Tracking the Timeline

Once you find a topic, you usually want to know when it spiked.

![Time Series](screenshots/timeseries.png)

### What the chart shows:
- **Daily Volume:** The bars just show the raw count of posts over time.
- **7-Day Rolling Average:** The teal line is a rolling average. I added this to smooth out normal dips (like people posting less on weekends) so you can see the actual trend.
- **Wikipedia Event Overlay (Gold Stars):** If you turn on Live Mode, the app pings the Wikipedia API to look up real-world news events that match your search terms. It places gold stars on the timeline so you can see if an online spike happened right after a major news event.

---

## 3. Sorting the Noise: Topic Clustering

When you get 5,000 search results, reading them all isn't practical. I built a clustering section to organize them into sub-topics automatically.

![Topic Clusters](screenshots/clusters.png)

### Simply put:
- The app takes the 384-dimensional embeddings from the search and squishes them down to 2D using **UMAP**. This groups similar posts together visually on a map.
- Then I used **HDBSCAN** to find the dense clusters.
- **Why use HDBSCAN?** Algorithms like K-Means force every single post into a cluster, which creates messy results. With HDBSCAN, if a post is just random noise, it gets labeled as noise and ignored.
- **Controls:** You can use the "Min Cluster Size" slider in the sidebar to control how granular you want the topics to be.

---

## 4. Finding the Key Accounts: Influence Network

Who is driving the conversation? This section looks at the actual accounts behind the posts.

![Network](screenshots/network.png)

### How the map works:
I used **NetworkX** to build a co-occurrence graph. Basically, if two authors are active in the same subreddits talking about the same topic, the app draws a line between them.

### What the math is doing:
- **PageRank:** Finds the "hubs"—the accounts that are highly connected to other active accounts.
- **Betweenness:** Finds the "bridges"—the accounts sitting between different communities and connecting them.
- **Louvain Communities:** Colors the nodes based on which distinct community group they belong to.

### The Removal Test:
At the bottom of this section, there's a feature to simulate removing an account. You pick a highly-connected user and "remove" them from the graph. The app recalculates everything to show you how much the network fragments when that hub is gone. 

---

## 5. Live Data Fetching

Static data is good for testing, but I wanted the dashboard to be able to handle current events.

![Live Mode](screenshots/live_mode.png)

### Pulling from Reddit:
- Using **PRAW**, you can connect to the Reddit API and fetch live posts.
- **How it handles the new data:** When you fetch new posts, the app cleans them up to match the existing database format. It re-calculates the text embeddings, updates the FAISS index, re-runs the HDBSCAN clustering, and redraws the network graph instantly. Everything updates without losing your search session.
