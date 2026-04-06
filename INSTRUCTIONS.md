# Research Engineering Intern Assignment

We're thrilled you're interested in joining SimPPL! This assignment is designed to give you a practical, hands-on experience in social media analysis, mirroring the kind of work you'd be doing with us. It's structured like a mini-research project, challenging you to explore how information spreads across social networks, specifically focusing on content from potentially unreliable sources. Instead of building a data collection tool from scratch for this initial exercise, you'll be provided with existing social media data. Your task is to design and build an interactive dashboard to analyze and visualize this data, uncovering patterns and insights about how specific links, hashtags, keywords, or topics are being shared and discussed. This will allow you to focus on your data science, machine learning, and analysis skills, which are crucial to the research we conduct at SimPPL.

## Why do we care about this?

We have built tools for collecting and analyzing data from Reddit and Twitter including now-obsolete platform [Parrot](https://www.youtube.com/watch?v=FVetP1D5u0o) to study the sharing of news from certain unreliable state-backed media. To ramp you up towards understanding how to go about extending such platforms, and to expand your understanding of the broader social media ecosystem, we would like you to develop a similar system to trace digital narratives. We would like you to present an analysis of a broader range of viewpoints from different (apolitical / politically biased) groups. You may even pick a case study to present e.g. a relevant controversy, campaign, or civic event. The goal is for you to be creative and explore what could be possible to contribute a meaningful assignment rather than just sticking to our instructions.

In the long run, this research intends to accomplish the following objectives:

1. Track different popular trends to understand how public content is shared on different social media platforms.
2. Identify digital threats such as actors and networks promoting scams, spam, fraud, hate, harassment, or misleading claims.
3. Analyze the trends across a large number of influential accounts over time in order to report on the _influence_ of a narrative.

## Task Objectives

1. **Visualize Insights**: Tell a story with a graph, building intuitive and engaging data visualizations.

2. **Apply AI/ML**: Use LLMs and machine learning to generate metrics and enhance your analysis.

3. **Build and Deploy an Investigative Reporting Dashboard**: Develop and host an interactive dashboard to showcase your analysis. You should be able to query the dashboard to generate meaningful insights.

Before moving on, please understand that even if you accomplish all of these objectives that does not mean you have submitted a *good* assignment. This is simply because almost all of the applying candidates accomplish all these objectives very well, making it a bare minimum for qualifying.

**Note:** What we ultimately use to evaluate applicants is how well the assignment reflects your engineering judgment, how you make decisions under ambiguity, and how robust your system is when we stress-test it, not how polished your README prose is. That's where we want to see how you think, and it comes across very clearly the minute we have interviews with applicants.

## Examples of "How to Tell a Story with Data"

There are some hosted web demos that **tell a story** with data that you should look into. We do not expect you to replicate or copy any of these but we do want you to understand the "tell us a story with data" goal of this assignment better by looking at these:

1. [Fabio Gieglietto's TikTok Coordinated Behavior Report](https://fabiogiglietto.github.io/tiktok_csbn/tt_viz.html)
2. [Integrity Institute's Information Amplification Dashboard](https://integrityinstitute.org/blog/misinformation-amplification-tracking-dashboard)
3. [News Literacy Project Dashboard](https://misinfodashboard.newslit.org/)
4. [Tableau examples](https://public.tableau.com/app/search/vizzes/misinformation) (note: we don't use Tableau, and expect you to use Python or Javascript for this assignment, but these are interesting examples for inspiration)

## Rubric for Evaluation

Below is the rubric we will use for your evaluation. Remember the note above: even if yours doesn't meet all these rubrics but it is unique from other submissions while reflective of your technical expertise, we would be open to advancing you in the interview process.

### 1. Documentation and Public Hosting (IMPORTANT)

- Is the solution well-documented such that it is easy to understand its usage?
- Is the solution hosted on a publicly accessible web dashboard with a neatly designed frontend?

### 2. Time-Series and Network Visualizations (IMPORTANT)

Does the solution visualize summary statistics for the results? For example:

- Time series of the number of posts matching a search query
- Time series of key topics, themes, or trends in the content
- Community breakdown of accounts that are key contributors to a set of results
- Network visualization of accounts that have shared a particular keyword, hashtag, or URL

For network analysis specifically: implement an influence or centrality score for accounts in the dataset. You must choose an appropriate algorithm (e.g. PageRank, betweenness, Louvain community detection), and your choice must be visible in the code and consistent with how you present the results in the dashboard. We will look at edge cases: what does your graph look like with a highly connected node removed? Does it handle disconnected components without crashing?

Each time-series plot must include a GenAI-generated plain-language summary beneath it so that a non-technical audience can understand the trend without interpreting the chart themselves. These summaries must be generated dynamically based on the actual data returned by a query, not hardcoded.

### 3. Semantic Search and Chatbot (IMPORTANT)

Implement search or a chatbot that returns results ranked by relevance, not just keyword match. We will test it with queries that have zero keyword overlap with correct answers. Include 3 such examples in your README showing: the query, the result returned, and a one-line note on why it is the correct result. This is binary: either semantic search works or it doesn't.

Your dashboard must also handle these cases gracefully without crashing: empty results, very short queries, and at least one non-English input. We will test these during evaluation.

**Nice-to-have:** After returning results, the chatbot proposes 2-3 related queries the user might want to explore next. Not required, but a strong signal of product thinking.

### 4. Topic Clustering and Embedding Visualization (IMPORTANT)

You must cluster posts by topic. Your dashboard must expose the number of clusters as a tunable parameter. We will test what happens at the extremes: the results must be coherent and your UI must not break.

As part of the clustering implementation, you must also visualize the topic model embeddings using one of the following: [Tensorflow Projector](https://projector.tensorflow.org/), [Datamapplot](https://github.com/TutteInstitute/datamapplot), or [Nomic](https://atlas.nomic.ai/). The visualization must be interactive and embedded in or linked from your dashboard.

In your README, for each ML/AI component, state only: the model or algorithm name, the key parameter choices you made (e.g. embedding dimension, number of clusters, distance metric), and which library or API call you used. One or two lines per component. We will verify these match your actual code.

### 5. Additional Nice-to-Have Features

- Connecting offline events (e.g. from Wikipedia or news APIs) with online post sharing for specific searches, mapping real-world events to spikes in online narratives
- Connecting multiple platform datasets together to search across social platforms in a single query

---

## Instructions for Submission

### Setting Up

1. Fork this repository by clicking the "Fork" button in the top right corner.
2. Clone your fork: `git clone <your_forked_repository_url>`
3. Navigate into the cloned directory and begin development.

### Committing Your Work

Commit regularly and meaningfully throughout development. We will review your full commit history. What we are looking for:

- Evidence of iteration: commits that show how your approach evolved
- For any AI-assisted component, we expect to see multiple commits that trace the progression from your first working version through to your final version. The commit history across all versions is what we review, not just the end state. A strong engineer's history shows targeted, incremental changes. A history of complete rewrites at each step is a signal that you were re-prompting from scratch rather than understanding and fixing your code.

Do not squash or rebase your commits before submitting.

### Pushing and Submitting

```
git add .
git commit -m "Your descriptive commit message"
git push origin main
```

Please notify us of your submission by emailing simppl.collabs@gmail.com with the subject line "Submitting Research Engineer Intern Assignment for SimPPL".

---

## Submission Requirements

Please ensure you include:

1. A detailed README file with screenshots of your solution and a URL to your publicly hosted web platform.
2. A link to a video recording of your dashboard hosted on YouTube or Google Drive. Walk us through the platform and explain your design decisions out loud: not what the code does line by line, but why you made the choices you did.
3. A file called `<yourname>-prompts.md` if you used AI to assist with any code. Number each prompt sequentially so we can follow your progression. For each prompt, include the component it was for, the prompt itself, and a one-line note on what was wrong with the output and how you fixed it. Submissions where every prompt succeeded on the first try will be treated skeptically, as we know that's not how AI-assisted development actually works.

---

## AI Usage Policy

We are an AI-first company and we appreciate the thoughtful use of coding copilots. If you use AI to write any code, the `<yourname>-prompts.md` file described above is mandatory. Without it, we cannot assess your ability to use AI tools effectively, and you lose credit you would otherwise receive.

We have seen enough AI-generated code that looks correct and breaks silently to treat this seriously. We are hiring engineers who can leverage AI tools with judgment, not engineers who delegate their thinking to them.

---

## Resources

1. [OSINT Tools](https://start.me/p/0Pqbdg/osint-500-tools)
2. [Colly](http://go-colly.org/)
3. [AppWorld](https://appworld.dev/)
4. [Scrapling](https://github.com/D4Vinci/Scrapling)
5. [Selenium](https://www.selenium.dev/)
6. [Puppeteer](https://pptr.dev/)
7. [DuckDB](https://github.com/duckdb/duckdb)
8. [Cloudflare Workers](https://workers.cloudflare.com/)
9. [Apache Superset](https://github.com/apache/superset)
10. [Terraform](https://www.hashicorp.com/en/products/terraform)

### Link to the Dataset

<a href="https://drive.google.com/drive/folders/13cYfPIV65j5AAh9GjuZR94sAx-7EFjnp?usp=sharing">Dataset</a>

---

#### A Final Note

Focus on the analysis you are presenting and the story you are telling through it. A well-designed and robust system is more important than a complex one with a ton of features. We will stress-test what you build: edge cases, unexpected inputs, parameter extremes. The goal is not to impress us with volume of output but to show us that you understand what you built well enough to defend every decision in it.

We're excited to see your solution!
