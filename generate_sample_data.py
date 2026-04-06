"""
generate_sample_data.py — Generate realistic sample Reddit data for testing.

Run this script to create a sample dataset if you haven't downloaded
the actual dataset from Google Drive yet. This generates ~2000 posts
across multiple subreddits with realistic distributions.

Usage: python generate_sample_data.py
"""

import json
import random
import os
from datetime import datetime, timedelta
from pathlib import Path

# Seed for reproducibility
random.seed(42)

# Configuration
NUM_POSTS = 2000
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "sample_reddit_data.jsonl"

# Realistic subreddit distribution
SUBREDDITS = {
    'politics': 0.20,
    'news': 0.15,
    'worldnews': 0.12,
    'technology': 0.10,
    'science': 0.08,
    'Conservative': 0.08,
    'Liberal': 0.05,
    'conspiracy': 0.05,
    'media_criticism': 0.05,
    'neutralnews': 0.04,
    'PoliticalDiscussion': 0.04,
    'AskReddit': 0.04,
}

# Realistic author distribution (power law — few prolific, many occasional)
NUM_AUTHORS = 350
AUTHORS = [f"user_{i:04d}" for i in range(NUM_AUTHORS)]
# Top 10% of authors generate 50% of posts
AUTHOR_WEIGHTS = [10.0 if i < NUM_AUTHORS * 0.1 else 1.0 for i in range(NUM_AUTHORS)]

# Topic templates with realistic content
TOPIC_TEMPLATES = {
    'election_discourse': [
        "The upcoming election highlights the deep divisions in how we understand democratic participation",
        "Voter turnout projections suggest significant shifts in traditional voting blocs this cycle",
        "Campaign finance records reveal interesting patterns in donation sources across parties",
        "Early voting data shows unprecedented engagement among younger demographics",
        "Political advertising spending has reached record levels across social media platforms",
        "Analysis of candidate speeches reveals shifting rhetoric on key policy issues",
        "Electoral college projections continue to show tight races in several swing states",
        "Grassroots organizing efforts are reshaping traditional campaign strategies",
        "Debate performance analysis shows contrasting approaches to policy presentation",
        "Registration drives in underserved communities are changing the electoral math",
    ],
    'media_trust': [
        "Public trust in mainstream media continues to decline according to recent surveys",
        "Alternative news sources are gaining audiences as people seek different perspectives",
        "Media bias studies show increasing polarization in news coverage across outlets",
        "The distinction between news reporting and opinion commentary has become increasingly blurred",
        "Social media algorithms may be creating echo chambers that reinforce existing beliefs",
        "Fact-checking organizations face challenges in keeping up with the pace of information",
        "Local news outlets are disappearing as national narratives dominate the conversation",
        "Podcast and independent journalist audiences are growing faster than traditional media",
        "The relationship between clickbait headlines and public understanding of complex issues",
        "Media literacy education should be a priority in addressing misinformation challenges",
    ],
    'tech_regulation': [
        "Proposed legislation would require algorithm transparency from major social platforms",
        "Content moderation policies continue to be debated across the political spectrum",
        "Data privacy concerns drive growing support for European-style digital regulations",
        "Antitrust investigations into major tech companies raise questions about market concentration",
        "Section 230 reform proposals could fundamentally change how platforms operate",
        "AI-generated content creates new challenges for platform governance and authenticity",
        "Digital literacy programs aim to help users navigate the evolving information landscape",
        "The tension between free speech principles and content safety continues to grow",
        "Cross-border data regulation creates compliance challenges for global platforms",
        "Encryption policy debates highlight conflicting priorities between security and privacy",
    ],
    'health_information': [
        "Public health communication strategies need to adapt to the social media environment",
        "Medical misinformation spreads faster than corrections on most online platforms",
        "Trusted health institutions are working to improve their online communication presence",
        "The pandemic accelerated adoption of telehealth and digital health resources",
        "Medical professionals are increasingly using social media for health education outreach",
        "Peer support communities online provide valuable resources but also carry risks",
        "The challenge of communicating uncertainty in evolving scientific understanding",
        "Health literacy gaps contribute to vulnerability to misleading medical claims",
        "Community health workers are bridging the gap between clinical information and public understanding",
        "Evidence-based medicine faces challenges in competing with compelling but unverified narratives",
    ],
    'climate_debate': [
        "Climate policy discussions reveal deep divides between economic and environmental priorities",
        "Renewable energy adoption metrics show accelerating transition in several sectors",
        "The framing of climate change as a political issue versus scientific consensus remains contentious",
        "Urban planning decisions increasingly incorporate climate resilience considerations",
        "Corporate sustainability commitments face scrutiny for potential greenwashing practices",
        "Grassroots environmental movements are using digital organizing to amplify their message",
        "The economic argument for climate action is gaining traction among previously skeptical groups",
        "Extreme weather events are shifting public perception of climate-related risks",
        "International climate agreements face implementation challenges at the national level",
        "Youth climate activism is driving new conversations about intergenerational responsibility",
    ],
    'social_justice': [
        "Community organizations are developing new frameworks for addressing systemic inequities",
        "Criminal justice reform proposals generate debate about competing approaches to public safety",
        "Educational equity initiatives aim to address disparities in resource allocation",
        "Housing affordability discussions highlight tensions between development and preservation",
        "The intersection of technology access and social equity continues to evolve",
        "Workplace diversity initiatives face both support and criticism from different perspectives",
        "Community-led accountability mechanisms offer alternatives to traditional oversight structures",
        "Economic mobility research reveals persistent barriers across different demographic groups",
        "The evolution of civil rights discourse in the digital age brings new challenges",
        "Restorative justice practices gain traction as communities seek alternatives to punitive approaches",
    ],
    'economic_anxiety': [
        "Inflation concerns continue to impact household purchasing decisions across income levels",
        "Housing market analysis reveals growing affordability challenges in major metropolitan areas",
        "Labor market shifts suggest fundamental changes in employment patterns and expectations",
        "Student debt policy debates highlight tensions between individual and collective responsibility",
        "Small business resilience in the face of changing economic conditions varies significantly",
        "Supply chain vulnerabilities exposed by global disruptions drive reshoring conversations",
        "The gig economy's growth raises questions about worker classification and protection",
        "Wealth inequality metrics show growing disparities despite overall economic growth",
        "Financial literacy initiatives aim to empower consumers in complex economic environments",
        "Automation and AI adoption in the workplace creates anxiety about job displacement",
    ],
}

# Variations to add natural noise
PREFIXES = [
    "", "Interesting: ", "This is worth discussing - ", "My take on this: ",
    "Can we talk about how ", "Not sure I agree with this but ", "Important thread: ",
    "Thread - ", "Genuine question about ", "Unpopular opinion: ",
    "Follow-up to yesterday's discussion: ", "Breaking: ", "Analysis: ",
    "PSA: ", "Discussion: ",
]

SUFFIXES = [
    "", " What are your thoughts?", " I'd love to hear other perspectives.",
    " This seems really important.", " Am I wrong about this?",
    " Source in comments.", " thoughts?", " Discuss.",
    " This is getting out of hand.", " We need to pay more attention to this.",
]


def weighted_choice(items, weights):
    """Choose from items with given weights."""
    total = sum(weights)
    r = random.uniform(0, total)
    cumulative = 0
    for item, weight in zip(items, weights):
        cumulative += weight
        if r <= cumulative:
            return item
    return items[-1]


def generate_post(post_id: int, base_date: datetime) -> dict:
    """Generate a single realistic Reddit post."""
    # Choose subreddit (weighted)
    subs = list(SUBREDDITS.keys())
    sub_weights = list(SUBREDDITS.values())
    subreddit = weighted_choice(subs, sub_weights)
    
    # Choose author (power-law distribution)
    author = weighted_choice(AUTHORS, AUTHOR_WEIGHTS)
    
    # Choose topic and generate text
    topic = random.choice(list(TOPIC_TEMPLATES.keys()))
    base_text = random.choice(TOPIC_TEMPLATES[topic])
    prefix = random.choice(PREFIXES)
    suffix = random.choice(SUFFIXES)
    text = f"{prefix}{base_text}{suffix}"
    
    # Generate timestamp (clustered around certain dates to create spikes)
    # Most posts within 90-day window, with spikes
    spike_dates = [
        base_date + timedelta(days=15),
        base_date + timedelta(days=35),
        base_date + timedelta(days=60),
        base_date + timedelta(days=75),
    ]
    
    if random.random() < 0.3:  # 30% chance of being on a spike date
        chosen_spike = random.choice(spike_dates)
        day_offset = random.gauss(0, 1)  # Tight around spike
        post_date = chosen_spike + timedelta(days=day_offset, hours=random.randint(0, 23))
    else:
        day_offset = random.uniform(0, 90)
        post_date = base_date + timedelta(days=day_offset, hours=random.randint(0, 23),
                                           minutes=random.randint(0, 59))
    
    # Score (follows power-law: most posts low score, few high)
    score = int(max(0, random.lognormvariate(2, 1.5)))
    
    # Number of comments
    num_comments = int(max(0, score * random.uniform(0.1, 0.5) + random.randint(0, 5)))
    
    return {
        "id": f"post_{post_id:06d}",
        "author": author,
        "subreddit": subreddit,
        "title": text,
        "body": "",
        "created_utc": int(post_date.timestamp()),
        "score": score,
        "num_comments": num_comments,
        "permalink": f"/r/{subreddit}/comments/post_{post_id:06d}/",
        "url": f"https://reddit.com/r/{subreddit}/comments/post_{post_id:06d}/",
    }


def main():
    """Generate sample dataset."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Base date: 90 days ago from a fixed reference
    base_date = datetime(2024, 9, 1)
    
    posts = []
    for i in range(NUM_POSTS):
        post = generate_post(i, base_date)
        posts.append(post)
    
    # Sort by timestamp
    posts.sort(key=lambda x: x['created_utc'])
    
    # Write JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for post in posts:
            f.write(json.dumps(post) + '\n')
    
    print(f"Generated {len(posts)} posts → {OUTPUT_FILE}")
    print(f"Date range: {datetime.fromtimestamp(posts[0]['created_utc'])} to {datetime.fromtimestamp(posts[-1]['created_utc'])}")
    print(f"Subreddits: {len(set(p['subreddit'] for p in posts))}")
    print(f"Authors: {len(set(p['author'] for p in posts))}")
    
    # Show distribution
    from collections import Counter
    sub_counts = Counter(p['subreddit'] for p in posts)
    print("\nSubreddit distribution:")
    for sub, count in sub_counts.most_common():
        print(f"  r/{sub}: {count} posts")


if __name__ == "__main__":
    main()
