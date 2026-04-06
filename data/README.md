Place your Reddit JSONL data files in this directory.

Expected format: One JSON object per line with fields like:
- author
- subreddit
- body or title (or selftext)
- created_utc (Unix timestamp)
- score
- id
- permalink
- num_comments

The data loader will automatically detect and normalize the schema.

You can download the dataset from:
https://drive.google.com/drive/folders/13cYfPIV65j5AAh9GjuZR94sAx-7EFjnp?usp=sharing
