import sys
import os

# mock Streamlit
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
sys.modules['tensorflow'] = None
sys.modules['tensorflow.compat.v2.compiler.tensorrt'] = None

from data_loader import load_data, init_duckdb
from search_engine import load_embedding_model, compute_embeddings, build_faiss_index, semantic_search
from network_analysis import build_cooccurrence_network, compute_centrality_metrics, detect_communities

def run_debug():
    df = load_data()
    model = load_embedding_model()
    embeddings = compute_embeddings(model, df['text'].tolist())
    
    con = init_duckdb(df)
    index = build_faiss_index(embeddings)
    
    # Simulate a search for 'reddit'
    results, info = semantic_search('reddit', model, index, df, embeddings)
    print(f"Results: {len(results)}")
    
    analysis_df = results
    
    # Network section
    G = build_cooccurrence_network(analysis_df, groupby_col='subreddit', node_col='author')
    print(f"Graph nodes: {G.number_of_nodes()}")
    if G.number_of_nodes() > 0:
        centrality_df = compute_centrality_metrics(G)
        print("Centrality calculated.")
        
        partition = detect_communities(G)
        n_communities = len(set(partition.values())) if partition else 0
        from genai_summarizer import summarize_network
        print("Summarizing network...")
        summary = summarize_network(centrality_df, n_communities)
        print("Network summary successful.")
        
        # UI part where it might fail
        display_centrality = centrality_df.head(15).copy()
        display_centrality['pagerank'] = display_centrality['pagerank'].round(6)
        display_centrality['betweenness'] = display_centrality['betweenness'].round(6)
        display_centrality['influence_score'] = display_centrality['influence_score'].round(4)
        print("Done!")

if __name__ == "__main__":
    try:
        run_debug()
    except Exception as e:
        import traceback
        traceback.print_exc()
