import pandas as pd
# Optional import for UMAP (umap-learn); avoid hard failure if it's not installed.
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    umap = None
    UMAP_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
# Optional import for interactive plotting; avoid hard failure if plotly is not installed.
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    PLOTLY_AVAILABLE = False
import os

# --- Configuration ---
INPUT_FILE = os.path.join('data', 'skill_clusters.csv')
OUTPUT_DIR = 'output'
STATIC_PLOT_FILE = os.path.join(OUTPUT_DIR, 'skill_clusters_static.png')
INTERACTIVE_PLOT_FILE = os.path.join(OUTPUT_DIR, 'skill_clusters_interactive.html')

# UMAP Parameters (you can tune these)
N_NEIGHBORS = 15  # Controls how UMAP balances local vs. global structure.
MIN_DIST = 0.1    # Controls how tightly UMAP packs points together.
N_COMPONENTS = 2  # We want a 2D plot.
METRIC = 'cosine' # Good for text data.
RANDOM_STATE = 42 # For reproducible results.

# TF-IDF Parameters
MAX_FEATURES = 5000 # Limit the number of features (words) to keep the matrix manageable.

def main():
    """
    Main function to load data, process it with TF-IDF and UMAP,
    and generate static and interactive visualizations.
    """
    print(f"Starting UMAP visualization process...")

    # --- 1. Load Data ---
    try:
        data = pd.read_csv(INPUT_FILE)
        print(f"Successfully loaded {len(data)} rows from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # --- 2. Prepare Data ---
    # Drop rows where 'phrase' is missing, as we can't vectorize them.
    data.dropna(subset=['phrase'], inplace=True)
    # Ensure cluster is treated as a categorical variable for plotting
    data['cluster'] = data['cluster'].astype(str)
    
    if data.empty:
        print("Error: No valid data left after dropping missing phrases.")
        return
        
    print(f"Processing {len(data)} rows with valid phrases.")
    phrases = data['phrase']
    clusters = data['cluster']

    # --- 3. Vectorize Text (TF-IDF) ---
    print("Vectorizing text data using TF-IDF...")
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=MAX_FEATURES)
        # .toarray() can be memory-intensive for large datasets, but UMAP often works with dense arrays.
        # If your dataset is very large, consider using sparse matrices (omit .toarray()) and ensure UMAP supports sparse input.
        X_tfidf = vectorizer.fit_transform(phrases).toarray()
    except Exception as e:
        print(f"Error during TF-IDF vectorization: {e}")
        return

    print(f"Running UMAP (n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST})...")
    if not UMAP_AVAILABLE:
        print("Error: 'umap' (umap-learn) is not installed; install it with: pip install umap-learn")
        return
    try:
        umap_model = umap.UMAP(
            n_neighbors=N_NEIGHBORS,
            min_dist=MIN_DIST,
            n_components=N_COMPONENTS,
            metric=METRIC,
            random_state=RANDOM_STATE
        )
        embedding = umap_model.fit_transform(X_tfidf)
        print(f"UMAP embedding created with shape: {embedding.shape}")
    except Exception as e:
        print(f"Error during UMAP processing: {e}")
        return

    # --- 5. Create Results DataFrame ---
    df_embedding = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
    # Reset index if rows were dropped, ensuring alignment
    data.reset_index(drop=True, inplace=True)
    df_result = pd.concat([df_embedding, data[['phrase', 'cluster']]], axis=1)
    df_result.rename(columns={'cluster': 'Cluster', 'phrase': 'Phrase'}, inplace=True)

    # --- 6. Create Visualizations ---
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Static Plot (Seaborn)
    print(f"Creating static plot at {STATIC_PLOT_FILE}...")
    try:
        plt.figure(figsize=(14, 10))
        # Use 'hue' for cluster and 'palette' for coloring
        num_clusters = len(df_result['Cluster'].unique())
        palette = sns.color_palette("hsv", n_colors=num_clusters)
        
        sns.scatterplot(
            data=df_result,
            x='UMAP_1',
            y='UMAP_2',
            hue='Cluster',
            palette=palette,
            s=50,      # Point size
            alpha=0.7  # Point transparency
        )
        
        plt.title('UMAP Projection of Skill Clusters (Static)', fontsize=16)
        plt.xlabel('UMAP Component 1', fontsize=12)
        plt.ylabel('UMAP Component 2', fontsize=12)
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.3)
        # Save and close the static plot
        plt.savefig(STATIC_PLOT_FILE, bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating static plot: {e}")

    # Interactive Plot (Plotly)
    print(f"Creating interactive plot at {INTERACTIVE_PLOT_FILE}...")
    if not PLOTLY_AVAILABLE:
        print("Plotly is not available; skipping interactive plot. To enable, install plotly: pip install plotly")
    else:
        try:
            fig = px.scatter(
                df_result,
                x='UMAP_1',
                y='UMAP_2',
                color='Cluster',      # Color points by cluster
                hover_data=['Phrase'], # Show the phrase on hover
                title='Interactive UMAP Projection of Skill Clusters'
            )
            
            fig.update_layout(
                legend_title_text='Cluster',
                xaxis_title='UMAP Component 1',
                yaxis_title='UMAP Component 2'
            )
            fig.update_traces(marker=dict(size=8, opacity=0.8))
            
            fig.write_html(INTERACTIVE_PLOT_FILE)
        except Exception as e:
            print(f"Error creating interactive plot: {e}")

    print("\nProcess finished.")
    print(f"Static plot saved to: {STATIC_PLOT_FILE}")
    print(f"Interactive plot saved to: {INTERACTIVE_PLOT_FILE}")

if __name__ == "__main__":
    main()