import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# --- 1. DEFINE FILE PATHS ---
# Use os.path.join to work on any OS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "extracted_skills.json")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

# Define output file paths
EVAL_PLOT_FILE = os.path.join(ANALYSIS_DIR, "kmeans_evaluation.png")
CLUSTER_CSV_FILE = os.path.join(ANALYSIS_DIR, "skill_clusters.csv")


# --- 2. DATA LOADING AND PREPARATION ---

def load_and_prepare_data(filepath):
    """
    Loads the JSON file and extracts a unique set of all skills and phrases.
    """
    print(f"Loading and preparing data from {filepath}...")
    try:
        with open(filepath, 'r') as f:
            jobs_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        print("Please ensure 'extracted_skills.json' is in the 'data' folder.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}.")
        return None

    all_phrases = set()
    
    # Iterate through each job ID (e.g., "5dc67ee121c77e94")
    for job_id, data in jobs_data.items():
        # Iterate through each category (e.g., "qualifications", "role")
        for category, phrases in data.items():
            if isinstance(phrases, list):
                # Add all items from a list
                all_phrases.update(phrases)
            elif isinstance(phrases, str):
                # Add the single string (for fields like "job_domain")
                all_phrases.add(phrases)
                
    # Remove any potential empty strings
    all_phrases.discard("")
    
    print(f"Found {len(all_phrases)} unique phrases.")
    return list(all_phrases)

# --- 3. SEMANTIC EMBEDDING ---

def get_embeddings(phrases_list):
    """
    Generates semantic embeddings using the all-MiniLM-L6-v2 model.
    """
    print("Generating embeddings with 'all-MiniLM-L6-v2'...")
    # This model is a high-performance variant of MiniLM-L6
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings. This may take a few minutes.
    embeddings = model.encode(phrases_list, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

# --- 4. K-MEANS EVALUATION ---

def evaluate_kmeans(embeddings, k_range):
    """
    Runs K-Means for each k in k_range and stores inertia and silhouette scores.
    """
    print(f"Running K-Means evaluation for k={k_range[0]} to k={k_range[-1]}...")
    inertia_scores = []
    silhouette_scores = []
    
    for k in k_range:
        print(f"  Calculating for k={k}...")
        # n_init=10 runs k-means 10 times to find the best result
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, verbose=0)
        labels = kmeans.fit_predict(embeddings)
        
        # Store scores
        inertia_scores.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, labels))
        
    return inertia_scores, silhouette_scores

def plot_evaluation(k_range, inertia_scores, silhouette_scores, output_path):
    """
    Plots the Elbow Method (Inertia) and Silhouette Scores.
    Saves the plot to the specified output_path.
    """
    print(f"Plotting evaluation results to {output_path}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Elbow Method (Inertia)
    ax1.plot(k_range, inertia_scores, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')

    # Plot 2: Silhouette Score
    ax2.plot(k_range, silhouette_scores, 'go-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score for Optimal k')

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show() # Uncomment this line if you want the plot to pop up

# --- 5. CLUSTER ANALYSIS ---

def analyze_clusters(phrases_list, embeddings, optimal_k, output_path):
    """
    Runs K-Means with the optimal k and prints the contents of each cluster.
    Saves the full cluster mapping to the specified output_path.
    """
    print(f"\nAnalyzing clusters for optimal k={optimal_k}...")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Create a DataFrame to map phrases to their assigned cluster
    df = pd.DataFrame({'phrase': phrases_list, 'cluster': labels})
    
    # Sort by cluster for easy reading
    df = df.sort_values(by='cluster')
    
    # Print a sample of each cluster to the console
    for i in range(optimal_k):
        print(f"\n--- Cluster {i} ---")
        cluster_phrases = df[df['cluster'] == i]['phrase'].tolist()
        
        # Print up to 20 sample phrases per cluster
        print(", ".join(cluster_phrases[:20]))
        if len(cluster_phrases) > 20:
            print(f"... and {len(cluster_phrases) - 20} more.")
            
    # Save the full results to a file
    df.to_csv(output_path, index=False)
    print("\n---")
    print(f"✅ Full cluster analysis saved to {output_path}")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    
    # --- Configuration ---
    K_RANGE = range(5, 21) # k=5 to 20, as requested
    
    # Create the analysis directory if it doesn't exist
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    print(f"Ensured analysis directory exists: {ANALYSIS_DIR}")

    # Step 1: Load and Prepare Data
    unique_phrases = load_and_prepare_data(DATA_FILE)
    
    if unique_phrases:
        # Step 2: Get Embeddings
        phrase_embeddings = get_embeddings(unique_phrases)
        
        # Step 3: Evaluate K-Means
        inertia, silhouette = evaluate_kmeans(phrase_embeddings, K_RANGE)
        
        # Plot Evaluation Results
        plot_evaluation(K_RANGE, inertia, silhouette, EVAL_PLOT_FILE)
        
        # Step 4: Analyze Final Clusters
        # Automatically select the k with the highest silhouette score
        # This score is generally best for finding semantically distinct groups
        optimal_k = K_RANGE[np.argmax(silhouette)]
        print(f"\nOptimal k based on highest silhouette score: {optimal_k}")
        
        # You can also manually override this k if you prefer the elbow plot
        # optimal_k = 12 # Example: Manually setting k
        
        analyze_clusters(unique_phrases, phrase_embeddings, optimal_k, CLUSTER_CSV_FILE)