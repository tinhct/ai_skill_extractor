import pandas as pd
import networkx as nx
from itertools import combinations
import community as community_louvain
import json
import os

# --- Configuration: Define file paths ---

# Get the absolute path to the directory containing THIS script (analysis.py)
# This will be: /Users/tinhct/.../ai_skill_extractor
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the paths *from* the script's directory
data_path = os.path.join(SCRIPT_DIR, 'data', 'extracted_category_codes.json')
codebook_path = os.path.join(SCRIPT_DIR, 'data', 'codebook.xlsx')
output_path = os.path.join(SCRIPT_DIR, 'output')

def load_and_prepare_data(data_path, codebook_path):
    """Loads, flattens, and merges the primary data files."""
    print("Step 1: Loading and preparing data...")
    
    # Load category codes
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Flatten the JSON data into a long-format DataFrame
    records = []
    for job_id, categories in data.items():
        for category, codes in categories.items():
            if isinstance(codes, list):
                for code in codes:
                    records.append({
                        'job_id': job_id,
                        'category_raw': category,
                        'category_code': code
                    })
    df = pd.DataFrame(records)

    # Load and merge codebook
    codebook = pd.read_excel(codebook_path)
    df_merged = pd.merge(df, codebook, on='category_code', how='left')
    
    # Handle potential merge issues (codes not in codebook)
    df_merged['construct'].fillna('Unknown', inplace=True)
    df_merged['category'].fillna('Unknown', inplace=True)
    df_merged['definition'].fillna('No definition available', inplace=True)
    
    print(f"Data loaded successfully. Total records: {len(df_merged)}")
    return df_merged

def section1_analysis(df, output_path):
    """Performs frequency analysis and tests H1."""
    print("\nSection 1 Analysis: Anatomy of the Role (H1)")
    
    # Frequency analysis of all competencies
    freq_counts = df['category_code'].value_counts().reset_index()
    freq_counts.columns = ['category_code', 'frequency']
    
    # Merge with codebook to get definitions and constructs
    top_competencies = pd.merge(freq_counts, df[['category_code', 'definition', 'construct']].drop_duplicates(), on='category_code')
    
    # Get top 10 for H1 test
    top_10 = top_competencies.head(10)
    
    # Test H1
    core_competencies_count = top_10[top_10['construct'] == 'Core Competencies'].shape
    technical_competencies_count = top_10[top_10['construct'] == 'Technical Competencies'].shape
    
    print(f"H1 Test: Top 10 contains {core_competencies_count} Core Competencies and {technical_competencies_count} Technical Competencies.")
    if core_competencies_count > technical_competencies_count:
        print("H1 is SUPPORTED.")
    else:
        print("H1 is NOT SUPPORTED.")
        
    # Save top 15 for the report table
    top_competencies.head(15).to_csv(os.path.join(output_path, 'section1_top_competencies.csv'), index=False)
    print("Saved 'section1_top_competencies.csv'")

def section2_analysis(df, output_path):
    """Compares conceptual vs. hands-on technical skills and tests H2."""
    print("\nSection 2 Analysis: Technical Skillset (H2)")
    
    tech_df = df[df['construct'] == 'Technical Competencies']
    category_counts = tech_df['category'].value_counts().reset_index()
    category_counts.columns = ['technical_category', 'frequency']
    
    # Test H2
    conceptual_count = category_counts[category_counts['technical_category'] == 'Conceptual AI Knowledge']['frequency'].sum()
    handson_count = category_counts[category_counts['technical_category'] == 'Hands-on Technical Skills']['frequency'].sum()
    
    print(f"H2 Test: Conceptual Knowledge mentions = {conceptual_count}, Hands-on Skills mentions = {handson_count}.")
    if conceptual_count > handson_count:
        print("H2 is SUPPORTED.")
    else:
        print("H2 is NOT SUPPORTED.")
        
    category_counts.to_csv(os.path.join(output_path, 'section2_technical_competency_profile.csv'), index=False)
    print("Saved 'section2_technical_competency_profile.csv'")

def section3_analysis(df, output_path):
    """Calculates the prevalence of the hybrid profile and tests H3."""
    print("\nSection 3 Analysis: Hybrid Profile (H3)")
    
    job_constructs = df.groupby('job_id')['construct'].unique().apply(set)
    
    total_jobs = len(job_constructs)
    hybrid_jobs = 0
    
    for constructs in job_constructs:
        if 'Core Competencies' in constructs and 'Technical Competencies' in constructs:
            hybrid_jobs += 1
            
    hybrid_percentage = (hybrid_jobs / total_jobs) * 100 if total_jobs > 0 else 0
    
    print(f"H3 Test: {hybrid_jobs} out of {total_jobs} jobs are Hybrid ({hybrid_percentage:.2f}%).")
    if hybrid_percentage > 75:
        print("H3 is SUPPORTED.")
    else:
        print("H3 is NOT SUPPORTED.")
        
    result_df = pd.DataFrame({
        'profile_type': ['Hybrid', 'Non-Hybrid'],
        'count': [hybrid_jobs, total_jobs - hybrid_jobs],
        'percentage': [hybrid_percentage, 100 - hybrid_percentage]
    })
    result_df.to_csv(os.path.join(output_path, 'section3_hybrid_profile_prevalence.csv'), index=False)
    print("Saved 'section3_hybrid_profile_prevalence.csv'")

def section4_analysis(df, output_path):
    """Analyzes educational and certification requirements and tests H4."""
    print("\nSection 4 Analysis: Credentials (H4)")
    
    total_jobs = df['job_id'].nunique()
    
    # H4a: Advanced Degrees
    jobs_with_adv_degree = df[df['category'] == 'Advanced Academic Degrees']['job_id'].nunique()
    adv_degree_perc = (jobs_with_adv_degree / total_jobs) * 100
    print(f"H4a Test: {adv_degree_perc:.2f}% mention Advanced Degrees (>40% threshold).")
    print(f"H4a is {'SUPPORTED' if adv_degree_perc > 40 else 'NOT SUPPORTED'}.")

    # H4b: Certifications
    jobs_with_cert = df[df['category'] == 'Professional Certifications']['job_id'].nunique()
    cert_perc = (jobs_with_cert / total_jobs) * 100
    print(f"H4b Test: {cert_perc:.2f}% mention Certifications (>40% threshold).")
    print(f"H4b is {'SUPPORTED' if cert_perc > 40 else 'NOT SUPPORTED'}.")

    # H4: Any Advanced Credential
    jobs_with_any_credential = df[df['construct'] == 'Educational Attainment']['job_id'].nunique()
    any_credential_perc = (jobs_with_any_credential / total_jobs) * 100
    print(f"H4 Test: {any_credential_perc:.2f}% mention Any Advanced Credential (>60% threshold).")
    print(f"H4 is {'SUPPORTED' if any_credential_perc > 60 else 'NOT SUPPORTED'}.")
    
    result_df = pd.DataFrame({
        'credential_type': ['Advanced Academic Degrees', 'Professional Certifications', 'Any Advanced Credential'],
        'percentage': [adv_degree_perc, cert_perc, any_credential_perc]
    })
    result_df.to_csv(os.path.join(output_path, 'section4_credentials_prevalence.csv'), index=False)
    print("Saved 'section4_credentials_prevalence.csv'")

def section5_analysis(df, output_path):
    """Analyzes governance and complexity requirements and tests H5."""
    print("\nSection 5 Analysis: New Frontiers (H5)")
    
    total_jobs = df['job_id'].nunique()
    
    # H5a: Governance & Ethics
    jobs_with_gov_ethic = df[df['category_code'].isin(['CONTEXT-ETHIC', 'CONTEXT-REG'])]['job_id'].nunique()
    gov_ethic_perc = (jobs_with_gov_ethic / total_jobs) * 100
    print(f"H5a Test: {gov_ethic_perc:.2f}% mention Governance/Ethics (>33% threshold).")
    print(f"H5a is {'SUPPORTED' if gov_ethic_perc > 33 else 'NOT SUPPORTED'}.")

    # H5b: Complexity
    jobs_with_complexity = df[df['category_code'] == 'CONTEXT-COMPLEX']['job_id'].nunique()
    complexity_perc = (jobs_with_complexity / total_jobs) * 100
    print(f"H5b Test: {complexity_perc:.2f}% mention Complexity (>20% threshold).")
    print(f"H5b is {'SUPPORTED' if complexity_perc > 20 else 'NOT SUPPORTED'}.")
    
    result_df = pd.DataFrame({
        'requirement_type': ['Governance & Ethics', 'Complexity Management'],
        'percentage': [gov_ethic_perc, complexity_perc]
    })
    result_df.to_csv(os.path.join(output_path, 'section5_governance_complexity_prevalence.csv'), index=False)
    print("Saved 'section5_governance_complexity_prevalence.csv'")

def section6_analysis(df, output_path):
    """Performs network analysis of competency co-occurrence."""
    print("\nSection 6 Analysis: Network Analysis")
    
    # Create edge list from co-occurrences
    job_skills = df.groupby('job_id')['category_code'].apply(lambda x: sorted(list(set(x))))
    edge_list = {}
    for skills in job_skills:
        for pair in combinations(skills, 2):
            edge_list[pair] = edge_list.get(pair, 0) + 1
            
    edges_df = pd.DataFrame([{'source': k, 'target': k[1], 'weight': v} for k, v in edge_list.items()])
    
    # Build graph
    G = nx.from_pandas_edgelist(edges_df, 'source', 'target', ['weight'])
    
    # Calculate network metrics
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, weight='weight')
    partition = community_louvain.best_partition(G, weight='weight')
    
    # Create node list with attributes
    nodes_df = pd.DataFrame({
        'id': G.nodes(),
        'degree_centrality': [degree.get(n, 0) for n in G.nodes()],
        'betweenness_centrality': [betweenness.get(n, 0) for n in G.nodes()],
        'community': [partition.get(n, 0) for n in G.nodes()]
    })
    
    # Merge with codebook for definitions
    node_attributes = df[['category_code', 'definition', 'construct', 'category']].drop_duplicates().rename(columns={'category_code': 'id'})
    nodes_df = pd.merge(nodes_df, node_attributes, on='id', how='left')
    
    # Save to CSV
    nodes_df.to_csv(os.path.join(output_path, 'section6_network_nodes.csv'), index=False)
    edges_df.to_csv(os.path.join(output_path, 'section6_network_edges.csv'), index=False)
    print("Saved 'section6_network_nodes.csv' and 'section6_network_edges.csv'")

def main():
    """Main function to run the entire analysis pipeline."""
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    category_codes_path = os.path.join(data_dir, 'extracted_category_codes.json')
    codebook_path = os.path.join(data_dir, 'codebook.xlsx')

    # Run analyses
    df_merged = load_and_prepare_data(category_codes_path, codebook_path)
    section1_analysis(df_merged, output_dir)
    section2_analysis(df_merged, output_dir)
    section3_analysis(df_merged, output_dir)
    section4_analysis(df_merged, output_dir)
    section5_analysis(df_merged, output_dir)
    section6_analysis(df_merged, output_dir)
    
    print("\nAnalysis complete. All output files are in the 'output' directory.")

if __name__ == '__main__':
    main()