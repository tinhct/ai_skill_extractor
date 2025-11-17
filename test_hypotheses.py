"""
AI Project Manager Skill Analysis Script

This script performs a quantitative analysis of AI Project Manager job postings
to test five key research hypotheses (H1-H5) and conduct one exploratory
analysis of skill clusters.

It reads data from the 'data/' directory and saves all processed results as CSV
files in the 'output/' directory, ready for visualization in Tableau.
"""

import json
import os
import pandas as pd
from collections import defaultdict

# --- File Paths ---
# Use os.path.join for cross-platform compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Input files
CAT_CODES_FILE = os.path.join(DATA_DIR, 'extracted_category_codes.json')
SKILLS_FILE = os.path.join(DATA_DIR, 'extracted_skills.json')
CLUSTERS_FILE = os.path.join(DATA_DIR, 'skill_clusters.csv')

# Output files
H1_OUTPUT = os.path.join(OUTPUT_DIR, 'h1_aggregate_competency.csv')
H2_OUTPUT = os.path.join(OUTPUT_DIR, 'h2_tech_dimensions.csv')
H3_TEST_OUTPUT = os.path.join(OUTPUT_DIR, 'h3_hybrid_test_result.csv')
H3_HEATMAP_OUTPUT = os.path.join(OUTPUT_DIR, 'h3_co_occurrence_heatmap.csv')
H4_H5_OUTPUT = os.path.join(OUTPUT_DIR, 'h4_h5_prevalence_results.csv')
EXPLORATORY_OUTPUT = os.path.join(OUTPUT_DIR, 'exploratory_skill_clusters.csv')


# --- Construct Definitions ---
# Based on the research design and JSON structure
CONSTRUCT_2_CATEGORIES = [
    'Project_management_skills',
    'leadership_and_soft_skills',
    'domain_and_business',
    'adaptability'
]

CONSTRUCT_3_CATEGORIES = [
    'conceptual_AI_knowledge',
    'hands-on_technical skills',
    'integration_and_deployment'
]

TECH_DIM_CODES = {
    'Conceptual': 'TECH-CONCEPT',
    'Hands-on': 'TECH-HANDSON',
    'Integration': 'TECH-OPS'
}

HYPOTHESIS_CODES = {
    'H4a (Advanced Degree)': {'EDU-ADV'},
    'H4b (Certification)': {'CERT-PM', 'CERT-TECH'},
    'H4 (Any Credential)': {'EDU-ADV', 'CERT-PM', 'CERT-TECH'},
    'H5a (Governance & Ethics)': {'CONTEXT-ETHIC', 'CONTEXT-REG'},
    'H5b (Complexity)': {'CONTEXT-COMPLEX'}
}


def load_data(file_path):
    """Loads a JSON file from the specified path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None

def process_h1(category_codes_data):
    """Calculates aggregate mentions for H1."""
    print("Processing H1...")
    core_mentions = 0
    technical_mentions = 0

    for categories in category_codes_data.values():
        for category_name, codes in categories.items():
            if category_name in CONSTRUCT_2_CATEGORIES:
                core_mentions += len(codes)
            elif category_name in CONSTRUCT_3_CATEGORIES:
                technical_mentions += len(codes)
    
    h1_df = pd.DataFrame([
        {'Competency': 'Core (Construct 2)', 'Total_Mentions': core_mentions},
        {'Competency': 'Technical (Construct 3)', 'Total_Mentions': technical_mentions}
    ])
    
    h1_df.to_csv(H1_OUTPUT, index=False)
    print(f"H1 results saved to {H1_OUTPUT}")

def process_h2(category_codes_data, total_jobs):
    """Calculates frequency and prevalence for H2."""
    print("Processing H2...")
    tech_mentions = {key: 0 for key in TECH_DIM_CODES}
    tech_prevalence = {key: 0 for key in TECH_DIM_CODES}

    for categories in category_codes_data.values():
        job_codes_flat = {code for code_list in categories.values() for code in code_list}
        
        for dim, code in TECH_DIM_CODES.items():
            if code in job_codes_flat:
                tech_prevalence[dim] += 1
            
            for code_list in categories.values():
                tech_mentions[dim] += code_list.count(code)

    h2_results = []
    for key, code in TECH_DIM_CODES.items():
        prevalence_pct = (tech_prevalence[key] / total_jobs) * 100
        h2_results.append({
            'Dimension': key,
            'Code': code,
            'Total_Mentions': tech_mentions[key],
            'Jobs_Prevalence': tech_prevalence[key],
            'Prevalence_Percent': f"{prevalence_pct:.1f}"
        })
    
    pd.DataFrame(h2_results).to_csv(H2_OUTPUT, index=False)
    print(f"H2 results saved to {H2_OUTPUT}")

def process_h3(category_codes_data, total_jobs):
    """Calculates hybrid profile test and co-occurrence matrix for H3."""
    print("Processing H3...")
    core_codes_interest = ['SS-STAKEHOLDER', 'SS-LEADERSHIP', 'PM-METHOD', 'PM-RISK', 'CORE-BUSINESS']
    tech_codes_interest = ['TECH-CONCEPT', 'TECH-HANDSON', 'TECH-OPS']
    
    heatmap_data = pd.DataFrame(0, index=core_codes_interest, columns=tech_codes_interest)
    hybrid_job_count = 0

    c2_codes_set = set()
    for cat_list in CONSTRUCT_2_CATEGORIES:
        # Assuming codes are in codebook, but for robustness, let's find all in data
        pass # We will check against the full list of codes in jobs
    # We need to find all unique codes in construct 2 and 3 from the data
    all_c2_codes = set()
    all_c3_codes = set()
    for job, categories in category_codes_data.items():
        for cat_name, codes in categories.items():
            if cat_name in CONSTRUCT_2_CATEGORIES:
                all_c2_codes.update(codes)
            elif cat_name in CONSTRUCT_3_CATEGORIES:
                all_c3_codes.update(codes)


    for categories in category_codes_data.values():
        job_codes_flat = {code for code_list in categories.values() for code in code_list}
        
        job_core_codes = job_codes_flat.intersection(all_c2_codes)
        job_tech_codes = job_codes_flat.intersection(all_c3_codes)
        
        # Test for H3 threshold
        if job_core_codes and job_tech_codes:
            hybrid_job_count += 1
            
        # Populate heatmap data
        job_core_heatmap = job_codes_flat.intersection(core_codes_interest)
        job_tech_heatmap = job_codes_flat.intersection(tech_codes_interest)

        for core_code in job_core_heatmap:
            for tech_code in job_tech_heatmap:
                heatmap_data.loc[core_code, tech_code] += 1

    # Save H3 threshold test result
    hybrid_pct = (hybrid_job_count / total_jobs) * 100
    h3_result_df = pd.DataFrame([
        {'Metric': 'Hybrid Job Count', 'Value': hybrid_job_count},
        {'Metric': 'Total Jobs', 'Value': total_jobs},
        {'Metric': 'Hybrid Prevalence (%)', 'Value': f"{hybrid_pct:.1f}"}
    ])
    h3_result_df.to_csv(H3_TEST_OUTPUT, index=False)
    print(f"H3 test results saved to {H3_TEST_OUTPUT}")

    # "Melt" the heatmap data for Tableau
    heatmap_long_df = heatmap_data.reset_index().melt(
        id_vars='index', 
        var_name='Tech_Competency', 
        value_name='Co_occurrence_Count'
    ).rename(columns={'index': 'Core_Competency'})
    
    heatmap_long_df.to_csv(H3_HEATMAP_OUTPUT, index=False)
    print(f"H3 heatmap data saved to {H3_HEATMAP_OUTPUT}")

def process_h4_h5(category_codes_data, total_jobs):
    """Calculates prevalence for H4 and H5."""
    print("Processing H4 & H5...")
    hyp_prevalence = {key: 0 for key in HYPOTHESIS_CODES}

    for categories in category_codes_data.values():
        job_codes_flat = {code for code_list in categories.values() for code in code_list}
            
        for key, code_set in HYPOTHESIS_CODES.items():
            if not job_codes_flat.isdisjoint(code_set):
                hyp_prevalence[key] += 1

    h4_h5_results = []
    for key, count in hyp_prevalence.items():
        prevalence_pct = (count / total_jobs) * 100
        h4_h5_results.append({
            'Hypothesis': key,
            'Jobs_Prevalence': count,
            'Prevalence_Percent': f"{prevalence_pct:.1f}"
        })

    pd.DataFrame(h4_h5_results).to_csv(H4_H5_OUTPUT, index=False)
    print(f"H4/H5 results saved to {H4_H5_OUTPUT}")

def process_exploratory(skills_data, total_jobs):
    """Processes exploratory skill cluster analysis."""
    print("Processing Exploratory Cluster Analysis...")
    try:
        skill_clusters_df = pd.read_csv(CLUSTERS_FILE)
    except FileNotFoundError:
        print(f"Error: File not found at {CLUSTERS_FILE}")
        return

    # Create a phrase-to-cluster lookup dictionary
    phrase_to_cluster = pd.Series(
        skill_clusters_df.cluster.values, 
        index=skill_clusters_df.phrase
    ).to_dict()

    cluster_mentions = defaultdict(int)
    job_cluster_pairs = set()

    for job_id, skill_categories in skills_data.items():
        if 'error' in skill_categories:
            continue
            
        for category, phrase_list in skill_categories.items():
            if not isinstance(phrase_list, list):
                continue # Skip if data is malformed
                
            for phrase in phrase_list:
                cluster = phrase_to_cluster.get(phrase)
                if cluster is not None:
                    cluster_mentions[cluster] += 1
                    job_cluster_pairs.add((job_id, cluster))

    cluster_prevalence_counts = defaultdict(int)
    for job_id, cluster in job_cluster_pairs:
        cluster_prevalence_counts[cluster] += 1

    cluster_results = []
    for cluster, mentions in cluster_mentions.items():
        prevalence_count = cluster_prevalence_counts.get(cluster, 0)
        prevalence_pct = (prevalence_count / total_jobs) * 100
        cluster_results.append({
            'Cluster_ID': cluster,
            'Total_Mentions': mentions,
            'Jobs_Prevalence': prevalence_count,
            'Prevalence_Percent': f"{prevalence_pct:.1f}"
        })

    cluster_results_df = pd.DataFrame(cluster_results).sort_values(
        by='Total_Mentions', ascending=False
    )
    
    cluster_results_df.to_csv(EXPLORATORY_OUTPUT, index=False)
    print(f"Exploratory cluster results saved to {EXPLORATORY_OUTPUT}")


def main():
    """Main function to run all analysis steps."""
    print("Starting AI Skill Extractor Analysis...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory ensured at: {OUTPUT_DIR}")

    # Load primary data
    category_codes_data = load_data(CAT_CODES_FILE)
    skills_data = load_data(SKILLS_FILE)
    
    if not category_codes_data or not skills_data:
        print("Error: Could not load critical data files. Exiting.")
        return

    total_jobs = len(category_codes_data)
    print(f"Total job postings loaded: {total_jobs}")

    # Run all processing steps
    process_h1(category_codes_data)
    process_h2(category_codes_data, total_jobs)
    process_h3(category_codes_data, total_jobs)
    process_h4_h5(category_codes_data, total_jobs)
    process_exploratory(skills_data, total_jobs)
    
    print("\nAll analyses complete. CSV files are available in the 'output' directory.")


if __name__ == "__main__":
    main()