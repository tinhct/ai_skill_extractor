import json
import pandas as pd
from collections import defaultdict
import os # Make sure os is imported at the top of your file

# --- Configuration: Define file paths ---

# Get the absolute path to the directory containing THIS script (analysis.py)
# This will be: /Users/tinhct/.../ai_skill_extractor
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the paths *from* the script's directory
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output') 

# These variables will now hold the correct, full, absolute paths
# e.g., /Users/tinhct/.../ai_skill_extractor/data/extracted_category_codes.json
CODES_FILE = os.path.join(DATA_DIR, 'extracted_category_codes.json')
CODEBOOK_FILE = os.path.join(DATA_DIR, 'codebook.csv')

# --- Helper Function: To load the codebook ---
def load_codebook():
    """
    Loads the codebook.csv and creates a dictionary mapping
    category_codes (e.g., 'PM-PLAN') to their main construct 
    (e.g., 'Core Competencies').
    """
    codebook_df = pd.read_csv(CODEBOOK_FILE)
    
    # Create a mapping of code -> construct
    # We use this to check if a code is 'Core', 'Technical', etc.
    code_to_construct = {}
    for _, row in codebook_df.iterrows():
        code_to_construct[row['category_code']] = row['construct']
    
    # Define our construct groups based on the codebook
    constructs = {
        'Core': [c for c, const in code_to_construct.items() if const == 'Core Competencies'],
        'Technical': [c for c, const in code_to_construct.items() if const == 'Technical Competencies'],
        'Education_Advanced': ['EDU-ADV'],
        'Education_Cert': ['CERT-PM', 'CERT-TECH'],
        'Context_Ethics': ['CONTEXT-ETHIC', 'CONTEXT-REG'], # As per H5a
        'Context_Complexity': ['CONTEXT-COMPLEX'] # As per H5b
    }
    
    return code_to_construct, constructs

# --- Helper Function: To load the job data ---
def load_job_data():
    """
    Loads the main extracted_category_codes.json file.
    """
    with open(CODES_FILE, 'r') as f:
        data = json.load(f)
    return data

# --- Main Analysis Function ---
def analyze_data():
    """
    This is the main function that runs all analyses for H1-H5.
    """
    # Load our data and codebook
    job_data = load_job_data()
    code_to_construct, constructs = load_codebook()
    
    total_jobs = len(job_data)
    
    # --- Data Structures for Analysis ---
    # For H1: Count frequency of *every individual code*
    code_frequencies = defaultdict(int)
    
    # For H2: Sum of all conceptual vs. all applied codes
    h2_counts = {
        'Conceptual (TECH-CONCEPT)': 0,
        'Applied (TECH-HANDSON)': 0
    }
    
    # For H3, H4, H5: Count *jobs* that match criteria
    job_counters = {
        'H3_Hybrid': 0,
        'H4a_Advanced_Degree': 0,
        'H4b_Certification': 0,
        'H4_Main_Either': 0,
        'H5a_Ethics_Governance': 0,
        'H5b_Complexity': 0
    }
    
    # --- Loop Through Every Job (The Main Analysis) ---
    for job_id, categories in job_data.items():
        
        # --- Create a single flat list of all codes for this job ---
        all_codes_for_this_job = []
        for category_name, code_list in categories.items():
            if isinstance(code_list, list):
                all_codes_for_this_job.extend(code_list)
        
        # --- H1 & H2 Analysis (Frequency Counting) ---
        # Flags for H2
        job_has_conceptual = False
        job_has_applied = False
        
        for code in all_codes_for_this_job:
            # H1: Increment frequency for every code
            code_frequencies[code] += 1
            
            # H2: Check if this code is one of the ones we're tracking
            if code == 'TECH-CONCEPT':
                h2_counts['Conceptual (TECH-CONCEPT)'] += 1
            elif code == 'TECH-HANDSON':
                h2_counts['Applied (TECH-HANDSON)'] += 1

        # --- H3, H4, H5 Analysis (Job-level Counting) ---
        
        # H3: Check for Hybrid Profile
        has_core = any(code in constructs['Core'] for code in all_codes_for_this_job)
        has_technical = any(code in constructs['Technical'] for code in all_codes_for_this_job)
        if has_core and has_technical:
            job_counters['H3_Hybrid'] += 1
            
        # H4: Check for Education
        has_adv_degree = any(code in constructs['Education_Advanced'] for code in all_codes_for_this_job)
        has_cert = any(code in constructs['Education_Cert'] for code in all_codes_for_this_job)
        
        if has_adv_degree:
            job_counters['H4a_Advanced_Degree'] += 1
        if has_cert:
            job_counters['H4b_Certification'] += 1
        if has_adv_degree or has_cert:
            job_counters['H4_Main_Either'] += 1

        # H5: Check for Context
        if any(code in constructs['Context_Ethics'] for code in all_codes_for_this_job):
            job_counters['H5a_Ethics_Governance'] += 1
        if any(code in constructs['Context_Complexity'] for code in all_codes_for_this_job):
            job_counters['H5b_Complexity'] += 1

    # --- Process and Print Results ---
    print("--- 📊 AI Project Manager Skills Analysis Results ---")
    print(f"Total Job Advertisements Analyzed: {total_jobs}\n")

    # --- H1 Results ---
    print("--- H1: Primacy of Core Competencies ---")
    print("Top 10 Most Frequent Skill Categories:")
    h1_df = pd.DataFrame(code_frequencies.items(), columns=['Code', 'Frequency'])
    h1_df = h1_df.sort_values(by='Frequency', ascending=False)
    # Map construct name for easy reading
    h1_df['construct'] = h1_df['Code'].map(lambda c: code_to_construct.get(c, 'Other'))
    top_10_df = h1_df.head(10)
    print(top_10_df.to_string()) # .to_string() prints it nicely
    
    # Get the counts for validation
    core_count = top_10_df[top_10_df['construct'] == 'Core Competencies'].shape[0]
    tech_count = top_10_df[top_10_df['construct'] == 'Technical Competencies'].shape[0]
    print(f"\nValidation: In the Top 10, {core_count} are Core, {tech_count} are Technical.")
    print(f"H1 Supported: {core_count > tech_count}\n")
    # Save for Tableau
    top_10_df.to_csv(os.path.join(OUTPUT_DIR, 'h1_top_10_codes.csv'), index=False)

    # --- H2 Results ---
    print("--- H2: Conceptual vs. Applied Technical Knowledge ---")
    print("Total Frequency of All Mentions:")
    h2_df = pd.DataFrame(h2_counts.items(), columns=['Category', 'Total_Frequency'])
    print(h2_df.to_string())
    h2_supported = h2_counts['Conceptual (TECH-CONCEPT)'] > h2_counts['Applied (TECH-HANDSON)']
    print(f"H2 Supported: {h2_supported}\n")
    # Save for Tableau
    h2_df.to_csv(os.path.join(OUTPUT_DIR, 'h2_technical_comparison.csv'), index=False)

    # --- H3, H4, H5 Results (Percentages) ---
    print("--- H3, H4, H5: Profile & Context Percentages ---")
    
    # Calculate percentages
    results = {
        'H3: Hybrid Profile (>75%)': (job_counters['H3_Hybrid'] / total_jobs, 75),
        'H4a: Advanced Degree (>40%)': (job_counters['H4a_Advanced_Degree'] / total_jobs, 40),
        'H4b: Certification (>40%)': (job_counters['H4b_Certification'] / total_jobs, 40),
        'H4 Main: Degree or Cert (>60%)': (job_counters['H4_Main_Either'] / total_jobs, 60),
        'H5a: Ethics/Governance (>33%)': (job_counters['H5a_Ethics_Governance'] / total_jobs, 33),
        'H5b: Complexity (>20%)': (job_counters['H5b_Complexity'] / total_jobs, 20)
    }
    
    # Format for printing and saving
    percentage_results = []
    for name, (result_ratio, target_perc) in results.items():
        result_perc = round(result_ratio * 100, 2)
        is_supported = result_perc > target_perc
        print(f"{name}: {result_perc}% (Target: >{target_perc}%) -> Supported: {is_supported}")
        percentage_results.append({
            'Hypothesis': name,
            'Result_Percentage': result_perc,
            'Target_Percentage': target_perc,
            'Supported': is_supported
        })
    
    # Save for Tableau
    perc_df = pd.DataFrame(percentage_results)
    perc_df.to_csv(os.path.join(OUTPUT_DIR, 'h3_h4_h5_results.csv'), index=False)
    
    # Create separate files for easier visualization
    h3_data = {
        'Profile_Type': ['Hybrid', 'Non-Hybrid'],
        'Job_Count': [job_counters['H3_Hybrid'], total_jobs - job_counters['H3_Hybrid']]
    }
    pd.DataFrame(h3_data).to_csv(os.path.join(OUTPUT_DIR, 'h3_hybrid_profile.csv'), index=False)

    h4_data = {
        'Credential_Type': ['H4a: Advanced Degree', 'H4b: Certification', 'H4 Main: Degree or Cert'],
        'Percentage_of_Jobs': [round(results['H4a: Advanced Degree (>40%)'][0] * 100, 2),
                               round(results['H4b: Certification (>40%)'][0] * 100, 2),
                               round(results['H4 Main: Degree or Cert (>60%)'][0] * 100, 2)]
    }
    pd.DataFrame(h4_data).to_csv(os.path.join(OUTPUT_DIR, 'h4_education.csv'), index=False)
    
    h5_data = {
        'Context_Type': ['H5a: Ethics/Governance', 'H5b: Complexity'],
        'Percentage_of_Jobs': [round(results['H5a: Ethics/Governance (>33%)'][0] * 100, 2),
                               round(results['H5b: Complexity (>20%)'][0] * 100, 2)]
    }
    pd.DataFrame(h5_data).to_csv(os.path.join(OUTPUT_DIR, 'h5_context.csv'), index=False)
    
    print("\n--- ✅ Analysis Complete ---")
    print(f"All output files saved to the '{OUTPUT_DIR}' folder.")


# --- This makes the script runnable ---
if __name__ == "__main__":
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    analyze_data()