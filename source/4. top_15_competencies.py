import pandas as pd
import json
import os
import sys

# Configuration
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
CODEBOOK_FILE = os.path.join(DATA_DIR, 'codebook.csv')
JSON_FILE = os.path.join(DATA_DIR, 'extracted_category_codes_and_values.json')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'top_15_competencies.csv')

# Target Constructs for "Competencies"
# We normalize to lowercase for comparison to be robust
TARGET_CONSTRUCTS = {
    "core competencies", 
    "technical competencies"
}

def load_codebook(filepath):
    """
    Loads the codebook and returns a dataframe of valid competency codes.
    Returns:
        pd.DataFrame: DataFrame indexed by 'sub_category_code' with columns 'construct', 'definition'.
        set: A set of valid sub_category_codes belonging to target constructs.
    """
    if not os.path.exists(filepath):
        print(f"Error: Codebook file not found at {filepath}")
        sys.exit(1)

    try:
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Normalize column names to be safe (strip whitespace, lowercase)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # specific mapping based on provided snippet headers:
        # construct, category, sub_category_code, definition
        
        # Normalize the 'construct' column for filtering
        if 'construct' not in df.columns:
            raise ValueError("Column 'construct' missing from codebook.")
            
        df['construct_norm'] = df['construct'].astype(str).str.strip().str.lower()
        
        # Filter for Core and Technical Competencies
        # We look for partial matches or exact matches in the target set
        # The prompt implies these are distinct constructs.
        valid_df = df[df['construct_norm'].isin(TARGET_CONSTRUCTS)].copy()
        
        if valid_df.empty:
            print("Warning: No codes found for 'Core Competencies' or 'Technical Competencies'. Checking partial matches...")
            # Fallback: check if 'competenc' is in the string (e.g. 'Core Competency')
            valid_df = df[df['construct_norm'].str.contains("competenc", na=False) & 
                         ~df['construct_norm'].str.contains("context|expectation|education", na=False)].copy()

        # Create a dictionary for definitions: code -> definition
        # We use the original code casing from the file but ensure keys are stripped
        code_map = valid_df.set_index('sub_category_code')[['definition']].to_dict()['definition']
        valid_codes = set(valid_df['sub_category_code'].dropna().astype(str).str.strip().unique())
        
        return code_map, valid_codes

    except Exception as e:
        print(f"Error reading codebook: {e}")
        sys.exit(1)

def load_extracted_codes(filepath):
    """
    Loads the nested JSON of extracted codes.
    """
    if not os.path.exists(filepath):
        print(f"Error: JSON file not found at {filepath}")
        sys.exit(1)
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)

def extract_ads_metrics(json_data, valid_codes_set):
    """
    Iterates through the JSON data. For each ad, finds the set of unique codes
    that exist in the valid_codes_set.
    
    Returns:
        dict: code -> count of ads containing this code
        int: total number of ads processed
    """
    code_counts = {code: 0 for code in valid_codes_set}
    total_ads = len(json_data)
    
    for job_id, job_data in json_data.items():
        # Collect all codes present in this ad's tree structure
        # Structure: Construct -> Category -> List of Codes
        present_codes = set()
        
        # We traverse the dictionary recursively or iteratively
        # Since we know the depth is likely 2 (Construct -> Category), we can use a nested loop
        # However, to be robust against slight schema variations, we'll traverse values.
        
        for construct_key, construct_val in job_data.items():
            # We assume the construct keys in JSON match the codebook constructs roughly, 
            # but we will just grab *all* codes found in lists and filter against valid_codes_set
            # to strictly adhere to the "valid constructs only" rule derived from the Codebook.
            
            if isinstance(construct_val, dict):
                for category_key, category_val in construct_val.items():
                    if isinstance(category_val, list):
                        # These are the codes
                        for code in category_val:
                            if code in valid_codes_set:
                                present_codes.add(code)
                                
        # Update counts
        for code in present_codes:
            code_counts[code] += 1
            
    return code_counts, total_ads

def main():
    print("Starting analysis of AI competencies...")
    
    # 1. Load Codebook and Definitions
    code_definition_map, valid_codes = load_codebook(CODEBOOK_FILE)
    print(f"Codebook loaded. Found {len(valid_codes)} valid competency codes.")
    
    # 2. Load Extracted Data
    json_data = load_extracted_codes(JSON_FILE)
    print(f"JSON loaded. Processing {len(json_data)} job advertisements.")
    
    # 3. Compute Frequencies
    counts, total_ads = extract_ads_metrics(json_data, valid_codes)
    
    if total_ads == 0:
        print("Error: No ads found in JSON.")
        sys.exit(1)

    # 4. Prepare DataFrame
    results = []
    for code, count in counts.items():
        if count > 0: # Optimization: skip 0s if desired, but requirements say "Top 10"
            pct = (count / total_ads) * 100
            definition = code_definition_map.get(code, "Definition not found")
            results.append({
                "Code": code,
                "Definition": definition,
                "n_ads": count,
                "pct_val": pct
            })
    
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("No matching competencies found in the job ads.")
        return

    # 5. Sort and Rank
    # Sort by Pct desc, then n_ads desc (redundant but safe), then Code asc
    df_results.sort_values(by=["pct_val", "n_ads", "Code"], 
                           ascending=[False, False, True], 
                           inplace=True)
    
    # Take top 15
    top_15 = df_results.head(15).copy()
    
    # Add Rank
    top_15.insert(0, 'Rank', range(1, 16))
    
    # Format Percentage
    top_15['Pct_of_ads_with_code'] = top_15['pct_val'].apply(lambda x: f"{x:.2f}%")

    # Select final columns
    final_output = top_15[['Rank', 'Code', 'Definition', 'Pct_of_ads_with_code']]

    # 6. Output to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_output.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSuccess! Top 15 competencies written to {OUTPUT_FILE}")
    print(final_output.to_string(index=False))

if __name__ == "__main__":
    main()