#!/usr/bin/env python3
"""
AI PM Job Ads Master Metrics Generator v1.0
Processes job ads and extracted competency codes to generate a comprehensive 
metrics CSV for hypothesis testing.
"""
import pandas as pd
import numpy as np
import json
import logging
import os
import sys
from pathlib import Path

# Setup logging and output directories
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=OUTPUT_DIR / 'master_metrics.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Metric Definitions Constants
CORE_PREFIXES = ("PM-", "SS-", "CORE-")
TECH_PREFIXES = ("TECH-", "DATA-")

CONCEPTUAL_CODES = {
    "TECH-CONCEPT", "TECH-GENAI-CONCEPT"
}
HANDSON_CODES = {
    "TECH-HANDSON", "TECH-GENAI-HANDSON"
}
GENAI_CODES = {"TECH-GENAI-CONCEPT", "TECH-GENAI-HANDSON"}
GOV_CONTEXT_CODES = {"CONT-GOV-SAFETY", "CONT-GOV-COM", "CONT-SECURITY"}
COMPLEXITY_CODES = {"CONT-COMPLEX", "CONT-SCALE"}

def flatten_codes(data, codes_set=None):
    """
    Recursively extract all unique code strings from the nested JSON structure 
    into a single set per job_id. Skips experience value keys.
    """
    if codes_set is None:
        codes_set = set()
        
    if isinstance(data, dict):
        for key, value in data.items():
            # Skip keys that hold raw values, not codes
            if key in ['exp_pm_years_value', 'exp_pm_years_type', 
                       'exp_ai_years_value', 'exp_ai_years_type']:
                continue
            flatten_codes(value, codes_set)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                codes_set.add(item.strip())
            elif isinstance(item, (dict, list)):
                flatten_codes(item, codes_set)
                
    return codes_set

def get_experience_values(job_data):
    """Extract experience years from the employer expectation section, preserving nulls."""
    emp_exp = job_data.get("employer expectation", {})
    # Default to None (null) if keys are missing
    exp_pm = emp_exp.get("exp_pm_years_value")
    exp_ai = emp_exp.get("exp_ai_years_value")
    return exp_pm, exp_ai

def process_metrics(df):
    """
    Apply metric logic to the combined DataFrame.
    """
    # Helper to calculate density safely
    def safe_div(num, denom):
        return round(num / denom, 4) if denom > 0 else 0.0

    # 1. Counts
    # using list comprehensions for speed over df.apply
    df['CoreCounts'] = df['codes'].apply(lambda x: sum(1 for c in x if c.startswith(CORE_PREFIXES)))
    df['TechCounts'] = df['codes'].apply(lambda x: sum(1 for c in x if c.startswith(TECH_PREFIXES)))
    df['Total_SkillCounts'] = df['CoreCounts'] + df['TechCounts']

    # 2. Densities
    df['CoreDensity'] = df.apply(lambda row: safe_div(row['CoreCounts'], row['Total_SkillCounts']), axis=1)
    df['TechDensity'] = df.apply(lambda row: safe_div(row['TechCounts'], row['Total_SkillCounts']), axis=1)
    
    # 3. LOHI (Log-Odds Hybridity Index) 
    # Formula: ln((TechCounts + 1) / (CoreCounts + 1))
    # Note: np.log is natural logarithm (ln)
    df['LOHI'] = np.log((df['TechCounts'] + 1) / (df['CoreCounts'] + 1))
    df['LOHI'] = df['LOHI'].round(4) # Maintain float precision rule

    # 4. Conceptual vs Hands-on
    df['ConceptualCounts'] = df['codes'].apply(lambda x: sum(1 for c in x if c in CONCEPTUAL_CODES))
    df['Hands-onCounts'] = df['codes'].apply(lambda x: sum(1 for c in x if c in HANDSON_CODES))
    
    df['Conceptual_Win'] = (df['ConceptualCounts'] > df['Hands-onCounts']).astype(int)
    df['Hands_on_Win'] = (df['Hands-onCounts'] > df['ConceptualCounts']).astype(int)

    # 5. Binary Flags (Degrees/Certs/Context/Experience/Advanced AI)
    df['Has_Advanced_Degree'] = df['codes'].apply(lambda x: 1 if "EDU-LEVEL-ADV" in x else 0)
    df['Has_Tech_Degree'] = df['codes'].apply(lambda x: 1 if ("EDU-FIELD-TECH" in x or "EDU-FIELD-AI" in x) else 0)
    df['Has_Biz_Degree'] = df['codes'].apply(lambda x: 1 if "EDU-FIELD-BIZ" in x else 0)
    df['Has_Method_Cert'] = df['codes'].apply(lambda x: 1 if "CERT-FIELD-PM" in x else 0)
    df['Has_Tech_Cert'] = df['codes'].apply(lambda x: 1 if "CERT-FIELD-TECH" in x else 0)
    
    df['Has_Gov_Context'] = df['codes'].apply(lambda x: 1 if not x.isdisjoint(GOV_CONTEXT_CODES) else 0)
    df['Has_Compliance'] = df['codes'].apply(lambda x: 1 if "CONT-GOV-COM" in x else 0)
    df['Has_Safety'] = df['codes'].apply(lambda x: 1 if "CONT-GOV-SAFETY" in x else 0)
    df['Has_Cybersecurity'] = df['codes'].apply(lambda x: 1 if "CONT-SECURITY" in x else 0)
    df['Has_Complexity_Context'] = df['codes'].apply(lambda x: 1 if not x.isdisjoint(COMPLEXITY_CODES) else 0)

    df['Software_Experience'] = df['codes'].apply(lambda x: 1 if "EXP-SOFTWARE" in x else 0)
    df['Advanced_AI'] = df['codes'].apply(lambda x: 1 if "TECH-AI-ADV" in x else 0)
    # 6. Roles
    # Strategic: STRAT and NOT EXEC
    df['Strategic_Role'] = df['codes'].apply(
        lambda x: 1 if ("ROLE-STRAT" in x and "ROLE-EXEC" not in x) else 0
    )
    # Execution: EXEC and NOT STRAT
    df['Execution_Role'] = df['codes'].apply(
        lambda x: 1 if ("ROLE-EXEC" in x and "ROLE-STRAT" not in x) else 0
    )
    # Hybrid: BOTH
    df['Hybrid_Role'] = df['codes'].apply(
        lambda x: 1 if ("ROLE-STRAT" in x and "ROLE-EXEC" in x) else 0
    )

    # 7. AI Specifics
    df['Generative_AI'] = df['codes'].apply(lambda x: 1 if not x.isdisjoint(GENAI_CODES) else 0)
    df['Agentic_AI'] = df['codes'].apply(lambda x: 1 if "TECH-AGENTS" in x else 0)

    return df

if __name__ == "__main__":
    print("Starting Master Metrics Generation...")
    
    # Paths
    csv_path = Path("data/job_ads.csv")
    json_path = Path("data/extracted_category_codes_and_values.json")
    
    # 1. Validation & Loading
    if not csv_path.exists():
        logging.error(f"Missing input file: {csv_path}")
        sys.exit(1)
    if not json_path.exists():
        logging.error(f"Missing input file: {json_path}")
        sys.exit(1)

    try:
        # Load CSV (force job_id to string to ensure matching)
        df_ads = pd.read_csv(csv_path, dtype={'job_id': str})
        logging.info(f"Loaded {len(df_ads)} rows from job_ads.csv")

        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        logging.info(f"Loaded JSON data for {len(json_data)} job_ids")

        # 2. Pre-process JSON into DataFrame format
        # This is more efficient than applying logic inside the join
        json_records = []
        for job_id, content in json_data.items():
            codes = flatten_codes(content)
            exp_pm, exp_ai = get_experience_values(content)
            json_records.append({
                'job_id': str(job_id),
                'codes': codes,
                'exp_pm_years_value': exp_pm,
                'exp_ai_years_value': exp_ai
            })
        
        df_json = pd.DataFrame(json_records)

        # 3. Merge
        # Left join to keep all ads, even if not in JSON (though ideally they should match)
        merged_df = pd.merge(df_ads, df_json, on='job_id', how='left')
        
        # Fill missing 'codes' with empty sets for ads not found in JSON
        # Fill missing experience with NaN (default behavior of merge is NaN, which is what we want)
        missing_json_mask = merged_df['codes'].isna()
        if missing_json_mask.any():
            count_missing = missing_json_mask.sum()
            logging.warning(f"{count_missing} job_ids from CSV were not found in JSON data. Metrics for these will be 0.")
            # Use map to assign empty sets to avoid SettingWithCopy warnings
            merged_df.loc[missing_json_mask, 'codes'] = merged_df.loc[missing_json_mask, 'codes'].apply(lambda x: set())

        # 4. Compute Metrics
        final_df = process_metrics(merged_df)

        # 5. Column Selection & Ordering
        required_columns = [
            'job_id', 'country', 'CoreCounts', 'TechCounts', 'Total_SkillCounts',
            'CoreDensity', 'TechDensity', 'ConceptualCounts', 'Hands-onCounts',
            'Conceptual_Win', 'Hands_on_Win', 'LOHI', 'Has_Advanced_Degree',
            'Has_Tech_Degree', 'Has_Biz_Degree', 'Has_Method_Cert', 'Has_Tech_Cert',
            'Has_Gov_Context', 'Has_Compliance', 'Has_Safety', 'Has_Cybersecurity', 'Has_Complexity_Context', 'exp_pm_years_value',
            'exp_ai_years_value', 'Software_Experience', 'Strategic_Role', 'Execution_Role', 'Hybrid_Role',
            'Generative_AI', 'Agentic_AI', 'Advanced_AI'
        ]

        # Filter and order
        output_df = final_df[required_columns]

        # 6. Save Output
        output_file = OUTPUT_DIR / "master_metrics.csv"
        output_df.to_csv(output_file, index=False, encoding='utf-8')
        
        logging.info(f"Successfully generated {output_file} with shape {output_df.shape}")
        print("✓ master_metrics.csv generated successfully")

    except Exception as e:
        logging.error(f"Critical Failure: {str(e)}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)