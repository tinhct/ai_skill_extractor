import json
import pandas as pd
from pathlib import Path
import sys
import math

# Constants for file paths
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
INPUT_CODEBOOK = DATA_DIR / "codebook.csv"
INPUT_JSON = DATA_DIR / "extracted_category_codes_and_values.json"
OUTPUT_CSV = OUTPUT_DIR / "densities_Core_vs_Technical.csv"

def load_json_data(path: Path) -> dict:
    """Loads the raw extracted job ads data."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_codebook(path: Path) -> pd.DataFrame:
    """Loads the codebook to ensure all schema definitions are captured."""
    if not path.exists():
        print(f"Warning: Codebook not found at {path}. Proceeding with JSON data only.")
        return pd.DataFrame()
    return pd.read_csv(path)

def adjust_percentages_to_100(raw_percentages):
    """
    Adjusts percentages to ensure they sum to exactly 100.0% using largest remainder method.
    
    Args:
        raw_percentages: List of exact percentage values (floats)
    
    Returns:
        List of adjusted percentages that sum to exactly 100.0%
    """
    # Step 1: Calculate what each percentage would be if rounded normally
    rounded_values = [round(p, 1) for p in raw_percentages]
    
    # Step 2: Check if the sum is already 100.0
    current_sum = sum(rounded_values)
    
    if abs(current_sum - 100.0) < 0.001:  # Already close enough
        return rounded_values
    
    # Step 3: Calculate the difference and determine how many values need adjustment
    difference = round(100.0 - current_sum, 1)
    adjustment_needed = abs(difference)
    adjustment_direction = 0.1 if difference > 0 else -0.1
    num_adjustments = int(round(adjustment_needed / 0.1))
    
    # Step 4: Use largest remainder method to decide which values to adjust
    remainders = []
    for i, raw_val in enumerate(raw_percentages):
        # Calculate the remainder when rounding
        scaled = raw_val * 10
        remainder = scaled - math.floor(scaled)
        remainders.append((remainder, i, raw_val))
    
    # Sort by remainder descending, then by original value descending for tie-breaking
    remainders.sort(key=lambda x: (-x[0], -x[2]))
    
    # Step 5: Apply adjustments to the values with largest remainders
    adjusted_values = rounded_values.copy()
    
    for i in range(min(num_adjustments, len(adjusted_values))):
        idx = remainders[i][1]
        adjusted_values[idx] += adjustment_direction
        adjusted_values[idx] = round(adjusted_values[idx], 1)
    
    return adjusted_values

def format_sub_category(name: str) -> str:
    """
    Formats sub-category keys to Title Case for display.
    Handles specific acronyms like 'AI' to ensure correct capitalization.
    """
    formatted = name.replace('_', ' ').title()
    
    words = formatted.split()
    fixed_words = [w.replace("Ai", "AI") if w == "Ai" or w.startswith("Ai ") else w for w in words]
    
    final_str = " ".join(fixed_words)
    final_str = final_str.replace("Agentic Ai", "Agentic AI")
    final_str = final_str.replace("Conceptual Ai", "Conceptual AI")
    final_str = final_str.replace("Generative Ai", "Generative AI")
    
    return final_str

def compute_counts(ads: dict) -> tuple:
    """
    Aggregates mention counts for Core and Technical competencies across all ads.
    Returns a DataFrame with columns: [Category, Sub-Category, Raw_Count, Mentions_Per_Ad]
    """
    target_categories = {
        "core competencies": "Core Competencies",
        "technical competencies": "Technical Competencies"
    }

    agg_data = {}
    known_structure = {v: set() for v in target_categories.values()}
    total_ads = len(ads)

    for ad_id, content in ads.items():
        # Handle nested structure properly - content might have nested job data
        actual_content = content
        if isinstance(content, dict) and len(content) == 1:
            # If content has one key that contains the actual job data
            first_key = list(content.keys())[0]
            if isinstance(content[first_key], dict):
                actual_content = content[first_key]
        
        for json_key, display_category in target_categories.items():
            cat_block = actual_content.get(json_key, {})
            
            if isinstance(cat_block, dict):
                for sub_cat_key, codes_list in cat_block.items():
                    sub_cat_fmt = sub_cat_key
                    known_structure[display_category].add(sub_cat_fmt)
                    count = len(codes_list) if isinstance(codes_list, list) else 0
                    
                    key = (display_category, sub_cat_fmt)
                    agg_data[key] = agg_data.get(key, 0) + count

    rows = []
    grand_total_mentions = sum(agg_data.values())
    
    print(f"Total mentions found: {grand_total_mentions}")
    print(f"Processing {total_ads} job advertisements")

    # ENHANCED: Create all rows first, then sort within categories
    ordered_cats = ["Core Competencies", "Technical Competencies"]
    
    for cat in ordered_cats:
        sub_cats = list(known_structure[cat])
        
        # Create temporary list for this category to sort by count
        temp_rows = []
        for sub_cat_raw in sub_cats:
            key = (cat, sub_cat_raw)
            total_count = agg_data.get(key, 0)
            mean_mentions = total_count / total_ads if total_ads > 0 else 0.0
            
            temp_rows.append({
                "Category": cat,
                "Sub-Category": format_sub_category(sub_cat_raw),
                "Raw Count": total_count,
                "Mean Mentions per Ad": mean_mentions
            })
        
        # Sort this category's rows by Raw Count (descending) to prepare for percentage sorting
        temp_rows.sort(key=lambda x: x["Raw Count"], reverse=True)
        
        # Add sorted rows to main list
        rows.extend(temp_rows)

    df = pd.DataFrame(rows)
    return df, grand_total_mentions

def calculate_shares_and_format(df: pd.DataFrame, grand_total: int) -> pd.DataFrame:
    """
    Calculates percentage shares and formats the final table structure.
    ENHANCED: Ensures percentages sum to exactly 100.0% and sorts by Share of All Mentions (%)
    """
    if grand_total == 0:
        df["Share of All Mentions (%)"] = 0.0
        df["Category Total (%)"] = 0.0
        return df

    # Calculate raw percentages (exact values before rounding)
    raw_percentages = (df["Raw Count"] / grand_total * 100).tolist()
    
    # Apply adjustment to ensure sum equals 100.0%
    adjusted_percentages = adjust_percentages_to_100(raw_percentages)
    
    # Assign the adjusted percentages
    df["Share of All Mentions (%)"] = adjusted_percentages

    # ENHANCED: Sort within each category by Share of All Mentions (%) descending
    ordered_cats = ["Core Competencies", "Technical Competencies"]
    final_sorted_rows = []
    
    for cat in ordered_cats:
        # Get rows for this category
        cat_rows = df[df["Category"] == cat].copy()
        
        # Sort by Share of All Mentions (%) descending
        cat_rows = cat_rows.sort_values("Share of All Mentions (%)", ascending=False)
        
        # Calculate category total from sorted data
        cat_total = cat_rows["Share of All Mentions (%)"].sum()
        
        # Add Category Total (%) to first row of each category, empty for others
        cat_rows["Category Total (%)"] = ""
        cat_rows.iloc[0, cat_rows.columns.get_loc("Category Total (%)")] = round(cat_total, 1)
        
        # Add to final results
        final_sorted_rows.append(cat_rows)
    
    # Combine all sorted categories
    final_df = pd.concat(final_sorted_rows, ignore_index=True)

    # Round Mean Mentions per Ad
    final_df["Mean Mentions per Ad"] = final_df["Mean Mentions per Ad"].round(1)

    # Verification
    total_percentage = final_df["Share of All Mentions (%)"].sum()
    print(f"\nVerification:")
    print(f"Total Share of All Mentions: {total_percentage:.1f}% (target: 100.0%)")
    
    if abs(total_percentage - 100.0) > 0.001:
        print(f"⚠️  WARNING: Total does not equal 100.0%!")
    else:
        print("✅ Success: Percentages sum to exactly 100.0%")
    
    # Print category breakdowns (now sorted)
    for cat in ordered_cats:
        cat_rows = final_df[final_df["Category"] == cat]
        cat_sum = cat_rows["Share of All Mentions (%)"].sum()
        print(f"{cat}: {cat_sum:.1f}%")
        
        # Show top 3 subcategories for each category
        top_3 = cat_rows.head(3)
        for _, row in top_3.iterrows():
            print(f"  - {row['Sub-Category']}: {row['Share of All Mentions (%)']:.1f}%")

    # Select final columns
    final_cols = [
        "Category", 
        "Sub-Category", 
        "Mean Mentions per Ad", 
        "Share of All Mentions (%)", 
        "Category Total (%)"
    ]
    
    return final_df[final_cols]

def main():
    print("Starting analysis...")
    
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    try:
        _ = load_codebook(INPUT_CODEBOOK) 
        ads_data = load_json_data(INPUT_JSON)
        print(f"Successfully loaded {len(ads_data)} job advertisements.")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Process Data
    df_counts, grand_total = compute_counts(ads_data)
    
    if grand_total == 0:
        print("Warning: No mentions found in data!")
        return
    
    # Finalize with corrected percentages and sorting
    final_df = calculate_shares_and_format(df_counts, grand_total)
    
    # Display results
    print("\nFinal Results (sorted by Share of All Mentions % within each category):")
    print(final_df.to_string(index=False))
    
    # Output
    try:
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nAnalysis complete. Output saved to: {OUTPUT_CSV}")
        print("Note: Within each category, sub-categories are sorted from highest to lowest share percentage.")
    except Exception as e:
        print(f"Error saving output CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()