import os
import json
import csv
import time
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

# --- CONFIGURATION ---

# Load environment variables (your API key) from the .env file
load_dotenv()

# Define the paths to your files
# Path.cwd() gets the current working directory (your 'ai_skill_extractor' folder)
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
CODEBOOK_FILE = DATA_DIR / "codebook.csv"
SKILLS_FILE = DATA_DIR / "extracted_skills.json"
OUTPUT_FILE = DATA_DIR / "extracted_category_codes.json"

# Set the name of the Gemini model we'll use
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

# --- HELPER FUNCTIONS ---

def load_codebook_as_markdown():
    """
    Reads the codebook.csv and formats it as a Markdown table.
    This is much easier for the LLM to read and understand than raw CSV.
    """
    print("Loading codebook.csv...")
    codebook_markdown = "| construct | category | category_code | definition | dictionaries |\n"
    codebook_markdown += "|---|---|---|---|---|\n"
    try:
        with open(CODEBOOK_FILE, mode='r', encoding='utf-8') as f:
            # Skip the header row
            reader = csv.reader(f)
            next(reader) 
            # Add each row to our markdown table
            for row in reader:
                codebook_markdown += f"| {' | '.join(row)} |\n"
        print("Codebook loaded and formatted as Markdown.")
        return codebook_markdown
    except FileNotFoundError:
        print(f"ERROR: codebook.csv not found at {CODEBOOK_FILE}")
        return None
    except Exception as e:
        print(f"ERROR reading codebook.csv: {e}")
        return None

def load_skills_data():
    """
    Loads the extracted_skills.json file into a Python dictionary.
    """
    print("Loading extracted_skills.json...")
    try:
        with open(SKILLS_FILE, mode='r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Found {len(data)} jobs to process.")
            return data
    except FileNotFoundError:
        print(f"ERROR: extracted_skills.json not found at {SKILLS_FILE}")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: extracted_skills.json is not a valid JSON file.")
        return None

def build_system_prompt(codebook_md):
    """
    Creates the main system instruction prompt using the prompt you provided.
    It inserts the codebook directly into the prompt.
    """
    # This is the detailed prompt you wrote, slightly tweaked for this script
    return f"""
# Step 1: Examine the raw data
You are an expert data analysis with specialization in analyzing technical job postings.
I'm going to provide you with two raw data:
(1) a JSON file containing a list of a job's extracted skills, qualifications, and requirements.
(2) a CSV file (formatted as Markdown) of the codebook that defines constructs, categories, category_code, definition, and dictionaries.

Your task is to carefully analyze the extracted text from the JSON file to extract structured category_code respectively from the codebook for the job.

Here is the codebook:
{codebook_md}

# Context Specification
The input JSON file is the result of the extracted skills, qualifications, and requirements of the job posting.

# Step 2: Extract and categorize category code
Follow this step-by-step process:
1. Read the entire extracted JSON for the single job carefully.
2. Compare the text in the JSON against the dictionaries for each category described in the codebook.
3. Categorize each identified item into the appropriate category_code. If a category from the JSON is empty (e.g., "qualifications": []), you must also return an empty list for it.

Here are the categories:
-- QUAL-PREF: Preferred or required educational degree from employers.
-- EXP-PM-YRS: Specific number of years of project management experience required.
-- EXP-AI-YRS: Specific number of years of AI/ML project experience required.
-- EXP-INDUSTRY: Requirement for experience in a specific technical skill or role or industry.
-- EXP_ROLE: Explicitly stated responsibilities and duties of the position.
-- PM-METHOD: Mention of Agile, Scrum, Kanban, Waterfall, or related methodologies.
-- PM-RISK: Mention of risk identification, assessment, or mitigation.
-- PM-PLAN: Mention of project planning, scheduling, or resource management.
-- PM-BUDGET: Mention of budget management, financial oversight, or cost control.
-- PM-TOOL: Mention of proficiency or experience with specific project management tools.
-- SS-STAKEHOLDER: Mention of stakeholder communication, management, or engagement.
-- SS-LEADERSHIP: Mention of team leadership, collaboration, or mentorship.
-- SS-CONFLICT: Mention of conflict resolution or negotiation skills.
-- CORE-BUSINESS: Mention of business acumen, strategy, or commercial awareness.
-- CORE-ADAPT: Mention of adaptability, continuous learning, or thriving in a fast-paced environment.
-- TECH-CONCEPT: Mention of understanding or familiarity with AI/ML, data, math, ai product lifecycle concepts.
-- TECH-HANDSON: Mention of proficiency or experience with specific languages, platforms or data tools.
-- TECH-OPS: Mention of technical operations such as MLOps, CI/CD for AI, model deployment, or production systems.
-- EDU-ADV: Formal postgraduate degrees or advanced educational attainment such as Master's or Doctorate required or preferred by the employer.
-- CERT-PM: Mention of Project Management/Agile methodology certification.
-- CERT-TECH: Mention of technical certifications (AI, Cloud).
-- CONTEXT-ETHICS: Mention of ethical AI, responsible AI, bias, fairness, or transparency.
-- CONTEXT-REG: Mention of specific regulations.
-- CONTEXT-COMPLEX: Mention of managing complex, ambiguous, or innovative projects.
-- other: Any requirements or expectations that don’t clearly fit the above codes.

# Output Format
You will be given the data for a *single job*. Your task is to process *only that job*.
Respond ONLY with a valid JSON object for that single job, using this exact structure (do NOT include a top-level job_id key):
{{
    "qualifications": ["category_code1", "category_code2", ...],
    "experience": ["category_code1", "category_code2", ...],
    "role": ["category_code1", "category_code2", ...],
    "Project_management_skills": ["category_code1", "category_code2", ...],
    "leadership_and_soft_skills": ["category_code1", "category_code2", ...],
    "domain_and_business": ["category_code1", "category_code2", ...],
    "adaptability": ["category_code1", "category_code2", ...],
    "conceptual_AI_knowledge": ["category_code1", "category_code2", ...],
    "hands-on_technical skills": ["category_code1", "category_code2", ...],
    "integration_and_deployment": ["category_code1", "category_code2", ...],
    "advanced_academic_degrees": ["category_code1", "category_code2", ...],
    "professional_certifications": ["category_code1", "category_code2", ...],
    "ethical_and_regulatory": ["category_code1", "category_code2", ...],
    "project_complexity": ["category_code1", "category_code2", ...],
    "other": ["item1", "item2", ...]
}}

Important guidelines:
-- Separate distinct category (don’t combine multiple categories into one entry)
-- The output must be valid JSON (double quotes, commas between items).
-- Contain ONLY the JSON object, no additional text, no markdown backticks.
-- If a source category is empty in the input JSON, the corresponding output list must also be empty.
"""

def call_gemini_api(model, system_prompt, job_data_str):
    """
    Calls the Gemini API with the system prompt and the specific job data.
    Includes error handling and exponential backoff for retries.
    """
    retries = 3
    delay = 2  # start with 2 seconds
    
    # We create the "user prompt" which is just the JSON data for the single job
    user_prompt = f"""
Here is the JSON data for the job. Please analyze it according to the instructions.

{job_data_str}
"""
    
    for attempt in range(retries):
        try:
            # Make the API call
            response = model.generate_content(
                user_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            # The model's response (as text) should be our JSON
            return response.text
        
        except Exception as e:
            print(f"  > API call failed (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"  > Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff (2s, 4s, 8s)
            else:
                print(f"  > All retries failed. Skipping this job.")
                return None

# --- MAIN EXECUTION ---

def main():
    """
    The main function that runs our script.
    """
    print("--- AI Skill Category Extractor ---")
    
    # 1. Load API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file.")
        print("Please follow Step 4 to get your key and add it to the .env file.")
        return
    
    # Configure the Gemini client
    genai.configure(api_key=api_key)

    # 2. Load data files
    codebook_md = load_codebook_as_markdown()
    all_jobs_data = load_skills_data()
    
    if not codebook_md or not all_jobs_data:
        print("Failed to load data. Exiting.")
        return

    # 3. Build the System Prompt
    system_prompt = build_system_prompt(codebook_md)
    
    # 4. Set up the AI Model
    # We pass the system_prompt in when we initialize the model
    model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=system_prompt
    )
    
    print(f"\nModel {MODEL_NAME} initialized. Starting processing...")
    
    # This dictionary will hold all our final results
    final_results = {}
    total_jobs = len(all_jobs_data)
    
    # 5. Loop through each job and process it
    for i, (job_id, job_data) in enumerate(all_jobs_data.items()):
        print(f"\nProcessing job {i+1}/{total_jobs} (ID: {job_id})...")
        
        # Convert the Python dictionary for this job into a JSON string
        job_data_str = json.dumps(job_data, indent=2)
        
        # Call the API
        response_text = call_gemini_api(model, system_prompt, job_data_str)
        
        if response_text:
            try:
                # Try to parse the AI's text response as JSON
                job_result = json.loads(response_text)
                # Add the valid JSON result to our main dictionary
                final_results[job_id] = job_result
                print(f"  > Success! Job {job_id} processed.")
            except json.JSONDecodeError:
                print(f"  > ERROR: AI returned invalid JSON. Skipping job {job_id}.")
                print(f"  > AI response was: {response_text[:200]}...") # Print first 200 chars
        
        # Be nice to the API - wait 1 second between calls
        time.sleep(1) 

    # 6. Save the final results
    if final_results:
        print(f"\nProcessing complete. Saving {len(final_results)} results to {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                # 'indent=2' makes the JSON file human-readable
                json.dump(final_results, f, indent=2)
            print("--- All done! ---")
            print(f"You can find your new file at: {OUTPUT_FILE}")
        except Exception as e:
            print(f"ERROR: Failed to write output file: {e}")
    else:
        print("No jobs were processed successfully.")

# This little line makes sure the 'main()' function runs when you
# execute the file directly (which is what we're about to do!)
if __name__ == "__main__":
    main()
