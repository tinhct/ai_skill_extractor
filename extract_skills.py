import os
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm  # This imports the progress bar library

# --- System Prompt Definition ---
# This is your detailed set of instructions for the AI.
# I've modified the "Output Format" section slightly to make it
# more reliable for batch processing. The AI will *only* return
# the skill dictionary, and our Python script will handle the 'job_id'.
SYSTEM_PROMPT = """
You are an expert skill extractor with specialization in analyzing technical job postings. Your task is to carefully analyze each job description I provide to extract structured skill and employer expectation information.

# Context Specification
The input is a raw job posting that may contain various sections including job overview, responsibilities, requirements, and company information.

# Step 2: Extract and categorize information
Follow this step-by-step process:
1. Read the entire job description carefully.
2. Identify all mentioned skills, qualifications, and requirements.
3. Categorize each identified item into the appropriate category:
-- qualifications: Preferred or required educational degree from employers.
-- experience: Required number of years of experience in project management, A/ML, or in as specific skills from employers.
-- role: Explicitly stated responsibilities and duties of the position from employers.
–- project_management_skills: Mention of PM methodologies such as Agile, Scrum, Waterfall, risk management, project planning, scheduling, resource management, budget management, cost control.
-- leadership_and_soft_skills: Mention of stakeholder communication, engagement, expectation management, communication, team leadership,  collaboration, mentorship, conflict resolution, negotiation skills, interpersonal skills
-- domain_and_business: Mention of business acumen, strategy, or commercial awareness
-- adaptability: Mention of adaptability, continuous learning, or thriving in a fast-paced environment
-- conceptual_AI_knowledge: Mention of understanding or familiarity with AI/ML Paradigms, Core concept,  Data Literacy, Math Foundations
-- hands-on_technical skills: Mention of proficiency or experience with specific languages, frameworks, platforms, data tools, project management Softwares, source code version control
-- integration_and_deployment: Mention of MLOps, CI/CD for AI, model deployment, or production systems
-- academic_degrees: Mention of formal postgraduate degrees (Master's or Doctorate) or specific undergraduate disciplines required or preferred by the employer
-- professional_certifications: Mention of Project Management/Agile methodology certification, technical certifications (AI, Cloud)
-- ethical_and_regulatory: Mention of contexts regard to ethical AI, responsible AI, bias, fairness, or transparency, specific regulations
-- project_complexity: Mention of contexts regard to managing complex, ambiguous, or innovative projects
-- other: Any requirements or expectations that don’t clearly fit the above categories

4. Determine the application domain where the job will be applied. This is NOT just ‘‘tech’’ or ‘‘AI’’ but the specific industry or field where the company will apply AI (e.g., healthcare, finance, education, marketing, software development)

# Output Format
Respond ONLY with a valid JSON object using this exact structure.
Do NOT include the job_id, as that will be handled by the script.

{
  "qualifications": ["qual1", "qual2", ...],
  "experience": ["expectation1", "expectation2", ...],
  "role": ["expectation1", "expectation2", ...],
  "project_management_skills": ["skill1", "skill2", ...],
  "leadership_and_soft_skills": ["skill1", "skill2", ...],
  "domain_and_business": ["skill1", "skill2", ...],
  "adaptability": ["skill1", "skill2", ...],
  "conceptual_AI_knowledge": ["skill1", "skill2", ...],
  "hands-on_technical skills": ["skill1", "skill2", ...],
  "integration_and_deployment": ["skill1", "skill2", ...],
  "academic_degrees": ["qual1", "qual2", ...],
  "professional_certifications": ["qual1", "qual2", ...],
  "ethical_and_regulatory": ["context1", "context2", ...],
  "project_complexity": ["context1", "context2", ...],
  "job_domain": "The specific industry or application area where prompt engineering will be applied",
  "other": ["item1", "item2", ...]
}

Important guidelines:
-- Extract skills or qualifications or employer expectations as concise noun phrases (1--4 words typically)
-- Separate distinct skills (don’t combine multiple skills into one entry)
-- Format consistently using lowercase
-- Include ALL relevant skills mentioned
-- For job_domain, identify the SPECIFIC industry or application area where the company operates and will apply prompt engineering (NOT just ‘‘technology’’ or ‘‘AI’’)
-- If multiple domains are mentioned, list the primary domain or focus
-- The output must be valid JSON (double quotes, commas between items)
-- Contain only the JSON object, no additional text or explanations.
"""

def main():
    """
    Main function to run the skill extraction process.
    """
    print("Starting AI Skill Extractor...")

    # --- 1. Load Configuration ---
    # Load the secret API key from our .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found. Please check your .env file.")
        return

    # Configure the Gemini client
    genai.configure(api_key=api_key)

    # Define the file paths
    input_csv_path = "data/job_ads.csv"
    output_json_path = "data/extracted_skills.json"

    # --- 2. Initialize the AI Model ---
    # We use "gemini-1.5-flash-latest" - it's fast and great for bulk tasks.
    # We set `response_mime_type="application/json"` to *force* the AI
    # to give us JSON back, which makes our code much more reliable.
    generation_config = {
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
        model_name="gemini-flash-latest",
        generation_config=generation_config,
        system_instruction=SYSTEM_PROMPT
    )

    print("Gemini model initialized.")

    # --- 3. Load Input Data ---
    try:
        # We use pandas to read the CSV file. It's like a smart spreadsheet.
        df = pd.read_csv(input_csv_path)
        # Handle cases where a job description might be empty
        df = df.dropna(subset=['job_description'])
        print(f"Loaded {len(df)} job ads from {input_csv_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        print("Please make sure you have created 'data/job_ads.csv' and added data to it.")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # --- 4. Process Each Job Ad ---
    # This dictionary will hold all our final results
    all_results = {}

    print("Processing job ads (this may take a few minutes)...")
    # We loop through each row in our spreadsheet.
    # `tqdm` wraps around our loop to create a beautiful progress bar!
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Skills"):
        job_id = str(row['job_id'])
        job_description = row['job_description']

        # We use a "try/except" block. This is a crucial safety net.
        # It "tries" to run the code inside. If it fails, it "excepts"
        # the error, prints a message, and continues to the next job
        # instead of crashing the whole script.
        try:
            # This is where we actually send the job description to the AI
            response = model.generate_content(job_description)

            # The AI's response is text, so we use `json.loads` to
            # turn the JSON text into a Python dictionary
            result_data = json.loads(response.text)

            # We store the result in our dictionary with the job_id as the key
            all_results[job_id] = result_data

        except Exception as e:
            # If anything went wrong for this *one* job, print the error
            # and move to the next one.
            print(f"\n[!] Error processing job {job_id}: {e}")
            print("   Skipping this job and continuing...")
            all_results[job_id] = {"error": str(e)} # Log the error in the output

    # --- 5. Save All Results ---
    print(f"\nProcessing complete. Saving results to {output_json_path}...")

    try:
        # We open the output file and use `json.dump` to write our
        # `all_results` dictionary into it in a nicely formatted way.
        with open(output_json_path, 'w', encoding='utf-8') as f:
            # `indent=4` makes the file human-readable
            json.dump(all_results, f, indent=4, ensure_ascii=False)

        print("Done! You can now check 'data/extracted_skills.json' for the output.")
    except Exception as e:
        print(f"Error saving results to file: {e}")

# This standard Python line checks if the script is being run directly
# (instead of being imported by another script) and, if so,
# calls our `main()` function to start the process.
if __name__ == "__main__":
    main()
