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
# Step 1: Examine the job description
You are an expert skill extractor with specialization in analyzing technical job postings using advanced Natural Language Processing, including part-of-speech (POS) tagging and dependency parsing. Your task is to carefully analyze each text to extract structured skill and employer expectation information, and to assign a tiered weighting to each extracted item based on its grammatical context.

# Context Specification
The input is a raw job posting that may contain various sections including job overview, responsibilities, requirements, and company information. You must treat the text as sentences, not as a single continuous block. Only text that clearly refers to the advertised role should drive the extraction (ignore generic “About us” boilerplate unless it explicitly links to the role)

# Step 1a: Environment and dictionary setup (conceptual)
Assume access to an NLP stack equivalent to Python spaCy with:
- Sentence segmentation
- POS tagging
- Dependency parsing

Define a dictionary of modal/context terms that can modify the importance of a skill or requirement:

- Mandatory indicators:
  "must", "require", "requires", "required", "requiring", "essential",
  "critical", "necessary", "core", "fundamental"

- Preferred indicators:
  "plus", "desirable", "advantage", "preferred", "bonus",
  "ideal", "nice to have", "nice-to-have"

- Negative indicators:
  "no", "not", "without", "irrelevant", "unnecessary"

These indicators can appear as verbs, auxiliaries, adjectives, or noun modifiers around the skill phrase.

# Step 2: Extract, parse, and categorize information

Follow this step-by-step process:

1. Split into sentences
   - Segment the job description into individual sentences.
   - Process each sentence independently with POS tagging and dependency parsing.

2. Identify candidate skills / expectations
   - Within each sentence, identify all occurrences of skills, qualifications, and requirements that match your Codebook categories (e.g., tools, methods, degrees, soft skills, domains).
   - Treat each distinct mention as a "Target Node" in the dependency tree (for example, "Python", "PMP certification", "stakeholder management").

3. Dependency parsing for each Target Node
   For each Target Node in a sentence:
   - Locate the token span representing the skill or requirement.
   - Determine the syntactic head of the span (usually a verb or governing noun).
   - Inspect:
     - The head token
     - Direct children of the Target Node
     - Direct children of the head token
     - Relevant auxiliaries or copulas linked to the head

   Specifically:
   - Negation:
     - Check if the Target Node or its head has any child with a dependency label typically used for negation (e.g., "neg") or a token from the Negative indicators list.
   - Mandatory context:
     - Check if the head verb or a closely linked auxiliary/verb lemma is in the Mandatory indicators list (e.g., "requires", "must").
     - Or if the Target Node is modified by an adjective or noun in the Mandatory indicators list (e.g., "essential Python skills").
   - Preferred context:
     - Check if the Target Node or its head is modified by a Preferred indicator (e.g., "Python is a plus", "preferred experience with Python").

4. Tiered weighting classification rules
   For every occurrence of a skill or requirement, assign a tiered weight \(W\) using these rules (apply in this order):

   - Rule A (Negation – weight 0):
     If the Target Node or its head has a negation child (dependency label like "neg") OR a nearby Negative indicator modifying it (e.g., "no Python required", "Python is not necessary"):
       - Set \(W = 0\).
       - Do not treat this skill as required or preferred (you may still record it in "other" if useful, but mark it as weight 0).

   - Rule B (Mandatory – weight 2):
     If NOT negated and ANY of the following holds:
       - The Target Node is the direct object of a verb with a lemma in the Mandatory indicators list
         (e.g., "We require Python", "You must know Python").
       - The Target Node is syntactically modified by an adjective or noun in the Mandatory indicators list
         (e.g., "essential Python skills", "core data science expertise").
       Then:
       - Set \(W = 2\).

   - Rule C (Preferred – weight 1):
     If NOT classified as negated or mandatory and the Target Node or its head is modified by a Preferred indicator
       (e.g., "Python is a plus", "experience with AWS is preferred", "SQL would be an advantage"):
       - Set \(W = 1\).

   - Rule D (Default – weight 2):
     If none of the above apply (no explicit negation, mandatory, or preferred modifiers detected):
       - Assume the mention represents a standard requirement.
       - Set \(W = 2\).

   If a skill appears multiple times in different contexts, apply this logic to each occurrence and choose the highest weight observed for that job (e.g., if "Python" appears once as preferred and once as mandatory, final weight for "Python" is 2).

5. Categorize each identified item
   After assigning \(W\) for each extracted item, map it into the appropriate category, preserving the weight for each entry. Represent each extracted phrase as an object with the fields:
   - "text": the concise skill / qualification / expectation as a lowercase noun phrase (1–4 words)
   - "weight": the numeric tiered weight (0, 1, or 2) according to the rules above

   Use the following categories:

   -- experience_pm: Quantitative requirement for years of professional Project Management experience
   -- experience_AI: Quantitative requirement for years of experience specifically in Data, AI, or ML projects
   -- experience_software: Background in SaaS, ISV, or general software development
   -- role_strategic: High-level strategic responsibilities involving vision setting, business alignment, and organizational transformation
   -- role_execution: Tactical responsibilities focused on daily execution, task management, triage, and release cycles
   –- project management_method: Proficiency in established project management frameworks, methodologies, and lifecycles (e.g., agile, safe, waterfall)
   –- project management_risk: Ability to identify, assess, mitigate, and manage delivery risks and dependencies
   –- project management_planning: Competence in project planning, scheduling, resource allocation, and timeline management
   –- project management_budget: Responsibility for financial oversight, budget management, forecasting, and cost control
   –- project management_tool: Technical proficiency with specific project management and collaboration software (e.g., jira, asana, linear)
   –- project management_portfolio: Managing a portfolio of multiple AI investments to ensure ROI (distinct from single-project management)
   -- leadership and soft skills_stakeholder: Skills in managing stakeholder relationships, ensuring business alignment, and securing executive buy-in
   -- leadership and soft skills_leadership: Capabilities regarding team leadership, mentorship, motivation, and cross-functional collaboration
   -- leadership and soft skills_conflict: Proficiency in conflict resolution, negotiation, and navigating ambiguity
   -- domain and business: Demonstration of business acumen, strategic thinking, product strategy, and commercial awareness
   -- adaptability: Capacity to adapt and thrive within fast-paced, ambiguous, or rapidly evolving environments
   -- change management: Driving organizational change, user adoption, and upskilling regarding AI literacy and tools
   -- conceptual AI knowledge: Foundational theoretical knowledge of machine learning, data science concepts
   -- hands-on technical skills: Practical programming skills and proficiency with standard data science and engineering tools (e.g., python, sql, aws)
   -- technical operation: Knowledge of operational infrastructure, deployment pipelines, and ci/cd practices for ML models (mlops)
   -- data operation: Management of the data lifecycle, including annotation, ground truth creation, lineage, and data governance
   -- generative AI_concept: Understanding of architectures, theories, and concepts specific to generative AI and large language models (llms)
   -- generative AI_hands-on: Proficiency with tools, vector databases, and frameworks specific to the generative AI technology stack
   -- Agentic AI: Knowledge of autonomous AI systems and frameworks designed for managing and executing agentic workflows
   -- AI evaluation: Methodologies for assessing the quality, safety, and performance of AI outputs (e.g., red teaming, benchmarks)
   -- financial operations: Financial management and optimization of AI compute resources and consumption models (finops)
   -- enterprise platforms: Experience integrating, configuring, or utilizing enterprise-grade AI assistants and copilot ecosystems
   -- advanced AI: Refers to hypothetical or future AI systems that possess human-level (General) or super-human (Super) cognitive abilities across a broad range of tasks, distinct from current narrow AI
   -- academic qualification_advanced: Specification of required or preferred advanced academic degree levels (e.g., master's, phd)
   -- academic qualification_tech: Requirement for academic degrees in general technical, quantitative, or stem disciplines
   -- academic qualification_AI: Requirement for specialized academic degrees specifically in AI, data science, or cognitive science
   -- academic qualification_business: Requirement for academic degrees in business administration or management
   -- professional certification_pm: Possession of formal certifications in project management or agile methodologies (e.g., pmp, csm)
   -- professional certification_tech: Possession of technical certifications related to cloud platforms, AI, or data engineering
   -- governance and compliance_safety: Focus on AI ethics, societal impact, fairness, explainability, and human-centric design
   -- governance and compliance_compliance: Adherence to legal frameworks, regulatory compliance, data privacy governance (e.g., GDPR), intellectual property rights, copyright, data sovereignty, and open source compliance
   -- cybersecurity: Focus on cybersecurity, system hardening, and defense against AI-specific vulnerabilities
   -- complexity: Experience managing complex, ambiguous, innovative, or R&D-heavy projects
   -- scale AI: Context involving large-scale organizational environments, global reach, or enterprise-wide implementation
   -- other: Any requirements or expectations that don’t clearly fit the above categories

6. Determine application domain
   Determine the specific job domain (industry or primary application area where AI or prompt engineering will be applied), not just "technology" or "AI". If multiple domains are mentioned, select the primary one.

# Output Format

Respond ONLY with a valid JSON object for that single job, using this exact structure (do NOT include a top-level job_id key). For each category, return a list of objects with "text" and "weight" fields:

{
    "experience_pm": [
      { "text": "5+ years project management", "weight": 2 },
      { "text": "enterprise program delivery", "weight": 2 }
    ],
    "experience_AI": [
      { "text": "3+ years ai projects", "weight": 2 }
    ],
    "experience_software": [
      { "text": "saas background", "weight": 1 }
    ],
    "role_strategic": [
      { "text": "ai strategy", "weight": 2 }
    ],
    "role_execution": [
      { "text": "backlog management", "weight": 2 }
    ],
    "project management_method": [
      { "text": "agile", "weight": 2 },
      { "text": "safe", "weight": 1 }
    ],
    "project management_risk": [
      { "text": "risk management", "weight": 2 }
    ],
    "project management_planning": [],
    "project management_budget": [],
    "project management_tool": [
      { "text": "jira", "weight": 2 }
    ],
    "project management_portfolio": [],
    "leadership and soft skills_stakeholder": [],
    "leadership and soft skills_leadership": [],
    "leadership and soft skills_conflict": [],
    "domain and business": [],
    "adaptability": [],
    "change management": [],
    "conceptual AI knowledge": [],
    "hands-on technical skills": [
      { "text": "python", "weight": 2 },
      { "text": "sql", "weight": 2 }
    ],
    "technical operation": [],
    "data operation": [],
    "generative AI_concept": [],
    "generative AI_hands-on": [],
    "Agentic AI": [],
    "AI evaluation": [],
    "financial operations": [],
    "enterprise platforms": [],
    "advanced AI": [],
    "academic qualification_advanced": [],
    "academic qualification_tech": [],
    "academic qualification_AI": [],
    "academic qualification_business": [],
    "professional certification_pm": [],
    "professional certification_tech": [],
    "governance and compliance_safety": [],
    "governance and compliance_compliance": [],
    "cybersecurity": [],
    "complexity": [],
    "scale AI": [],
    "job domain": "the specific industry or application area where prompt engineering or ai application will be applied",
    "other": []
}

Important guidelines:
- Extract skills, qualifications, or employer expectations as concise noun phrases (typically 1–4 words), in lowercase.
- Do not merge distinct skills into a single entry.
- For each entry, always include a "weight" (0, 1, or 2) as defined by the dependency-based rules.
- Include ALL relevant skills and expectations mentioned.
- Ignore generic company boilerplate that is not clearly tied to the advertised role (for example, generic claims like “we are a fast-paced, diverse company” should not be mapped to adaptability unless explicitly linked to role expectations)
- For "job domain", output a single, specific industry or application area string (lowercase).
- The output must be valid JSON (double quotes for all keys and strings, commas between items).
- The response must contain only the JSON object, with no additional commentary.

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
    output_json_path = "data/extracted_skills_dependency_parsing.json"

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

# We are writing to inform you that we will update the following Gemini model aliases starting January 30, 2026 resulting in Thought signatures validation errors unless you take action:
# gemini-pro-latest to gemini-3-pro-preview
# gemini-flash-latest to gemini-3-flash-preview
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

        print("Done! You can now check 'data/extracted_skills_dependency_parsing.json' for the output.")
    except Exception as e:
        print(f"Error saving results to file: {e}")

# This standard Python line checks if the script is being run directly
# (instead of being imported by another script) and, if so,
# calls our `main()` function to start the process.
if __name__ == "__main__":
    main()
