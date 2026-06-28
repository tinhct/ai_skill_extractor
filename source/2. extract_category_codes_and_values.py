import os
import json
import csv
import asyncio
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Set
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --- CONFIGURATION ---
load_dotenv()

# Paths
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
CODEBOOK_FILE = DATA_DIR / "codebook.csv"
SKILLS_FILE = DATA_DIR / "extracted_skills_dependency_parsing.json"
OUTPUT_FILE = DATA_DIR / "extracted_category_codes_and_values.json"

# API & Model Config
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

# Performance Tuning
CONCURRENT_REQUESTS = 5 
SAVE_BATCH_SIZE = 10 

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class SkillExtractor:
    def __init__(self):
        self.codebook_md = ""
        self.system_instruction = ""
        self.valid_codes: Set[str] = set() # Store valid codes for validation
        self.results: Dict[str, Any] = {}
        
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        
        genai.configure(api_key=API_KEY)
        self.model = None 

    def load_data(self) -> bool:
        """Loads codebook and skills data. Extracts valid codes for validation."""
        # 1. Load Codebook
        if not CODEBOOK_FILE.exists():
            logger.error(f"Codebook not found at {CODEBOOK_FILE}")
            return False
            
        try:
            with open(CODEBOOK_FILE, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = [row for row in reader]

                # Identify the column index for 'sub_category_code'
                try:
                    code_idx = header.index("sub_category_code")
                except ValueError:
                    # Fallback if header name is slightly different, usually index 2 based on your snippet
                    code_idx = 2 
                
                # Populate the validation set
                for row in rows:
                    if len(row) > code_idx:
                        code = row[code_idx].strip()
                        if code:
                            self.valid_codes.add(code)

            # Format for LLM
            self.codebook_md = "| " + " | ".join(header) + " |\n"
            self.codebook_md += "|---" * len(header) + "|\n"
            for row in rows:
                self.codebook_md += f"| {' | '.join(row)} |\n"
            
            logger.info(f"Codebook loaded. Found {len(self.valid_codes)} valid unique codes.")
            
        except Exception as e:
            logger.error(f"Error reading codebook: {e}")
            return False

        # 2. Load Skills
        if not SKILLS_FILE.exists():
            logger.error(f"Skills file not found at {SKILLS_FILE}")
            return False
            
        try:
            with open(SKILLS_FILE, mode='r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
                logger.info(f"Loaded {len(self.raw_data)} jobs to process.")
                return True
        except Exception as e:
            logger.error(f"Error reading skills JSON: {e}")
            return False

    def build_system_prompt(self):
        """Constructs the prompt with strict negative constraints."""
        
        intro = f"""
You are a senior data extraction specialist. Process job postings using the provided codebook.

--- CODEBOOK START ---
{self.codebook_md}
--- CODEBOOK END ---
"""
        instructions = r"""
YOUR MANDATE:
1. Map extracted skills, expectations, requirements to 'sub_category_code' values EXACTLY as they appear in the codebook.
2. INPUT FILTERING: The input JSON contains skills with a 'weight' attribute. You must ONLY code skills where "weight" is 1 or 2. Ignore any items with "weight" = 0.
3. CRITICAL: NEVER output raw text (e.g., "experience in python"). If a skill does not map to a code, OMIT IT.
4. Output lists must ONLY contain uppercase codes (e.g., ["TECH-NLP", "EXP-PM-YRS"]).
5. Apply REGEX logic for years of experience.

Step 1 – Mapping Logic
- Map skills to sub_category_codes based on semantic meaning.
- Ignore low-relevance items (weight < 1).
- Structure output into: Employer expectations, Core competencies, Technical competencies, Educational attainment, Project context.
- Use ONLY codes. Empty lists [] are preferred over raw text.

- Pattern: (\d+)(\+)?\s*(?:years?|yrs?)\s*(?:of)?\s*(.*?)
- Context Window: ±50 chars.
- Logic: 
    - AI tokens (ml, ai, data, nlp) -> EXP-AI-YRS.
    - PM tokens (project manager, pm, leadership) -> EXP-PM-YRS.
- Values: Integer for value. Type: "Standard", "Modifier" (+), "Range", "Qualitative" (text only), "Missing".
- Handling Complex Cases (The "Plus" and Ranges)
    - "Modifier" (+): If the extraction captures a "plus" sign (e.g., "5+ years"), the integer 5 is retained. The "plus" is treated as a floor value, which is standard practice in minimum qualification analysis.
    - Ranges: If a range is detected (e.g., "5-7 years"), the lower bound (5) is extracted as the conservative "Minimum Requirement".
- Fallback and Missing Data Resolution: A critical distinction must be made between "No Experience Mentioned" and "Unparseable Experience": 
    - Scenario A (No Match): If the dictionary search for EXP-PM-YRS terms returns 0 hits (i.e., the ad never mentions "years of project management"), the value is coded as NA (Null) and Type is missing. We do not impute 0, as 0 implies "No experience required" (Entry Level), whereas NA implies the employer failed to state a preference.
    - Scenario B (Keyword Match, No Number): If the ad matches the dictionary (e.g., "Must have strong Project Management experience") but lacks a detectable integer near the term, the value is coded as NA (Null), and Type is Qualitative.

Step 3 – Output Schema: Respond ONLY with a valid JSON object for that single job, using this exact structure (do NOT include a top-level job_id key)
{
    "employer expectation": {
      "experience": ["CODE"], 
      "exp_pm_years_value": null, "exp_pm_years_type": "Missing",
      "exp_ai_years_value": null, "exp_ai_years_type": "Missing", "role": ["CODE"]
    },
    "core competencies": {
      "project management": ["CODE"], "leadership and soft skills": ["CODE"], "domain and business": ["CODE"], "adaptability": [], "change management": []
    },
    "technical competencies": {
      "conceptual AI knowledge": ["CODE"], "hands-on technical skills": ["CODE"], "technical operation": [], 
      "data operation": [], "generative AI": [], "Agentic AI": [], "AI evaluation": [], 
      "financial operations": [], "enterprise platforms": [],"advanced AI": []
    },
    "educational attainment": { "academic qualification": [], "professional certification": [] },
    "project context": {
      "governance and compliance": [], "cybersecurity": [], "complexity": [], "scale AI": []
    }
}
"""
        self.system_instruction = intro + instructions
        self.model = genai.GenerativeModel(
            MODEL_NAME,
            system_instruction=self.system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )

    def validate_and_clean_output(self, data: Any) -> Any:
        """
        Recursively walks the JSON. 
        If it finds a list, it filters out any string that is NOT in self.valid_codes.
        """
        if isinstance(data, dict):
            return {k: self.validate_and_clean_output(v) for k, v in data.items()}
        elif isinstance(data, list):
            cleaned_list = []
            for item in data:
                # If item is a string, it MUST be a valid code.
                if isinstance(item, str):
                    # We strip whitespace and check validity
                    clean_item = item.strip()
                    if clean_item in self.valid_codes:
                        cleaned_list.append(clean_item)
                    else:
                        # Optional: Log dropped items if you want to debug hallucinations
                        # logger.debug(f"Dropped invalid code: {clean_item}")
                        pass
                else:
                    # Keep non-string items if any (though schema says strings)
                    cleaned_list.append(item)
            return cleaned_list
        else:
            return data

    async def process_job(self, job_id: str, job_data: dict, semaphore: asyncio.Semaphore) -> Optional[tuple]:
        async with semaphore:
            retries = 3
            delay = 2
            user_prompt = f"Analyze this job data:\n{json.dumps(job_data)}"

            for attempt in range(retries):
                try:
                    response = await asyncio.to_thread(self.model.generate_content, user_prompt)
                    text = response.text
                    
                    if "```" in text:
                        text = re.sub(r"```json\s*|\s*```", "", text)
                    
                    parsed_json = json.loads(text)
                    
                    # Normalize structure
                    if job_id not in parsed_json:
                        if "employer expectation" in parsed_json or list(parsed_json.keys())[0] != job_id:
                             parsed_json = {job_id: parsed_json}
                    
                    job_result = parsed_json[job_id]

                    # --- CRITICAL STEP: CLEANING LAYER ---
                    # We run the validator here to remove any raw text hallucinations
                    cleaned_result = self.validate_and_clean_output(job_result)
                    
                    return job_id, cleaned_result

                except json.JSONDecodeError:
                    logger.warning(f"[{job_id}] JSON Error (Attempt {attempt+1}).")
                except google_exceptions.ResourceExhausted:
                    logger.warning(f"[{job_id}] Rate limit (429). Sleeping 10s...")
                    await asyncio.sleep(10)
                except Exception as e:
                    logger.warning(f"[{job_id}] Error: {e}")
                
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
            
            logger.error(f"[{job_id}] Failed.")
            return None

    def save_results(self):
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Checkpoint: Saved {len(self.results)} jobs.")
        except Exception as e:
            logger.error(f"Failed to save output: {e}")

    async def run(self):
        print(f"--- AI Skill Extractor ({MODEL_NAME}) ---")
        if not self.load_data(): return
        self.build_system_prompt()
        
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        tasks = []
        
        logger.info(f"Processing {len(self.raw_data)} jobs...")

        for job_id, job_data in self.raw_data.items():
            task = asyncio.create_task(self.process_job(job_id, job_data, semaphore))
            tasks.append(task)

        completed_count = 0
        for future in asyncio.as_completed(tasks):
            result = await future
            completed_count += 1
            if result:
                jid, data = result
                self.results[jid] = data
            
            if completed_count % SAVE_BATCH_SIZE == 0:
                self.save_results()
                print(f"  > Progress: {completed_count}/{len(self.raw_data)} ({int(completed_count/len(self.raw_data)*100)}%)")

        self.save_results()
        print("\n--- Processing Complete ---")

if __name__ == "__main__":
    try:
        extractor = SkillExtractor()
        asyncio.run(extractor.run())
    except KeyboardInterrupt:
        print("\nInterrupted.")