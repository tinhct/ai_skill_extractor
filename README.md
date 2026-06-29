# Skill Extractor

AI Skill Extractor is a thesis analysis pipeline for AI Project Manager job ads. It turns raw job descriptions into structured skill data, codebook mappings, cluster assignments, and hypothesis-test outputs for downstream analysis and visualization.

## What it does

1. Extracts skills and requirements from job descriptions with Gemini and dependency-based parsing rules.
2. Maps extracted phrases to codebook categories and values.
3. Clusters semantically similar skills with SentenceTransformer embeddings and K-Means.
4. Produces CSV outputs for the thesis metrics, hypothesis tests, and exploratory analysis.

## Repository Layout

- `data/`: raw and intermediate datasets used by the pipeline.
- `output/`: thesis CSV outputs from the analysis scripts.
- `source/`: all Python scripts for extraction, mapping, clustering, and hypothesis testing.
- `venv_py313/`: local Python 3.13 virtual environment bundled with the workspace.

## Main Scripts

- `source/1. extract_skills_dependency_parsing.py`: reads `data/job_ads.csv` and writes `data/extracted_skills_dependency_parsing.json`.
- `source/2. extract_category_codes_and_values.py`: maps extracted skills to codebook values and writes `data/extracted_category_codes_and_values.json` and `data/extracted_category_codes.json`.
- `source/clustering.py`: builds embeddings from `data/extracted_skills.json`, evaluates K-Means cluster quality, and writes `source/analysis/kmeans_evaluation.png` and `source/analysis/skill_clusters.csv`.
- `source/test_hypotheses.py`: runs the main thesis hypothesis analysis and writes CSV outputs in `output/`.
- `source/3. master_metrics.py`, `source/4. top_15_competencies.py`, and `source/5. densities_Core_vs_Technical 2.py`: generate additional thesis summary metrics in `output/`.

## Input Files

- `data/job_ads.csv`
- `data/codebook.csv`
- `data/extracted_skills.json`
- `data/extracted_category_codes.json`
- `data/extracted_skills_dependency_parsing.json`

## Generated Outputs

### Data and Analysis Artifacts

- `data/extracted_skills_dependency_parsing.json`
- `data/extracted_category_codes_and_values.json`
- `data/extracted_category_codes.json`
- `source/analysis/kmeans_evaluation.png`
- `source/analysis/skill_clusters.csv`

### Thesis CSV Outputs

- `output/master_metrics.csv`
- `output/top_15_competencies.csv`
- `output/densities_Core_vs_Technical.csv`
- `output/h1_aggregate_competency.csv`
- `output/h2_tech_dimensions.csv`
- `output/h3_hybrid_test_result.csv`
- `output/h3_co_occurrence_heatmap.csv`
- `output/h4_h5_prevalence_results.csv`
- `output/exploratory_skill_clusters.csv`

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The extraction scripts expect a local `.env` file containing `GEMINI_API_KEY`.

If you want to use the bundled environment, activate it with:

```bash
source venv_py313/bin/activate
```

## Typical Workflow

Run the scripts in this order:

```bash
python source/1.\ extract_skills_dependency_parsing.py
python source/2.\ extract_category_codes_and_values.py
python source/clustering.py
python source/test_hypotheses.py
python source/3.\ master_metrics.py
python source/4.\ top_15_competencies.py
python source/5.\ densities_Core_vs_Technical\ 2.py
```

## Notes

- The project currently focuses on AI Project Manager job postings.
- Clustering uses `sentence-transformers`, `scikit-learn`, and silhouette scoring to select a suitable `k`.
- `source/clustering.py` creates its own `source/analysis/` folder if it does not already exist.
- The numbered script filenames include spaces, so quoting or escaping them is required on the command line.
