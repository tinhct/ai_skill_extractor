# AI Skill Extractor

This project analyzes AI Project Manager job postings and transforms them into structured skill data for thesis analysis. The pipeline combines LLM-based extraction, codebook-based categorization, embedding-based clustering, and hypothesis testing to quantify competency demands.

## What the project does

1. Extracts skills and requirements from raw job posting text with dependency-parsing rules and Gemini.
2. Maps extracted phrases to codebook categories and values.
3. Groups skills into semantic clusters with SentenceTransformer embeddings and K-Means.
4. Produces CSV outputs for hypothesis testing and visualization.

## Main scripts

- `source/1. extract_skills_dependency_parsing.py`: extracts structured skills from job ads and writes `data/extracted_skills_dependency_parsing.json`.
- `source/2. extract_category_codes_and_values.py`: maps extracted skills to codebook codes and writes `data/extracted_category_codes_and_values.json`.
- `clustering.py`: builds embeddings from extracted skills, evaluates K-Means cluster quality, and writes `analysis/kmeans_evaluation.png` and `analysis/skill_clusters.csv`.
- `source/test_hypotheses.py`: runs the thesis hypothesis analysis and writes CSV outputs in `output/`.
- `source/3. master_metrics.py`, `source/4. top_15_competencies.py`, and `source/5. densities_Core_vs_Technical 2.py`: generate additional thesis metrics in `output/`.

## Input data

- `data/job_ads.csv`
- `data/codebook.csv`
- `data/extracted_skills.json`
- `data/extracted_category_codes.json`
- `data/extracted_skills_dependency_parsing.json`

## Outputs

- `output/master_metrics.csv`
- `output/top_15_competencies.csv`
- `output/densities_Core_vs_Technical.csv`
- `output/h1_aggregate_competency.csv`
- `output/h2_tech_dimensions.csv`
- `output/h3_hybrid_test_result.csv`
- `output/h3_co_occurrence_heatmap.csv`
- `output/h4_h5_prevalence_results.csv`
- `output/exploratory_skill_clusters.csv`
- `analysis/kmeans_evaluation.png`
- `analysis/skill_clusters.csv`

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The extraction pipeline expects a `GEMINI_API_KEY` in a local `.env` file.

## Typical workflow

```bash
source venv/bin/activate
python source/1.\ extract_skills_dependency_parsing.py
python source/2.\ extract_category_codes_and_values.py
python clustering.py
python source/test_hypotheses.py
```

## Notes

- The project currently focuses on AI Project Manager job postings.
- Clustering uses `sentence-transformers`, `scikit-learn`, and silhouette scoring to choose a suitable `k`.
- The hypothesis scripts are designed to convert the extracted JSON into CSV tables for downstream analysis and visualization.
