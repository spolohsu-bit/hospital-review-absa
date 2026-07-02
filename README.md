# JMIR #92325 Analysis Materials

This repository contains reproducibility materials for the manuscript:

**What Single-Topic Summaries Miss: Aspect-Level Evaluative Structure in Hospital Reviews Using GPT-Based Sentiment Analysis**

The materials include prompt templates and analysis scripts used for aspect-based sentiment analysis, LDA topic modeling, validation metrics, cross-model agreement checks, and prevalence-adjusted PMI sensitivity analysis.

## Repository Contents

```text
prompts/
  stage1_closed_absa_prompt_zh-TW.txt
  stage1_closed_absa_prompt_en.txt
  stage2_open_absa_prompt_zh-TW.txt

scripts/
  run_absa_demo.py
  run_lda_k7.py
  calc_gold_irr_accuracy.py
  compute_cross_model_agreement.py
  compute_pmi_by_rating.py
```

## Data Availability

Raw Google Reviews are not redistributed because of Google Maps platform terms of service. The scripts are provided to document the analysis logic and to support reproducibility using locally available or authorized data with the same structure.

Derived aggregate tables and additional verification materials may be available from the corresponding author upon reasonable request, subject to data-use restrictions.

## Prompt Templates

- `prompts/stage1_closed_absa_prompt_zh-TW.txt`: closed-set ABSA prompt using seven predefined service-quality aspects (Traditional Chinese, as used in the study).
- `prompts/stage1_closed_absa_prompt_en.txt`: English translation of the closed-set ABSA prompt.
- `prompts/stage2_open_absa_prompt_zh-TW.txt`: open-set prompt used to inspect possible emerging aspects outside the seven-category taxonomy.

Both prompt templates use `{review_text}` as the review placeholder.

## Scripts

### `run_absa_demo.py`

Minimal single-review demo using the Stage 1 prompt and the OpenAI API.

```bash
pip install openai
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
python scripts/run_absa_demo.py --review "醫生很專業但掛號等太久"
```

This script processes one user-provided review only. It does not include batch-processing, checkpointing, retry logic, or the production API pipeline.

### `run_lda_k7.py`

Trains the K=7 LDA model on the full analytic corpus used in the manuscript.

Expected input: a CSV file with at least:

```text
review_text
```

Example:

```bash
pip install pandas jieba gensim scikit-learn
python scripts/run_lda_k7.py --data path/to/combined_hospital_data.csv \
  --stopwords path/to/stopwords.txt \
  --medical-dict path/to/medical_dict.txt
```

The manuscript analysis used 5,467 reviews in the full analytic sample.

### `calc_gold_irr_accuracy.py`

Computes human inter-rater agreement and GPT-vs-gold validation metrics.

Expected input: an Excel file in long format with columns such as:

```text
dimension, Human1, Human2, gpt
```

Example:

```bash
pip install pandas openpyxl numpy scikit-learn
python scripts/calc_gold_irr_accuracy.py --input path/to/gold_standard.xlsx --all-rows
```

### `compute_cross_model_agreement.py`

Computes agreement metrics for cross-model validation.

Expected local files:

```text
scripts/gold_standard.xlsx
scripts/results/deepseek_absa_results.json
scripts/results/claude_absa_results.json
```

Expected JSON format for each model result:

```json
[
  {
    "review_id": 1,
    "aspect_scores": {
      "Professional Quality": 1,
      "Administrative Processes": -1
    }
  }
]
```

### `compute_pmi_by_rating.py`

Computes rating-stratified Jaccard and PMI values for aspect co-mention.

Expected local file:

```text
scripts/layer2_data.csv
```

Required columns:

```text
review_id, star_rating, aspect_category
```

The script defines positive reviews as 4-5 stars and negative reviews as 1-2 stars.

## Security Notes

This repository does not include API keys, passwords, cookies, raw Google Reviews, or production batch API scripts. Users must provide their own API keys and authorized local data where needed.

The included `.gitignore` excludes common local data folders, generated outputs, environment files, and credential files to reduce the risk of accidental disclosure.

## Persistent Identifier

For journal submission, create a public GitHub release and archive it through Zenodo, OSF, or another repository service that issues a persistent identifier. Replace the manuscript placeholder with the final DOI or persistent URL after archiving.
