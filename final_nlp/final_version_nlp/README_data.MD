# SemEval 2014 – Aspect-Based Sentiment Analysis (Restaurants)

## Overview
This folder contains the **SemEval-2014 Task 4 (Aspect-Based Sentiment Analysis)** dataset
for the **Restaurants** domain, exported to CSV format.

The data is automatically downloaded from the Hugging Face Hub and converted
from its original XML format into structured tables using the `datasets` library.

---

## Data Source
- **Task:** SemEval-2014 Task 4 – Aspect-Based Sentiment Analysis (ABSA)
- **Domain:** Restaurants
- **Original organizers:** SemEval / ACL
- **Hugging Face dataset:** `alexcadillon/SemEval2014Task4`

The Hugging Face version is a faithful conversion of the official SemEval-2014 data.

---

## How the data is generated
Running the extraction script:

1. Downloads the SemEval-2014 Restaurants dataset from Hugging Face  
2. Loads the dataset using the official configuration (`restaurants`)
3. Exports each official split into a separate CSV file

No preprocessing or filtering is applied — the CSV files preserve the original annotations.

---

## Files
| File name | Description | Rows |
|----------|-------------|------|
| `semeval2014_restaurants_train.csv` | Training set | 3041 |
| `semeval2014_restaurants_test.csv` | Test set | 800 |
| `semeval2014_restaurants_trial.csv` | Trial / development set | 100 |

---

## CSV Structure
Each row corresponds to **one sentence** and contains:

- `sentenceId` — unique sentence identifier
- `text` — raw sentence text
- `aspectTerms` — list of annotated aspect terms with polarity and character spans
- `aspectCategories` — list of aspect categories with sentiment polarity

This format supports both:
- **Aspect Term Sentiment Analysis (ATSA)**
- **Aspect Category Sentiment Analysis (ACSA)**

---

## Notes
- Dataset size is intentionally small; SemEval-2014 is a benchmark dataset.
- CSV files contain nested JSON-like fields (`aspectTerms`, `aspectCategories`).
  These can be flattened if required for model training.

