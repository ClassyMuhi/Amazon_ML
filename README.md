# Amazon Business Entity Resolution Challenge 2026

## Overview
The **Amazon Business Entity Resolution Challenge 2026** focuses on identifying, matching, and linking records that refer to the same real-world business entity across disparate, noisy data sources. Business entity data commonly suffers from variations, typographical errors, disparate naming conventions, missing attributes, and conflicting address formats. The objective is to design a high-precision, high-recall, and computationally scalable entity resolution system capable of linking business records accurately.

---

## Team Members
*Team Name: [To be determined]*
- **Member 1:** [Name / Role / Contact]
- **Member 2:** [Name / Role / Contact]
- **Member 3:** [Name / Role / Contact]
- **Member 4:** [Name / Role / Contact]

---

## Project Structure
```text
Business-Entity-Resolution/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── normalization.py
│   ├── blocking.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   ├── threshold.py
│   ├── evaluation.py
│   └── pipeline.py
│
├── experiments/
│
├── models/
│
├── output/
│
├── utils/
│
├── README.md
├── requirements.txt
├── .gitignore
└── DATASET_ANALYSIS.md
```

---

## Final Methodology (Placeholder)
*This section will document the end-to-end algorithmic strategy once experiments are finalized.*

1. **Data Preprocessing & Normalization:** Standardizing entity names, addresses, contact information, abbreviations, and casing.
2. **Blocking / Candidate Selection:** Partitioning data into blocks (e.g., prefix blocking, soundex, Q-grams, sorted neighborhood) to avoid $O(N^2)$ comparisons.
3. **Feature Engineering:** String similarity distances (Levenshtein, Jaro-Winkler, token matching), phonetic encodings, and geographic proximity metrics.
4. **Classification & Scoring:** Machine learning model (e.g., LightGBM / GBDT) trained to score likelihood of candidate pairs being true duplicates.
5. **Threshold Tuning & Post-Processing:** Optimization of classification cutoffs for maximum F1 score, followed by transitive closure or graph clustering (connected components) to form final entity clusters.

---

## How to Run the Project (Placeholder)

### 1. Environment Setup
Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Placement
Place raw competition data into the appropriate folders:
- Training data: `dataset/train/`
- Test data: `dataset/test/`

### 3. Pipeline Execution
Run the end-to-end pipeline:
```bash
python -m src.pipeline
```
*Detailed training and inference flags will be updated as modules are implemented.*
