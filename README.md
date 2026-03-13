# COMP3610 - Big Data Analytics
**Assignment 2: ML Model Training & Evaluation**

---
 
## Overview

This project builds, evaluates, and interprets machine learning models to predict taxi trip tip amounts using the NYC Yellow Taxi Trip dataset (January 2024). It covers feature engineering, regression and classification modelling with Scikit-learn, and a feedforward neural network built with PyTorch.
 
---
 
## Repository Structure
 
```
├── assignment2.ipynb   # Main notebook (Parts 1, 2 & 3)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Excludes data files and model artifacts
```

---
 
## Setup Instructions
 
### 1. Clone the repository
 
```bash
git clone <your-repo-url>
cd <repo-folder>
```
 
### 2. Create and activate a virtual environment (optional but recommended)
 
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```
 
### 3. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4. Download the data
 
The notebook will automatically download the required files on first run:
 
- **NYC Yellow Taxi Trip Records (January 2024):** https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
- **Taxi Zone Lookup Table:** https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
 
Files are saved to `data/raw/` which is excluded from version control.
 
### 5. Run the notebook
 
```bash
jupyter notebook assignment2.ipynb
```
 
Or open in [Google Colab](https://colab.research.google.com/) — all required libraries are pre-installed in Colab's free tier.
 
---
 
## Notebook Structure
 
| Section | Description |
|---|---|
| Part 1 | Data preprocessing, feature engineering, train/val/test splitting |
| Part 2 | Baseline models, hyperparameter tuning, neural network training |
| Part 3 | Test set evaluation, visualisations, feature importance, written analysis |
 
---
 
## Models Trained
 
**Regression** (predicting `tip_amount`):
- Linear Regression
- Random Forest Regressor (baseline + tuned)
- Feedforward Neural Network (PyTorch)
 
**Classification** (predicting `high_tip` — tip > 20% of fare):
- Logistic Regression
- Random Forest Classifier
 
---
 
## Requirements
 
See `requirements.txt` for full details. Key libraries:
 
- Python 3.11+
- `polars`, `pandas`, `numpy`, `pyarrow`
- `scikit-learn`, `scipy`
- `torch`
- `matplotlib`
 
---
 
## Notes
 
- `random_state=42` is used throughout for reproducibility.
- Hyperparameter tuning was performed on a 200,000-row stratified sample due to runtime constraints.
- The dataset is filtered to credit card payments only (`payment_type == 1`) as tip amounts are only reliably recorded for these transactions.