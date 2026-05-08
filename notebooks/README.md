# Notebooks

Jupyter notebooks demonstrating the NLP pipeline components using 
publicly available multilingual mental health datasets.

## Notebooks (in order)
- `01_data_exploration.ipynb` — Exploratory data analysis of CNSocialDepress dataset
- `02_text_preprocessing.ipynb` — Multilingual (ZH/EN) text cleaning pipeline
- `03_classification_demo.ipynb` — MentalBERT fine-tuning demonstration
- `04_risk_scoring.ipynb` — XGBoost feature importance analysis

## Requirements
```bash
pip install transformers torch pandas numpy matplotlib seaborn jieba
```

## Data
All notebooks use publicly available research datasets only.
No real patient data is included. See `/data/README.md` for details.
