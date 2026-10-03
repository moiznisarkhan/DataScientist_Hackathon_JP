# E-Commerce Intelligence Hackathon

End-to-end e-commerce analytics project built from an SQLite database through SQL business analysis, customer churn prediction, neural-network comparison, NLP sentiment classification, and a Streamlit dashboard.

## Business Highlights

- Total cleaned net revenue (excluding returned orders): **Rs. 865.17M**.
- **Electronics generated ~69.9% of net revenue**, making it the dominant revenue category.
- **Karachi generated ~26.3% of net revenue**, the largest city contribution.
- **Fashion had the highest return rate at 11.25%**, materially above Groceries (2.29%) and Books & Stationery (2.09%).
- July 2026 was the strongest month in the supplied 2026 period at **Rs. 76.61M**, followed by an ~11.6% decline in August.

## Data Quality Decisions

- 30 orders had unusable dates; 45 had zero quantity; 35 had non-positive unit prices. These 110 transaction rows were excluded from transaction-level analytics because their values could not be repaired safely from the available schema.
- 100 customer ages were missing and were handled with median imputation in the modeling pipeline.
- City and product-category text inconsistencies were normalized with trim/case mapping.
- Missing order payment method was assigned `Unknown`; delivery days used median imputation for modeling.
- Exact duplicates and orphan foreign keys were not found.

## Churn Modeling

Features are created only from orders on or before **31 May 2026**. Churn is `1` when the customer makes no purchase during **1 June 2026–31 August 2026**.

Compared models:

- Logistic Regression
- Random Forest
- Feed-Forward Neural Network (32 → 16 → 1 via `MLPClassifier`)

The supplied test split uses `random_state=42` and stratification. The final ML model is selected using F1-score with ROC-AUC as a secondary reference.

## Sentiment Analysis

Review labels are derived from rating:

- 1–2 → Negative
- 3 → Neutral
- 4–5 → Positive

Text is cleaned, converted with TF-IDF word unigrams/bigrams, and classified using Logistic Regression.

**Dataset limitation:** the sentiment target is derived directly from star ratings, so the model learns language patterns associated with the rating-derived labels rather than independently human-annotated sentiment.

## Run

```bash
pip install -r requirements.txt
python hackathon_solution.py
streamlit run app.py
```

## Files

- `ecommerce_hackathon.db`
- `hackathon_solution.py`
- `business_queries.sql`
- `churn_model.joblib`
- `churn_nn_model.joblib`
- `sentiment_model.joblib`
- `app.py`
- `requirements.txt`
- `README.md`
