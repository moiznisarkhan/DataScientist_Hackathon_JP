# E-Commerce Hackathon — Verified Results

## A. Data Quality

| Issue | Finding | Decision |
|---|---:|---|
| customers | 8,000 | retained |
| products | 1,000 | retained |
| orders | 65,000 raw / 64,890 valid | 110 invalid transaction rows excluded |
| reviews | 26,000 raw / 25,885 usable for NLP | unusable rating/text rows excluded |
| Missing customer age | 100 | median imputation = 31 for modeling |
| Missing order payment method | 55 | `Unknown` |
| Missing delivery days | 40 | median imputation for modeling |
| Missing product brand | 20 | `Unknown` |
| Invalid order dates | 30 | excluded from time-based analytics |
| Zero/negative quantities | 45 | excluded |
| Negative/zero order prices | 35 | excluded |
| Invalid review ratings | 25 | excluded from NLP |
| Missing review text | 90 | excluded from NLP |
| Invalid review dates | 16 | coerced to missing; date not used for sentiment |
| Exact duplicates | 0 | no deletion |
| Foreign-key orphan records | 0 | no deletion |

## B. SQL Business Results

### Net Revenue
**Rs. 865,169,922.03**

Definition: transaction revenue = `quantity × unit_price × (1 - discount)`; invalid transaction rows are excluded and returned orders are excluded from realized net revenue.

### Top 10 Customers

| Customer | City | Orders | Spending (Rs.) |
|---|---|---:|---:|
| Shahzaib Abbasi | Lahore | 37 | 1,537,800.62 |
| Zoya Awan | Lahore | 50 | 1,515,401.07 |
| Ahmed Qureshi | Lahore | 27 | 1,452,407.09 |
| Kiran Farooq | Karachi | 57 | 1,353,157.41 |
| Laiba Khan | Karachi | 40 | 1,340,945.41 |
| Adeel Khan | Karachi | 38 | 1,333,846.74 |
| Fahad Khan | Islamabad | 51 | 1,252,693.65 |
| Bilal Awan | Karachi | 45 | 1,248,563.98 |
| Imran Qureshi | Karachi | 44 | 1,247,069.53 |
| Danish Awan | Faisalabad | 43 | 1,240,877.18 |

### Category Revenue

| Category | Net Revenue (Rs.) | Orders | Quantity |
|---|---:|---:|---:|
| Electronics | 604,463,146.49 | 10,615 | 14,861 |
| Home & Kitchen | 67,176,614.34 | 7,117 | 10,047 |
| Fashion | 55,323,743.24 | 10,859 | 15,221 |
| Sports & Fitness | 51,472,454.53 | 6,654 | 9,352 |
| Beauty | 44,284,735.44 | 11,009 | 15,302 |
| Baby & Kids | 27,304,584.17 | 3,533 | 5,046 |
| Groceries | 7,835,444.02 | 5,942 | 8,303 |
| Books & Stationery | 7,309,199.81 | 4,819 | 6,673 |

### Top 5 Products

| Product | Category | Net Revenue (Rs.) |
|---|---|---:|
| Samsung Headphones Classic | Electronics | 105,698,173.53 |
| Anker Headphones Classic | Electronics | 66,013,975.20 |
| Dawlance LED TV Premium | Electronics | 19,030,700.98 |
| Infinix LED TV 2026 | Electronics | 15,654,003.13 |
| Dawlance Router Eco | Electronics | 14,176,597.19 |

## C. Business Insights

1. **Electronics concentration is extreme:** Electronics contributes about **69.9% of net revenue**. This creates a major growth opportunity but also concentration risk: inventory, pricing and promotion decisions in Electronics can materially move overall company performance.

2. **Karachi is the largest revenue market:** Karachi contributes about **26.3% of net revenue**, ahead of Lahore. Management can prioritize assortment, delivery capacity and customer retention programs in these high-contribution markets.

3. **Fashion is a return-risk category:** Fashion has an **11.25% return rate**, the highest of all categories, while Groceries and Books & Stationery are around 2%. The business should investigate product sizing, descriptions, quality expectations and fulfillment for Fashion before increasing acquisition spend aggressively there.

4. **July 2026 was the peak month in the supplied 2026 period:** net revenue reached **Rs. 76.61M**, up about **39.7% from May**, before declining about **11.6% in August**. This suggests the business should investigate what drove July demand and whether the August slowdown was seasonal, promotion-related or operational.

## D. Churn Model

The churn dataset contains **6,535 customers** who purchased on or before 31 May 2026.

- Churned: **3,206**
- Not churned: **3,329**
- Churn rate: **49.06%**
- Test set: **1,307 customers**

Features:

- total_orders
- total_spending
- average_order_value
- days_since_last_order
- return_rate
- average_delivery_days
- age
- membership_type

### Verified Test Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7169 | 0.6570 | **0.8846** | **0.7540** | **0.7956** |
| Logistic Regression | 0.7154 | 0.6615 | 0.8596 | 0.7476 | 0.7876 |
| Feed-Forward NN | 0.7123 | **0.6760** | 0.7941 | 0.7303 | 0.7811 |

### Final Model

**Random Forest** is the final ML model because it gives the strongest F1-score and ROC-AUC among the two required ML models while also achieving the highest churn recall. The difference is not huge, so the model should be described as a practical classifier rather than a high-confidence production predictor.

Random Forest feature importance is dominated by:

1. total_orders — 42.1%
2. days_since_last_order — 26.6%
3. total_spending — 14.3%
4. return_rate — 7.9%

This supports a practical business interpretation: **relationship frequency and recency are more informative for churn than demographics or membership tier in this dataset.**

## E. Sentiment

Usable review records: **25,885**.

Label distribution:

- Positive: 18,421
- Neutral: 5,458
- Negative: 2,006

Verified test metrics using TF-IDF + Logistic Regression:

- Accuracy: **1.0000**
- Macro Precision: **1.0000**
- Macro Recall: **1.0000**
- Macro F1: **1.0000**

This perfect score should **not** be presented as proof of real-world sentiment intelligence. The labels are derived directly from the rating field and the supplied synthetic reviews are highly structured/template-driven. The appropriate limitation is that this is a rating-derived classifier on a synthetic/semi-template-like review corpus, not an independently human-annotated sentiment benchmark.

## F. Deep Learning Decision

The feed-forward network uses a compact **32 → 16 → 1** architecture through `MLPClassifier` with ReLU hidden layers and a binary output. It achieved F1 = **0.7303**, below Random Forest (0.7540) and Logistic Regression (0.7476).

For this small tabular feature set, the additional neural-network complexity did not produce better test performance. The final solution therefore uses Random Forest for churn and keeps the neural network as a comparative experiment.
