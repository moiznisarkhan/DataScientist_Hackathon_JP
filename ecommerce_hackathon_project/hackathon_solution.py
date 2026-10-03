from pathlib import Path
import re
import sqlite3
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "ecommerce_hackathon.db"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# A. DATABASE INSPECTION, CLEANING & SQL VIEWS
# ============================================================

conn = sqlite3.connect(DB_PATH)

TABLES = ["customers", "products", "orders", "reviews"]

print("DATABASE TABLES")
print(pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
))

for table in TABLES:
    info = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
    count = pd.read_sql_query(f"SELECT COUNT(*) AS rows FROM {table}", conn).iloc[0, 0]
    print(f"\n{table}: {count:,} rows")
    print("Columns:", info["name"].tolist())

# Load raw tables for detailed quality checks.
customers = pd.read_sql_query("SELECT * FROM customers", conn)
products = pd.read_sql_query("SELECT * FROM products", conn)
orders = pd.read_sql_query("SELECT * FROM orders", conn)
reviews = pd.read_sql_query("SELECT * FROM reviews", conn)

# ---------- Data-quality summary ----------

def missing_summary(df, name):
    result = df.isna().sum().to_frame("missing")
    result["missing_pct"] = (result["missing"] / len(df) * 100).round(3)
    result.insert(0, "table", name)
    return result.reset_index(names="column")

quality_summary = pd.concat(
    [missing_summary(customers, "customers"),
     missing_summary(products, "products"),
     missing_summary(orders, "orders"),
     missing_summary(reviews, "reviews")],
    ignore_index=True
)

print("\nMISSING VALUE SUMMARY")
print(quality_summary[quality_summary["missing"] > 0].to_string(index=False))
quality_summary.to_csv(OUTPUT_DIR / "data_quality_missing_values.csv", index=False)

# Dates / numeric checks.
orders["order_date_parsed"] = pd.to_datetime(orders["order_date"], errors="coerce")
reviews["review_date_parsed"] = pd.to_datetime(reviews["review_date"], errors="coerce")
customers["signup_date_parsed"] = pd.to_datetime(customers["signup_date"], errors="coerce")

print("\nIMPORTANT QUALITY ISSUES")
print("Customer duplicate IDs:", customers["customer_id"].duplicated().sum())
print("Product duplicate IDs:", products["product_id"].duplicated().sum())
print("Order duplicate IDs:", orders["order_id"].duplicated().sum())
print("Review duplicate IDs:", reviews["review_id"].duplicated().sum())
print("Invalid order dates:", orders["order_date_parsed"].isna().sum())
print("Zero/negative quantities:", (orders["quantity"] <= 0).sum())
print("Zero/negative order prices:", (orders["unit_price"] <= 0).sum())
print("Missing customer age:", customers["age"].isna().sum())
print("Missing payment method:", orders["payment_method"].isna().sum())
print("Missing delivery days:", orders["delivery_days"].isna().sum())
print("Missing product brand:", products["brand"].isna().sum())
print("Invalid review ratings:", (~reviews["rating"].between(1, 5)).sum())
print("Missing review text:", reviews["review_text"].isna().sum())
print("Invalid review dates:", reviews["review_date_parsed"].isna().sum())

# Foreign-key integrity.
print("\nFOREIGN KEY ORPHANS")
for q in [
    "SELECT COUNT(*) AS n FROM orders o LEFT JOIN customers c ON o.customer_id=c.customer_id WHERE c.customer_id IS NULL",
    "SELECT COUNT(*) AS n FROM orders o LEFT JOIN products p ON o.product_id=p.product_id WHERE p.product_id IS NULL",
    "SELECT COUNT(*) AS n FROM reviews r LEFT JOIN customers c ON r.customer_id=c.customer_id WHERE c.customer_id IS NULL",
    "SELECT COUNT(*) AS n FROM reviews r LEFT JOIN products p ON r.product_id=p.product_id WHERE p.product_id IS NULL",
]:
    print(pd.read_sql_query(q, conn).iloc[0, 0])

# ---------- Cleaning rules ----------
# We do NOT mutate the source database. We clean through views / DataFrames.
# 1) Exact duplicates: none found, so no deletion needed.
# 2) City/category capitalization/spacing is normalized.
# 3) Customer age: median imputation later in model pipeline.
# 4) Product brand: use "Unknown" for missing descriptive values.
# 5) Payment method: use "Unknown" for missing descriptive values.
# 6) Delivery days: median imputation in modeling; revenue does not depend on this field.
# 7) Order rows with impossible date/quantity/unit_price are excluded from financial and churn analysis.
#    They cannot be repaired safely without inventing transaction history.
# 8) Review rows with unusable text or invalid rating are excluded from NLP.

# Canonical category names.
category_map = {
    "FASHION": "Fashion",
    "ELECTRONICS": "Electronics",
    "HOME & KITCHEN": "Home & Kitchen",
    "BEAUTY": "Beauty",
    "GROCERIES": "Groceries",
    "SPORTS & FITNESS": "Sports & Fitness",
    "BOOKS & STATIONERY": "Books & Stationery",
    "BABY & KIDS": "Baby & Kids",
}

city_map = {
    "KARACHI": "Karachi", "LAHORE": "Lahore", "ISLAMABAD": "Islamabad",
    "RAWALPINDI": "Rawalpindi", "FAISALABAD": "Faisalabad", "MULTAN": "Multan",
    "PESHAWAR": "Peshawar", "HYDERABAD": "Hyderabad", "GUJRANWALA": "Gujranwala",
    "QUETTA": "Quetta", "SIALKOT": "Sialkot", "BAHAWALPUR": "Bahawalpur",
    "SARGODHA": "Sargodha"
}

customers_clean = customers.copy()
customers_clean["city"] = customers_clean["city"].astype(str).str.strip().str.upper().map(city_map).fillna(
    customers_clean["city"].astype(str).str.strip().str.title()
)
customers_clean["age"] = customers_clean["age"].fillna(customers_clean["age"].median())

products_clean = products.copy()
products_clean["category"] = products_clean["category"].astype(str).str.strip().str.upper().map(category_map).fillna(
    products_clean["category"].astype(str).str.strip().str.title()
)
products_clean["brand"] = products_clean["brand"].fillna("Unknown")

orders_clean = orders.copy()
orders_clean["order_date"] = pd.to_datetime(orders_clean["order_date"], errors="coerce")
orders_clean["payment_method"] = orders_clean["payment_method"].fillna("Unknown")
orders_clean["delivery_days"] = orders_clean["delivery_days"].fillna(orders_clean["delivery_days"].median())
orders_clean = orders_clean[
    orders_clean["order_date"].notna()
    & (orders_clean["quantity"] > 0)
    & (orders_clean["unit_price"] > 0)
].copy()
orders_clean["line_revenue"] = (
    orders_clean["quantity"]
    * orders_clean["unit_price"]
    * (1 - orders_clean["discount"])
)

reviews_clean = reviews.copy()
reviews_clean["review_date"] = pd.to_datetime(reviews_clean["review_date"], errors="coerce")
reviews_clean["review_text"] = reviews_clean["review_text"].fillna("").astype(str).str.strip()
reviews_clean = reviews_clean[
    reviews_clean["rating"].between(1, 5)
    & reviews_clean["review_text"].ne("")
].copy()

print("\nCLEANING RESULT")
print("Valid orders retained:", len(orders_clean), "of", len(orders))
print("Rows removed from order analytics:", len(orders) - len(orders_clean))
print("Usable review rows retained for NLP:", len(reviews_clean), "of", len(reviews))

# ---------- Temporary SQL views ----------
conn.executescript("""
DROP VIEW IF EXISTS clean_orders;
DROP VIEW IF EXISTS clean_customers;
DROP VIEW IF EXISTS clean_products;

CREATE TEMP VIEW clean_orders AS
SELECT
    order_id,
    customer_id,
    product_id,
    date(order_date) AS order_date,
    quantity,
    unit_price,
    discount,
    COALESCE(payment_method, 'Unknown') AS payment_method,
    delivery_days,
    returned,
    quantity * unit_price * (1 - discount) AS line_revenue
FROM orders
WHERE date(order_date) IS NOT NULL
  AND quantity > 0
  AND unit_price > 0;

CREATE TEMP VIEW clean_customers AS
SELECT
    customer_id,
    customer_name,
    age,
    CASE UPPER(TRIM(city))
        WHEN 'KARACHI' THEN 'Karachi'
        WHEN 'LAHORE' THEN 'Lahore'
        WHEN 'ISLAMABAD' THEN 'Islamabad'
        WHEN 'RAWALPINDI' THEN 'Rawalpindi'
        WHEN 'FAISALABAD' THEN 'Faisalabad'
        WHEN 'MULTAN' THEN 'Multan'
        WHEN 'PESHAWAR' THEN 'Peshawar'
        WHEN 'HYDERABAD' THEN 'Hyderabad'
        WHEN 'GUJRANWALA' THEN 'Gujranwala'
        WHEN 'QUETTA' THEN 'Quetta'
        WHEN 'SIALKOT' THEN 'Sialkot'
        WHEN 'BAHAWALPUR' THEN 'Bahawalpur'
        WHEN 'SARGODHA' THEN 'Sargodha'
        ELSE TRIM(city)
    END AS city,
    signup_date,
    membership_type,
    email,
    phone
FROM customers;

CREATE TEMP VIEW clean_products AS
SELECT
    product_id,
    product_name,
    CASE UPPER(TRIM(category))
        WHEN 'FASHION' THEN 'Fashion'
        WHEN 'ELECTRONICS' THEN 'Electronics'
        WHEN 'HOME & KITCHEN' THEN 'Home & Kitchen'
        WHEN 'BEAUTY' THEN 'Beauty'
        WHEN 'GROCERIES' THEN 'Groceries'
        WHEN 'SPORTS & FITNESS' THEN 'Sports & Fitness'
        WHEN 'BOOKS & STATIONERY' THEN 'Books & Stationery'
        WHEN 'BABY & KIDS' THEN 'Baby & Kids'
        ELSE TRIM(category)
    END AS category,
    COALESCE(brand, 'Unknown') AS brand,
    unit_price,
    stock_quantity
FROM products;
""")

# ============================================================
# B. REQUIRED SQL BUSINESS ANALYSIS
# ============================================================

sql_queries = {
    "total_net_revenue": """
        SELECT ROUND(SUM(line_revenue), 2) AS total_net_revenue
        FROM clean_orders
        WHERE returned = 0;
    """,
    "top_10_customers": """
        SELECT
            c.customer_name,
            c.city,
            COUNT(DISTINCT o.order_id) AS number_of_orders,
            ROUND(SUM(o.line_revenue), 2) AS total_spending
        FROM clean_orders o
        JOIN clean_customers c ON c.customer_id = o.customer_id
        WHERE o.returned = 0
        GROUP BY c.customer_id, c.customer_name, c.city
        ORDER BY total_spending DESC
        LIMIT 10;
    """,
    "category_analysis": """
        SELECT
            p.category,
            ROUND(SUM(o.line_revenue), 2) AS net_revenue,
            COUNT(DISTINCT o.order_id) AS order_count,
            SUM(o.quantity) AS quantity_sold
        FROM clean_orders o
        JOIN clean_products p ON p.product_id = o.product_id
        WHERE o.returned = 0
        GROUP BY p.category
        ORDER BY net_revenue DESC;
    """,
    "monthly_revenue": """
        SELECT
            strftime('%Y-%m', order_date) AS month,
            ROUND(SUM(line_revenue), 2) AS net_revenue
        FROM clean_orders
        WHERE returned = 0
        GROUP BY month
        ORDER BY month;
    """,
    "top_5_products": """
        SELECT
            p.product_name,
            p.category,
            ROUND(SUM(o.line_revenue), 2) AS net_revenue,
            SUM(o.quantity) AS quantity_sold,
            COUNT(DISTINCT o.order_id) AS order_count
        FROM clean_orders o
        JOIN clean_products p ON p.product_id = o.product_id
        WHERE o.returned = 0
        GROUP BY p.product_id, p.product_name, p.category
        ORDER BY net_revenue DESC
        LIMIT 5;
    """
}

sql_results = {}
for name, query in sql_queries.items():
    sql_results[name] = pd.read_sql_query(query, conn)
    sql_results[name].to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    print(f"\n{name.upper()}")
    print(sql_results[name].to_string(index=False))

# Dashboard KPI queries.
net_revenue = sql_results["total_net_revenue"].iloc[0, 0]
total_valid_orders = pd.read_sql_query("SELECT COUNT(*) AS n FROM clean_orders", conn).iloc[0, 0]
total_customers = pd.read_sql_query("SELECT COUNT(*) AS n FROM clean_customers", conn).iloc[0, 0]

print("\nDASHBOARD KPIs")
print("Net revenue:", f"{net_revenue:,.2f}")
print("Valid orders:", f"{total_valid_orders:,}")
print("Customers:", f"{total_customers:,}")

# ============================================================
# C. EDA
# ============================================================

monthly = sql_results["monthly_revenue"].copy()
category = sql_results["category_analysis"].copy()
city = pd.read_sql_query("""
    SELECT
        c.city,
        ROUND(SUM(o.line_revenue), 2) AS net_revenue
    FROM clean_orders o
    JOIN clean_customers c ON c.customer_id = o.customer_id
    WHERE o.returned = 0
    GROUP BY c.city
    ORDER BY net_revenue DESC;
""", conn)
return_category = pd.read_sql_query("""
    SELECT
        p.category,
        COUNT(*) AS orders,
        SUM(o.returned) AS returned_orders,
        ROUND(100.0 * SUM(o.returned) / COUNT(*), 2) AS return_rate_pct
    FROM clean_orders o
    JOIN clean_products p ON p.product_id = o.product_id
    GROUP BY p.category
    ORDER BY return_rate_pct DESC;
""", conn)

plt.figure(figsize=(12, 5))
sns.lineplot(data=monthly, x="month", y="net_revenue")
plt.xticks(rotation=60)
plt.title("Monthly Net Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Net Revenue")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_revenue.png", dpi=150)
plt.show()

plt.figure(figsize=(9, 5))
plt.barh(category["category"], category["net_revenue"])
plt.gca().invert_yaxis()
plt.title("Category Wise Net Revenue")
plt.xlabel("Net Revenue")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "category_revenue.png", dpi=150)
plt.show()

plt.figure(figsize=(9, 5))
plt.barh(city["city"], city["net_revenue"])
plt.gca().invert_yaxis()
plt.title("City Wise Net Revenue")
plt.xlabel("Net Revenue")
plt.ylabel("City")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "city_revenue.png", dpi=150)
plt.show()

plt.figure(figsize=(9, 5))
plt.barh(return_category["category"], return_category["return_rate_pct"])
plt.gca().invert_yaxis()
plt.title("Return Rate by Product Category")
plt.xlabel("Return Rate (%)")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "return_rate_category.png", dpi=150)
plt.show()

# ============================================================
# D. CUSTOMER CHURN DATASET
# ============================================================

churn_query = """
WITH valid_orders AS (
    SELECT
        order_id,
        customer_id,
        date(order_date) AS order_date,
        quantity,
        unit_price,
        discount,
        delivery_days,
        returned,
        quantity * unit_price * (1 - discount) AS line_revenue
    FROM orders
    WHERE date(order_date) IS NOT NULL
      AND quantity > 0
      AND unit_price > 0
),
pre_cutoff AS (
    SELECT *
    FROM valid_orders
    WHERE order_date <= '2026-05-31'
),
target_window AS (
    SELECT DISTINCT customer_id
    FROM valid_orders
    WHERE order_date BETWEEN '2026-06-01' AND '2026-08-31'
)
SELECT
    c.customer_id,
    c.age,
    c.membership_type,
    COUNT(p.order_id) AS total_orders,
    COALESCE(
        SUM(CASE WHEN p.returned = 0 THEN p.line_revenue ELSE 0 END),
        0
    ) AS total_spending,
    AVG(CASE WHEN p.returned = 0 THEN p.line_revenue END) AS average_order_value,
    CAST(julianday('2026-05-31') - julianday(MAX(p.order_date)) AS INTEGER)
        AS days_since_last_order,
    AVG(p.returned) AS return_rate,
    AVG(p.delivery_days) AS average_delivery_days,
    CASE
        WHEN tw.customer_id IS NULL THEN 1
        ELSE 0
    END AS churn
FROM customers c
JOIN pre_cutoff p ON c.customer_id = p.customer_id
LEFT JOIN target_window tw ON c.customer_id = tw.customer_id
GROUP BY c.customer_id, c.age, c.membership_type;
"""

churn_df = pd.read_sql_query(churn_query, conn)
churn_df.to_csv(OUTPUT_DIR / "churn_features.csv", index=False)

print("\nCHURN DATASET")
print("Shape:", churn_df.shape)
print("Target distribution:")
print(churn_df["churn"].value_counts())
print("Churn rate:", round(churn_df["churn"].mean() * 100, 2), "%")

FEATURES = [
    "total_orders",
    "total_spending",
    "average_order_value",
    "days_since_last_order",
    "return_rate",
    "average_delivery_days",
    "age",
    "membership_type",
]
TARGET = "churn"

X = churn_df[FEATURES]
y = churn_df[TARGET]

numeric_features = [
    "total_orders", "total_spending", "average_order_value",
    "days_since_last_order", "return_rate", "average_delivery_days", "age"
]
categorical_features = ["membership_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        drop="first",
                        sparse_output=False
                    )
                )
            ]),
            categorical_features
        )
    ]
)

# Parameters selected from small validation experiments on the training split.
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        LogisticRegression(
            C=0.05,
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED
        )
    )
])

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_split=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        )
    )
])

nn_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=32,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.20,
            n_iter_no_change=30,
            max_iter=500,
            random_state=SEED
        )
    )
])

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
    "Feed-Forward Neural Network": nn_model,
}

model_results = []
model_predictions = {}
model_probabilities = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    model_predictions[name] = pred
    model_probabilities[name] = prob

    model_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    })

model_results_df = pd.DataFrame(model_results).sort_values("F1 Score", ascending=False)
model_results_df.to_csv(OUTPUT_DIR / "churn_model_comparison.csv", index=False)

print("\nCHURN MODEL COMPARISON")
print(model_results_df.to_string(index=False))

for name, pred in model_predictions.items():
    print("\n" + "=" * 80)
    print(name)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("Classification Report:")
    print(classification_report(y_test, pred, digits=4, zero_division=0))

# ROC curve.
plt.figure(figsize=(8, 6))
for name, prob in model_probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "--")
plt.title("Churn ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "churn_roc_curves.png", dpi=150)
plt.show()

# Select final ML model by F1-score, using ROC-AUC as a secondary reference.
best_ml_name = (
    model_results_df[model_results_df["Model"].isin(["Logistic Regression", "Random Forest"])]
    .sort_values(["F1 Score", "ROC-AUC"], ascending=False)
    .iloc[0]["Model"]
)
final_churn_model = models[best_ml_name]
joblib.dump(final_churn_model, BASE_DIR / "churn_model.joblib")
print("Final churn model:", best_ml_name)

# Save NN separately for completeness.
joblib.dump(nn_model, BASE_DIR / "churn_nn_model.joblib")

# Feature importance for the Random Forest.
rf_preprocessor = random_forest_model.named_steps["preprocessor"]
rf_model = random_forest_model.named_steps["model"]
rf_features = rf_preprocessor.get_feature_names_out()
rf_importance = pd.DataFrame({
    "feature": rf_features,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)
rf_importance.to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv", index=False)
print("\nRandom Forest feature importance")
print(rf_importance.to_string(index=False))

# ============================================================
# E. NLP SENTIMENT ANALYSIS
# ============================================================
# ============================================================
# E. NLP SENTIMENT ANALYSIS
# ============================================================
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
import sklearn


def clean_review_text(text):
    """Keep this function IDENTICAL in app.py. Contractions are expanded first so 'isn't good' keeps its 'not'."""
    text = str(text).lower().replace("\u2019", "'")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


sentiment_df = reviews_clean.copy()
sentiment_df["clean_text"] = sentiment_df["review_text"].apply(clean_review_text)
sentiment_df = sentiment_df[sentiment_df["clean_text"].ne("")].copy()

sentiment_df["sentiment"] = np.select(
    [sentiment_df["rating"].between(1, 2), sentiment_df["rating"].eq(3), sentiment_df["rating"].between(4, 5)],
    ["Negative", "Neutral", "Positive"],
    default="Neutral"
)

print("\nSENTIMENT DATASET")
print(sentiment_df["sentiment"].value_counts())

# ---------- E1. Can the review text really predict the label? ----------
unique_texts = sentiment_df["clean_text"].nunique()
dup_share = 1 - unique_texts / len(sentiment_df)
conflicting = int((sentiment_df.groupby("clean_text")["sentiment"].nunique() > 1).sum())
vocab = {w for t in sentiment_df["clean_text"] for w in t.split()}
print(f"\nReviews: {len(sentiment_df):,} | unique texts: {unique_texts:,} ({dup_share:.0%} repeated) | "
      f"vocabulary: {len(vocab):,} words | avg length: {sentiment_df['clean_text'].str.split().str.len().mean():.1f} words")
if dup_share > 0.5:
    print("WARNING: most reviews are repeated texts. The model only knows these phrasings, and a random "
          "train/test split would look far better than real-world accuracy, so evaluation below uses unique texts only.")
if conflicting:
    print(f"WARNING: {conflicting} identical texts carry different sentiment labels (label noise).")

# ---------- E2. Honest evaluation + tuning on UNIQUE texts ----------
eval_df = sentiment_df.drop_duplicates("clean_text")
if eval_df["sentiment"].value_counts().min() < 10:
    eval_df = sentiment_df  # too few unique texts per class to split honestly

text_train, text_test, y_text_train, y_text_test = train_test_split(
    eval_df["clean_text"], eval_df["sentiment"],
    test_size=0.20, random_state=SEED, stratify=eval_df["sentiment"]
)

base_sentiment_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),   # stop words kept on purpose: "not" matters
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
])
search = GridSearchCV(
    base_sentiment_model,
    {"tfidf__min_df": [1, 2], "model__C": [0.5, 2, 10]},
    scoring="f1_macro",
    cv=min(5, int(y_text_train.value_counts().min())),
    n_jobs=-1
)
search.fit(text_train, y_text_train)
print("\nBest settings:", search.best_params_, "| CV macro-F1:", round(search.best_score_, 4))

sent_pred = search.predict(text_test)
sentiment_metrics = pd.DataFrame([{
    "Accuracy": accuracy_score(y_text_test, sent_pred),
    "Precision (Macro)": precision_score(y_text_test, sent_pred, average="macro", zero_division=0),
    "Recall (Macro)": recall_score(y_text_test, sent_pred, average="macro", zero_division=0),
    "F1 Score (Macro)": f1_score(y_text_test, sent_pred, average="macro", zero_division=0),
    "F1 Score (Weighted)": f1_score(y_text_test, sent_pred, average="weighted", zero_division=0)
}])
sentiment_metrics.to_csv(OUTPUT_DIR / "sentiment_metrics.csv", index=False)
print("\nSENTIMENT METRICS (hold-out, unique texts)")
print(sentiment_metrics.to_string(index=False))
print("\nSENTIMENT CLASSIFICATION REPORT")
print(classification_report(y_text_test, sent_pred, digits=4, zero_division=0))
print("Confusion matrix (rows = true, columns = predicted):", list(search.classes_))
print(confusion_matrix(y_text_test, sent_pred, labels=search.classes_))

# Final model: same settings, trained on ALL usable reviews.
sentiment_model = clone(search.best_estimator_).fit(sentiment_df["clean_text"], sentiment_df["sentiment"])
joblib.dump(sentiment_model, BASE_DIR / "sentiment_model.joblib")
print("Saved sentiment_model.joblib (trained with scikit-learn", sklearn.__version__ + ")")

# ---------- E3. What did the model learn? ----------
vec = sentiment_model.named_steps["tfidf"]
clf = sentiment_model.named_steps["model"]
names = vec.get_feature_names_out()
print("\nTOP WORDS PER CLASS (if a clearly negative word shows up under Positive, the labels are noisy)")
for i, cls in enumerate(clf.classes_):
    print(f"  {cls}: ", ", ".join(names[np.argsort(clf.coef_[i])[-10:][::-1]]))

# ---------- E4. Smoke test with free-text phrasing the training reviews may not contain ----------
smoke = [
    ("The product is of good quality", "Positive"), ("Very satisfied, excellent quality and good value for money.", "Positive"),
    ("I love it, works perfectly", "Positive"), ("Fast delivery and great packaging", "Positive"),
    ("The product is of bad quality", "Negative"), ("Terrible quality, disappointed and would not buy again.", "Negative"),
    ("This is not good at all", "Negative"), ("It isn't worth the money", "Negative"),
    ("Arrived damaged and support never replied", "Negative"), ("Worst purchase ever", "Negative"),
    ("Nothing special, the product is okay but average.", "Neutral"), ("It is fine, does the job", "Neutral"),
]
smoke_df = pd.DataFrame(smoke, columns=["text", "expected"])
smoke_df["predicted"] = sentiment_model.predict(smoke_df["text"].apply(clean_review_text))
smoke_df["unseen_words"] = smoke_df["text"].apply(
    lambda t: ", ".join(w for w in clean_review_text(t).split() if w not in vec.vocabulary_))
smoke_df.to_csv(OUTPUT_DIR / "sentiment_smoke_test.csv", index=False)
print(f"\nSMOKE TEST: {(smoke_df.expected == smoke_df.predicted).mean():.0%} correct")
print(smoke_df[smoke_df.expected != smoke_df.predicted].to_string(index=False))
print("(unseen_words = words that never appeared in the training reviews; the model cannot use them)")