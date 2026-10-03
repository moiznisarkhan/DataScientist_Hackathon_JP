from pathlib import Path
import re
import warnings
import sqlite3
import joblib
import sklearn
from sklearn.exceptions import NotFittedError
import numpy as np
import pandas as pd
import scipy.sparse as sp
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "ecommerce_hackathon.db"
CHURN_MODEL_PATH = BASE_DIR / "churn_model.joblib"
SENTIMENT_MODEL_PATH = BASE_DIR / "sentiment_model.joblib"

# One accent, two signal colours, warm neutrals. Nothing else.
PAPER, CARD, LINE = "#F7F8FA", "#FFFFFF", "#DDE4E9"
INK, MUTED = "#1F2933", "#66737F"
GREEN, CLAY, OCHRE, SLATE, SAGE = "#2F6F95", "#BF4E30", "#C99A2E", "#5B7A99", "#AFC1CC"

CITIES = {"Karachi": (24.86, 67.01), "Lahore": (31.52, 74.36), "Islamabad": (33.68, 73.05),
          "Rawalpindi": (33.60, 73.04), "Faisalabad": (31.42, 73.08), "Multan": (30.20, 71.47),
          "Peshawar": (34.01, 71.58), "Hyderabad": (25.39, 68.37), "Gujranwala": (32.16, 74.19),
          "Quetta": (30.18, 66.98), "Sialkot": (32.49, 74.53), "Bahawalpur": (29.40, 71.68),
          "Sargodha": (32.08, 72.67)}

st.set_page_config(page_title="E-Commerce Intelligence", page_icon="🛒", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: %PAPER%; color: %INK%; }
.stApp label, .stApp p, .stApp li { color: %INK%; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1180px; }
[data-testid="stSidebar"] { background: #F0F3F5; border-right: 1px solid %LINE%; }

.eyebrow { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: %MUTED%; font-weight: 600; margin-top: 2.2rem; }
h1.title { font-family: 'Fraunces', serif; font-weight: 600; font-size: clamp(1.9rem, 4vw, 2.8rem); line-height: 1.1; margin: .3rem 0 .2rem; letter-spacing: -.02em; }
.lede { color: %MUTED%; margin: 0 0 .5rem; max-width: 60ch; }
h3.h { font-family: 'Fraunces', serif; font-weight: 600; font-size: clamp(1.15rem, 2.2vw, 1.45rem); line-height: 1.25; margin: .15rem 0 .1rem; }
.sub { color: %MUTED%; font-size: .88rem; margin: 0 0 .6rem; }
.headline { font-family: 'Fraunces', serif; font-size: clamp(1.4rem, 3vw, 2.1rem); line-height: 1.25; margin: 1rem 0 1.4rem; max-width: 30ch; }

.kgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: .5rem; }
.k { background: %CARD%; border: 1px solid %LINE%; border-radius: 10px; padding: 1rem 1.1rem .6rem; }
.kl { font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: %MUTED%; font-weight: 600; }
.kv { font-family: 'Fraunces', serif; font-size: clamp(1.5rem, 2.6vw, 1.9rem); margin: .25rem 0 .1rem; }
.ks { font-size: .82rem; color: %MUTED%; } .ks.bad { color: %CLAY%; } .ks.good { color: %GREEN%; }

ol.take { padding-left: 1.2rem; margin: .3rem 0; } ol.take li { margin: .45rem 0; line-height: 1.5; }
.lb { display: grid; grid-template-columns: 1.6rem 1fr auto; gap: .2rem .6rem; align-items: baseline; padding: .6rem 0; border-bottom: 1px solid %LINE%; }
.lb .n { font-family: 'Fraunces', serif; color: %MUTED%; } .lb .v { font-variant-numeric: tabular-nums; }
.lb .cat { color: %MUTED%; font-size: .8rem; grid-column: 2 / 4; }
.panel { background: %CARD%; border: 1px solid %LINE%; border-radius: 10px; padding: 1.1rem 1.2rem; }
.meter { position: relative; height: 10px; border-radius: 6px; margin: 1.1rem 0 .4rem;
  background: linear-gradient(90deg, %SAGE% 0 33%, #E6CF8E 33% 66%, #E0A090 66% 100%); }
.meter i { position: absolute; top: -5px; width: 4px; height: 20px; background: %INK%; border-radius: 2px; transform: translateX(-2px); }
.big { font-family: 'Fraunces', serif; font-size: clamp(2.6rem, 6vw, 3.6rem); line-height: 1; }
.pb { margin: .5rem 0; } .pb .row { display: flex; justify-content: space-between; font-size: .9rem; }
.pb .t { height: 8px; background: %LINE%; border-radius: 5px; overflow: hidden; margin-top: .2rem; } .pb .t i { display: block; height: 100%; }

.stTabs [data-baseweb="tab-list"] { gap: 1.2rem; border-bottom: 1px solid %LINE%; }
.stTabs [data-baseweb="tab"] { padding: .5rem 0; font-weight: 500; }
.stTabs [aria-selected="true"] { color: %GREEN%; }
.stTabs [data-baseweb="tab-highlight"] { background: %GREEN%; }
button[kind="primary"], [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] { background: %GREEN%; border: 0; color: #fff; }
[data-testid="stForm"] { background: %CARD%; border: 1px solid %LINE%; border-radius: 10px; }
[data-testid="stExpander"] { background: %CARD%; border: 1px solid %LINE%; border-radius: 10px; }

@media (max-width: 768px) {
  .block-container { padding: 1rem .9rem 3rem; }
  .eyebrow { margin-top: 1.6rem; }
  .kgrid { grid-template-columns: repeat(2, 1fr); gap: 8px; }
  .k { padding: .8rem .8rem .4rem; }
  .headline { max-width: none; }
}

/* ---- theme-proof overrides: readable even if Streamlit runs in dark mode ---- */
:root, .stApp { color-scheme: light; }
.stApp, .stApp [data-testid="stAppViewContainer"] { background: %PAPER% !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: %INK% !important; }
%M%, %M% p, %M% li, %M% b, %M% strong, %M% span { color: %INK% !important; }
%M% .sub, %M% .lede, %M% .eyebrow, %M% .kl, %M% .ks, %M% .lb .n, %M% .lb .cat { color: %MUTED% !important; }
%M% .ks.bad { color: %CLAY% !important; } %M% .ks.good { color: %GREEN% !important; }
%M% .big { color: %INK%; }
header[data-testid="stHeader"] *, [data-testid="stSidebar"], [data-testid="stSidebar"] * { color: %INK% !important; }
header[data-testid="stHeader"] svg { fill: %INK% !important; }
.stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stWidgetLabel"] *, .stApp label, .stApp label * { color: %INK% !important; }
.stApp input, .stApp textarea { color: %INK% !important; -webkit-text-fill-color: %INK% !important; caret-color: %INK%; background: %CARD% !important; }
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="textarea"], .stApp [data-baseweb="select"] > div { background: %CARD% !important; border-color: %LINE% !important; }
.stApp [data-baseweb="select"] *, .stApp [data-baseweb="select"] svg { color: %INK% !important; fill: %INK% !important; }
.stApp [data-baseweb="tag"] { background: %SAGE% !important; } .stApp [data-baseweb="tag"] * { color: %INK% !important; }
.stApp [data-testid="stNumberInput"] button { background: #EEF2F4 !important; } .stApp [data-testid="stNumberInput"] button * { color: %INK% !important; fill: %INK% !important; }
[data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: %INK% !important; }
.stApp [role="slider"] { background: %GREEN% !important; }
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul { background: %CARD% !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] li * { color: %INK% !important; }
.stApp .stTabs [data-baseweb="tab"], .stApp .stTabs [data-baseweb="tab"] p { color: %INK% !important; }
.stApp .stTabs [data-baseweb="tab"][aria-selected="true"] p { color: %GREEN% !important; }
.stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary * { color: %INK% !important; }
.stApp [data-testid="stBaseButton-secondary"], .stApp button[kind="secondary"] { background: %CARD% !important; border: 1px solid %LINE% !important; }
.stApp [data-testid="stBaseButton-secondary"] *, .stApp button[kind="secondary"] * { color: %INK% !important; }
.stApp [data-testid="stBaseButton-primary"], .stApp [data-testid="stBaseButton-primaryFormSubmit"], .stApp button[kind="primary"] { background: %GREEN% !important; border: 0 !important; }
.stApp [data-testid="stBaseButton-primary"] *, .stApp [data-testid="stBaseButton-primaryFormSubmit"] *, .stApp button[kind="primary"] * { color: #FFFFFF !important; }
.stApp [data-testid="stAlert"], .stApp [data-testid="stAlert"] * { color: %INK% !important; }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * { color: %MUTED% !important; }
</style>
"""
for token, value in {"M": '.stApp [data-testid="stMarkdownContainer"]', "PAPER": PAPER, "CARD": CARD, "LINE": LINE, "INK": INK, "MUTED": MUTED,
                     "GREEN": GREEN, "CLAY": CLAY, "SAGE": SAGE}.items():
    CSS = CSS.replace(f"%{token}%", value)
st.markdown(CSS, unsafe_allow_html=True)


# ---------------- helpers ----------------
def get_connection():
    return sqlite3.connect(DB_PATH)


def clean_review_text(text):
    """Must stay IDENTICAL to clean_review_text in hackathon_solution.py."""
    text = str(text).lower().replace("\u2019", "'")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fmt(v):
    if abs(v) >= 1e6:
        return f"Rs. {v / 1e6:,.2f}M"
    if abs(v) >= 1e3:
        return f"Rs. {v / 1e3:,.1f}K"
    return f"Rs. {v:,.0f}"


def show(fig):
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def style(fig, h=340):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=INK, size=12),
                      margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                      hoverlabel=dict(bgcolor=CARD, font_color=INK, bordercolor=LINE))
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE,
                     tickfont=dict(color=INK, size=12), title_font=dict(color=INK, size=12))
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor="rgba(0,0,0,0)",
                     tickfont=dict(color=INK, size=12), title_font=dict(color=INK, size=12))
    fig.update_traces(textfont=dict(color=INK, size=12))
    return fig


def spark(vals, color=GREEN):
    v = np.array(list(vals), float)
    if len(v) < 2:
        return ""
    v = (v - v.min()) / (np.ptp(v) or 1)
    pts = " ".join(f"{i * 120 / (len(v) - 1):.1f},{26 - x * 22:.1f}" for i, x in enumerate(v))
    return (f'<svg width="100%" height="28" viewBox="0 0 120 28" preserveAspectRatio="none">'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.8" points="{pts}"/></svg>')


def head(eyebrow, title, sub=""):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><h3 class="h">{title}</h3><p class="sub">{sub}</p>',
                unsafe_allow_html=True)


def page_header(title, lede):
    st.markdown(f'<div class="eyebrow" style="margin-top:0">E-commerce intelligence</div>'
                f'<h1 class="title">{title}</h1><p class="lede">{lede}</p>', unsafe_allow_html=True)


@st.cache_resource
def load_model(path):
    """Returns (model, mismatch). mismatch = (trained_version, running_version) if they differ, else None."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(path)
    mismatch = next(((w.message.original_sklearn_version, w.message.current_sklearn_version)
                     for w in caught if w.category.__name__ == "InconsistentVersionWarning"), None)
    return model, mismatch


def repair_model(model, mismatch):
    """Patch the two things that break when a model saved by a NEWER scikit-learn runs on an older one:
    1. TF-IDF weights ('idf vector is not fitted'): rebuild whichever of idf_ / _idf_diag is missing.
    2. LogisticRegression: newer versions store multi_class='deprecated', which older ones read as
       multinomial even for 2 classes, giving over-confident probabilities. Reset it to 'auto'."""
    if not mismatch:
        return model
    try:
        for _, step in getattr(model, "steps", [("model", model)]):
            tfidf = getattr(step, "_tfidf", None)
            if tfidf is None and step.__class__.__name__ == "TfidfTransformer":
                tfidf = step
            if tfidf is not None and not hasattr(tfidf, "idf_"):
                if "idf_" in tfidf.__dict__:
                    idf = np.asarray(tfidf.__dict__["idf_"]).ravel()
                    tfidf._idf_diag = sp.diags(idf, offsets=0, shape=(len(idf), len(idf)), format="csr", dtype=idf.dtype)
                elif hasattr(tfidf, "_idf_diag"):
                    tfidf.idf_ = np.asarray(tfidf._idf_diag.diagonal()).ravel()
            if step.__class__.__name__ == "LogisticRegression" and getattr(step, "multi_class", "deprecated") == "deprecated":
                step.multi_class = "auto"
    except Exception:
        pass
    return model


def final_estimator_name(model):
    steps = getattr(model, "steps", None)
    return (steps[-1][1] if steps else model).__class__.__name__


try:  # optional general-language check: pip install vaderSentiment
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None


def language_score(text):
    """General English sentiment score from -1 (negative) to +1 (positive); None if vaderSentiment is missing."""
    return None if _VADER is None else _VADER.polarity_scores(text)["compound"]


def model_view(model, clean_text, cls):
    """What the trained model actually saw: words it has never met, and the words that pushed it toward `cls`."""
    try:
        vec, clf = model.steps[0][1], model.steps[-1][1]
        X = vec.transform([clean_text]).tocoo()
        unseen = [w for w in clean_text.split() if w not in vec.vocabulary_]
        names, classes = vec.get_feature_names_out(), list(clf.classes_)
        i = classes.index(cls)
        coef = clf.coef_[i] if clf.coef_.shape[0] > 1 else (clf.coef_[0] if i == 1 else -clf.coef_[0])
        push = sorted(((names[j], v * coef[j]) for j, v in zip(X.col, X.data)), key=lambda t: -t[1])
        return X.nnz, unseen, [w for w, e in push if e > 0][:4]
    except Exception:
        return None, [], []


# ---------------- data ----------------
@st.cache_data(ttl=300)
def load_dashboard_data():
    conn = get_connection()
    o = pd.read_sql_query("SELECT * FROM orders", conn)
    c = pd.read_sql_query("SELECT * FROM customers", conn)
    p = pd.read_sql_query("SELECT * FROM products", conn)
    conn.close()
    o["order_date"] = pd.to_datetime(o["order_date"], errors="coerce")
    o = o[o.order_date.notna() & (o.quantity > 0) & (o.unit_price > 0)].copy()
    o["gross"] = o.quantity * o.unit_price * (1 - o.discount.fillna(0))
    c = c.drop_duplicates("customer_id").assign(city=lambda d: d.city.astype(str).str.strip().str.title())
    p = p.drop_duplicates("product_id").assign(category=lambda d: d.category.astype(str).str.strip().str.title())
    df = o.merge(c[["customer_id", "customer_name", "city"]], on="customer_id", how="left") \
          .merge(p[["product_id", "product_name", "category"]], on="product_id", how="left")
    for col in ["city", "category", "customer_name", "product_name"]:
        df[col] = df[col].fillna("Unknown")
    return df, len(c)


# ---------------- navigation ----------------
page = st.sidebar.radio("Navigate", ["Dashboard", "Churn Prediction", "Sentiment Analysis"])

# =====================================================================
if page == "Dashboard":
    page_header("Business performance", "Revenue, products and customers, with returns and invalid records already removed.")
    df_all, total_customers = load_dashboard_data()

    with st.expander("Filter the data"):
        f1, f2, f3 = st.columns(3)
        lo, hi = df_all.order_date.min().date(), df_all.order_date.max().date()
        dr = f1.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
        all_cats, all_cities = sorted(df_all.category.unique()), sorted(df_all.city.unique())
        cats = f2.multiselect("Category", all_cats, default=all_cats)
        cities = f3.multiselect("City", all_cities, default=all_cities)

    df = df_all[df_all.category.isin(cats) & df_all.city.isin(cities)]
    if len(dr) == 2:
        df = df[(df.order_date.dt.date >= dr[0]) & (df.order_date.dt.date <= dr[1])]
    net = df[df.returned == 0]
    if df.empty or net.empty:
        st.warning("No orders match these filters.")
        st.stop()

    per = net.order_date.dt.to_period("M")
    monthly = net.groupby(per.astype(str)).gross.sum().reset_index()
    monthly.columns = ["month", "rev"]
    has_mom = len(monthly) > 1 and monthly.rev.iloc[-2] > 0
    mom = (monthly.rev.iloc[-1] / monthly.rev.iloc[-2] - 1) * 100 if has_mom else 0
    last = net.order_date.max()
    partial = last.day < last.days_in_month
    lost, ret_rate = df[df.returned == 1].gross.sum(), df.returned.mean() * 100
    by_cust = net.groupby("customer_id").gross.sum()
    top10_share = by_cust.nlargest(10).sum() / net.gross.sum() * 100
    peak = monthly.loc[monthly.rev.idxmax()]
    cat_rev = net.groupby("category").gross.sum().sort_values()
    best_cat, best_cat_share = cat_rev.index[-1], cat_rev.iloc[-1] / cat_rev.sum() * 100
    city_rev = net.groupby("city").gross.sum().sort_values()
    best_city, best_city_share = city_rev.index[-1], city_rev.iloc[-1] / city_rev.sum() * 100
    cat_ret = df.groupby("category").returned.mean().sort_values()

    tab1, tab2, tab3 = st.tabs(["Overview", "Products", "Customers"])

    # ---------- OVERVIEW ----------
    with tab1:
        trend = ""
        if has_mom:
            trend = f", {'up' if mom >= 0 else 'down'} {abs(mom):.1f}% on the previous month" + (" (month still in progress)" if partial else "")
        st.markdown(f'<div class="headline">{fmt(net.gross.sum())} in net revenue{trend}.</div>', unsafe_allow_html=True)

        cards = [
            ("Net revenue", fmt(net.gross.sum()), f"{net.order_id.nunique():,} orders kept", "", spark(monthly.rev), f"Rs. {net.gross.sum():,.0f}"),
            ("Valid orders", f"{df.order_id.nunique():,}", f"{total_customers:,} customers on file", "", spark(df.groupby(df.order_date.dt.to_period('M')).order_id.nunique()), ""),
            ("Average order", fmt(net.gross.sum() / net.order_id.nunique()), "after discounts", "", spark(net.groupby(per).gross.mean()), ""),
            ("Return rate", f"{ret_rate:.1f}%", f"{fmt(lost)} revenue lost", "bad", spark(df.groupby(df.order_date.dt.to_period('M')).returned.mean(), CLAY), ""),
        ]
        st.markdown('<div class="kgrid">' + "".join(
            f'<div class="k"><div class="kl">{l}</div><div class="kv" title="{t}">{v}</div>'
            f'<div class="ks {c}">{s}</div>{sp}</div>' for l, v, s, c, sp, t in cards) + "</div>", unsafe_allow_html=True)

        head("01 · Trend", f"Revenue peaked in {peak.month}", "Net revenue per month; dashed line is the 3-month average.")
        monthly["ma"] = monthly.rev.rolling(3, min_periods=1).mean()
        f = go.Figure()
        f.add_trace(go.Scatter(x=monthly.month, y=monthly.rev, name="Net revenue", mode="lines",
                               line=dict(color=GREEN, width=2.5), fill="tozeroy", fillcolor="rgba(31,92,74,.08)"))
        f.add_trace(go.Scatter(x=monthly.month, y=monthly.ma, name="3-month average", mode="lines",
                               line=dict(color=MUTED, width=1.5, dash="dash")))
        f.add_annotation(x=peak.month, y=peak.rev, text=f"Peak: {fmt(peak.rev)}", showarrow=True, arrowcolor=MUTED,
                         ay=-34, font=dict(color=INK, size=12))
        show(style(f))

        head("02 · Takeaways", "Three things worth knowing")
        st.markdown(f'<ol class="take"><li><b>{best_cat}</b> brings in {best_cat_share:.0f}% of net revenue, and <b>{best_city}</b> is the strongest city.</li>'
                    f'<li><b>{cat_ret.index[-1]}</b> has the highest return rate at {cat_ret.iloc[-1] * 100:.1f}%.</li>'
                    f'<li>The top 10 customers account for {top10_share:.1f}% of revenue.</li></ol>', unsafe_allow_html=True)

    # ---------- PRODUCTS ----------
    with tab2:
        a, b = st.columns([3, 2], gap="large")
        with a:
            head("01 · Categories", f"{best_cat} brings in {best_cat_share:.0f}% of revenue", "Net revenue by category.")
            f = go.Figure(go.Bar(x=cat_rev.values, y=cat_rev.index, orientation="h",
                                 marker_color=[GREEN if i == len(cat_rev) - 1 else SAGE for i in range(len(cat_rev))],
                                 text=[fmt(v) for v in cat_rev.values], textposition="outside", cliponaxis=False))
            f.update_xaxes(visible=False)
            show(style(f, max(260, 46 * len(cat_rev))))
        with b:
            head("02 · Best sellers", "Top five products", "By net revenue.")
            tp = net.groupby(["product_name", "category"]).gross.sum().nlargest(5).reset_index()
            st.markdown("".join(
                f'<div class="lb"><span class="n">{i + 1}</span><b>{r.product_name}</b><span class="v">{fmt(r.gross)}</span>'
                f'<span class="cat">{r.category}</span></div>' for i, r in tp.iterrows()), unsafe_allow_html=True)

        head("03 · Returns", f"{cat_ret.index[-1]} returns the most, at {cat_ret.iloc[-1] * 100:.1f}% of orders",
             "Each dot is a category. Red dots sit above the overall return rate; the higher and further right, the more it costs.")
        r = df.groupby("category").agg(rev=("gross", "sum"), ret=("returned", "mean"), orders=("order_id", "nunique")).reset_index()
        r["ret"] *= 100
        f = go.Figure(go.Scatter(x=r.rev, y=r.ret, mode="markers+text", text=r.category, textposition="top center",
                                 marker=dict(size=np.sqrt(r.orders / r.orders.max()) * 38 + 8, opacity=.85,
                                             color=[CLAY if v > ret_rate else SAGE for v in r.ret]),
                                 hovertemplate="%{text}<br>Revenue: Rs. %{x:,.0f}<br>Returns: %{y:.1f}%<extra></extra>"))
        f.add_hline(y=ret_rate, line=dict(color=MUTED, dash="dot", width=1), annotation_text="overall rate",
                    annotation_position="bottom right", annotation_font_color=MUTED)
        f.update_yaxes(ticksuffix="%", title="Return rate")
        f.update_xaxes(title="Gross revenue (Rs.)")
        show(style(f, 360))

    # ---------- CUSTOMERS ----------
    with tab3:
        a, b = st.columns(2, gap="large")
        with a:
            head("01 · Geography", f"{best_city} leads with {best_city_share:.0f}% of revenue", "Bubble size shows net revenue.")
            g = net.groupby("city").agg(rev=("gross", "sum"), orders=("order_id", "nunique")).reset_index()
            g["lat"] = g.city.map(lambda x: CITIES.get(x, (np.nan, np.nan))[0])
            g["lon"] = g.city.map(lambda x: CITIES.get(x, (np.nan, np.nan))[1])
            g = g.dropna(subset=["lat", "lon"])
            if g.empty:
                st.info("No mappable cities.")
            else:
                m = px.scatter_geo(g, lat="lat", lon="lon", size="rev", hover_name="city",
                                   hover_data={"orders": True, "lat": False, "lon": False, "rev": ":,.0f"},
                                   color_discrete_sequence=[GREEN], size_max=38, opacity=.65)
                m.update_geos(fitbounds="locations", bgcolor="rgba(0,0,0,0)", landcolor="#EAEFF2", countrycolor=LINE,
                              showcountries=True, showframe=False, showcoastlines=False, showland=True)
                show(style(m, 360))
        with b:
            head("02 · Concentration", f"Top 10 customers = {top10_share:.1f}% of revenue", "Bars: spend per customer. Line: cumulative share of all revenue.")
            cu = net.groupby("customer_name").gross.sum().sort_values(ascending=False).head(10)
            f = go.Figure(go.Bar(x=cu.index, y=cu.values, marker_color=GREEN, name="Spend"))
            f.add_trace(go.Scatter(x=cu.index, y=cu.cumsum() / net.gross.sum() * 100, yaxis="y2", mode="lines+markers",
                                   line=dict(color=CLAY, width=2), name="Cumulative share"))
            f.update_layout(yaxis2=dict(overlaying="y", side="right", ticksuffix="%", showgrid=False))
            f.update_xaxes(tickangle=-40)
            show(style(f, 360))

        wd_rev = net.groupby(net.order_date.dt.day_name()).gross.sum()
        head("03 · Timing", f"{wd_rev.idxmax()} is the strongest day to sell", "Net revenue by weekday and month; darker is higher.")
        h = net.assign(wd=net.order_date.dt.day_name(), mo=net.order_date.dt.strftime("%Y-%m")) \
               .pivot_table(index="wd", columns="mo", values="gross", aggfunc="sum")
        h = h.reindex([d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] if d in h.index])
        show(style(px.imshow(h, aspect="auto", color_continuous_scale=["#EEF2F4", GREEN]).update_coloraxes(showscale=False), 300))

    st.caption("Valid orders only (bad dates, quantity ≤ 0 and price ≤ 0 removed). Net revenue excludes returns. "
               "Orphan product or customer IDs are labelled 'Unknown' so totals reconcile.")

# =====================================================================
elif page == "Churn Prediction":
    page_header("Customer churn", "Enter a customer's history as of 31 May 2026 to see how likely they are to leave.")
    model, mism = load_model(CHURN_MODEL_PATH)
    model = repair_model(model, mism)
    if mism:
        st.warning(f"This churn model was saved with scikit-learn {mism[0]}, but the app is running {mism[1]}. "
                   f"Run `pip install scikit-learn=={mism[0]}` and restart the app so predictions are exact.")

    with st.form("churn_form"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            total_orders = st.number_input("Total Orders", min_value=1, max_value=200, value=5, step=1)
            total_spending = st.number_input("Total Spending (Rs.)", min_value=0.0, max_value=5000000.0, value=50000.0, step=1000.0)
            average_order_value = st.number_input("Average Order Value (Rs.)", min_value=0.0, max_value=1000000.0, value=10000.0, step=500.0)
            days_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=2000, value=60, step=1)
        with c2:
            return_rate = st.slider("Return Rate", 0.0, 1.0, 0.05, 0.01)
            average_delivery_days = st.slider("Average Delivery Days", 1.0, 10.0, 4.0, 0.5)
            age = st.number_input("Age", min_value=18, max_value=100, value=31, step=1)
            membership_type = st.selectbox("Membership Type", ["Standard", "Silver", "Gold", "Premium"])
        submitted = st.form_submit_button("Predict churn", type="primary")

    def build(days):
        return pd.DataFrame([{
            "total_orders": total_orders, "total_spending": total_spending,
            "average_order_value": average_order_value, "days_since_last_order": days,
            "return_rate": return_rate, "average_delivery_days": average_delivery_days,
            "age": age, "membership_type": membership_type}])

    if submitted:
        input_df = build(days_since_last_order)
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0, 1])
        tier, color = (("High risk", CLAY) if probability >= 0.66 else
                       ("Medium risk", OCHRE) if probability >= 0.33 else ("Low risk", GREEN))
        action = {"High risk": "Reach out now with a personal win-back offer, such as a discount or free delivery.",
                  "Medium risk": "Send a reminder with a personalised recommendation and a time-limited coupon.",
                  "Low risk": "Keep them engaged with loyalty perks. No urgent action needed."}[tier]

        head("Result", "Likely to leave" if prediction == 1 else "Likely to stay", "")
        a, b = st.columns([1, 1], gap="large")
        with a:
            st.markdown(f'<div class="panel"><div class="kl">Churn probability</div><div class="big">{probability:.0%}</div>'
                        f'<div class="meter"><i style="left:{probability * 100:.0f}%"></i></div>'
                        f'<div class="sub">Low · Medium · High</div></div>', unsafe_allow_html=True)
        with b:
            st.markdown(f'<div class="panel"><div class="kl">Risk tier</div>'
                        f'<div class="big" style="font-size:2rem;color:{color}">{tier}</div>'
                        f'<p style="margin:.6rem 0 0">{action}</p></div>', unsafe_allow_html=True)

        days = list(range(0, 366, 15))
        probs = model.predict_proba(pd.concat([build(d) for d in days], ignore_index=True))[:, 1] * 100
        head("What-if", "How risk changes with time away", "The model's output with every other input held fixed; only days since last order changes.")
        f = go.Figure(go.Scatter(x=days, y=probs, mode="lines", line=dict(color=GREEN, width=2.5)))
        f.add_trace(go.Scatter(x=[days_since_last_order], y=[probability * 100], mode="markers",
                               marker=dict(size=12, color=CLAY, line=dict(width=2, color=CARD)),
                               hovertemplate="This customer: %{y:.1f}%<extra></extra>"))
        f.update_xaxes(title="Days since last order")
        f.update_yaxes(title="Churn probability", ticksuffix="%")
        show(style(f, 300))

# =====================================================================
elif page == "Sentiment Analysis":
    page_header("Review sentiment", "Paste a customer review and the model will tell you how it reads.")
    examples = {"Try a positive review": "Great quality, fast delivery and excellent value for money.",
                "Try a negative review": "Terrible product, arrived broken and support never replied.",
                "Try a mixed review": "Product is okay but the delivery took too long."}
    cols = st.columns(len(examples))
    for col, (label, txt) in zip(cols, examples.items()):
        if col.button(label):
            st.session_state["review_text"] = txt

    text = st.text_area("Enter a product review", key="review_text",
                        placeholder="Example: Great quality, fast delivery and excellent value for money.")

    if st.button("Analyze sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter a review.")
        else:
            model, mism = load_model(SENTIMENT_MODEL_PATH)
            model = repair_model(model, mism)
            if mism and final_estimator_name(model) != "LogisticRegression":
                st.error(f"This model was saved with scikit-learn {mism[0]} but the app is running {mism[1]}, and "
                         f"results would not be reliable. Run `pip install scikit-learn=={mism[0]}` and restart the app.")
                st.stop()
            clean_text = clean_review_text(text)
            try:
                prediction = model.predict([clean_text])[0]
                probabilities = model.predict_proba([clean_text])[0]
            except NotFittedError:
                st.error(f"The saved model is not compatible with scikit-learn {sklearn.__version__}. "
                         f"Run `pip install -U scikit-learn` and restart the app, or retrain and save the model again.")
                st.stop()
            try:
                classes = list(model.classes_)
            except AttributeError:
                classes = list(model[-1].classes_)
            ml_label, ml_conf = prediction, float(probabilities.max())
            known, unseen, pushed = model_view(model, clean_text, ml_label)
            if known == 0:
                ml_conf = 0.0                      # nothing the model recognises: its answer is just its default
            lex = language_score(text)
            lex_label = None if lex is None else ("Positive" if lex >= 0.25 else "Negative" if lex <= -0.25 else None)

            final, source = ml_label, "trained model"
            if lex_label in classes and lex_label != ml_label and ml_conf < 0.90:
                final, source = lex_label, "general-language check (the trained model was unsure and disagreed)"
            elif known == 0 and lex_label is None:
                final, source = "Unclear", "nothing recognisable in the text"
            color = {"Positive": GREEN, "Negative": CLAY}.get(final, OCHRE)

            bars = "".join(
                f'<div class="pb"><div class="row"><span>{c}</span><span>{p:.0%}</span></div>'
                f'<div class="t"><i style="width:{p * 100:.0f}%;background:{ {"Positive": GREEN, "Negative": CLAY}.get(c, OCHRE) }"></i></div></div>'
                for c, p in sorted(zip(classes, probabilities), key=lambda x: -x[1]))
            a, b = st.columns(2, gap="large")
            with a:
                st.markdown(f'<div class="panel"><div class="kl">Predicted sentiment</div>'
                            f'<div class="big" style="color:{color}">{final}</div>'
                            f'<p class="sub" style="margin-top:.6rem">Decided by: {source}</p></div>', unsafe_allow_html=True)
            with b:
                st.markdown(f'<div class="panel"><div class="kl">Trained model: probability by class</div>{bars}</div>', unsafe_allow_html=True)

            if lex_label and lex_label != ml_label and final == ml_label:
                st.info(f"Mixed signals: the trained model says {ml_label} ({ml_conf:.0%} sure) but a general-language "
                        f"check reads this as {lex_label}. Treat this result with caution.")
            with st.expander("How this was decided"):
                st.write(f"**Trained model:** {ml_label} ({probabilities.max():.0%})"
                         + (f", pushed by: {', '.join(pushed)}" if pushed else ""))
                if lex is None:
                    st.write("**General-language check:** off. Run `pip install vaderSentiment` and restart to enable it.")
                else:
                    st.write(f"**General-language check:** {lex_label or 'Neutral / weak signal'} (score {lex:+.2f})")
                if unseen:
                    st.write(f"**Words the model has never seen in training reviews:** {', '.join(unseen)}. "
                             f"It cannot use these, which is the usual reason a clearly negative sentence gets the wrong label.")
            if mism:
                st.caption(f"Compatibility mode: model saved with scikit-learn {mism[0]}, app running {mism[1]}. "
                           f"For exact results run: pip install scikit-learn=={mism[0]}")