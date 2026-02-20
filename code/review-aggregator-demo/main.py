import os
import json
import joblib
from pathlib import Path
from typing import Dict
import pandas as pd
import streamlit as st
import re


st.set_page_config(page_title="AI Review Aggregator", page_icon="⭐", layout="wide")

DATA_DIR = Path(__file__).parent
TOP_PRODUCTS_PATH = DATA_DIR / "top_products.csv"
ARTICLES_PATH = DATA_DIR / "all_articles_fixed.json"

st.title("Review Aggregator")
st.caption("Explore top products by category and view summary insights.")
# ---- Load data

@st.cache_resource
def load_sentiment_model():
    vectorizer = joblib.load(DATA_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(DATA_DIR / "svm_model.joblib")
    return vectorizer, model

def suggest_stars(sentiment: str) -> int:
    sentiment = sentiment.lower()
    if sentiment == "negative":
        return 2
    elif sentiment == "neutral":
        return 3
    else:
        return 5

@st.cache_data
def load_top_products():
    return pd.read_csv(TOP_PRODUCTS_PATH)

@st.cache_data
def load_articles() -> Dict[str, str]:
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    
    import re

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def clean_summary_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n")
    s = s.replace(",,,", "").replace(",,", ",")
    return s.strip()

def extract_rank_sections(category_text: str):
    pattern = r"(##\s*Rank\s*(\d+)\s*:\s*(.*?))\n"
    matches = list(re.finditer(pattern, category_text))

    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(category_text)

        rank = int(m.group(2))
        title = (m.group(3) or "").strip()
        body = category_text[start:end].strip()

        sections.append({"rank": rank, "title": title, "body": body})
    return sections

def pick_best_section_for_product(sections, product_name: str):
    pn = normalize(product_name)

    if not pn:
        return None

    for sec in sections:
        title_norm = normalize(sec["title"])

        # Direct match
        if pn in title_norm:
            return sec

        # Reverse partial match
        if title_norm in pn:
            return sec

    for sec in sections:
        body_norm = normalize(sec["body"])
        if pn in body_norm:
            return sec

    return None

if not TOP_PRODUCTS_PATH.exists():
    st.error("Missing file: top_products.csv (put it in the same folder as app.py)")
    st.stop()

if not ARTICLES_PATH.exists():
    st.warning("Optional file missing: all_articles_fixed.json (raw section will be skipped)")

df = load_top_products()

# ---- Layout
left, right = st.columns([1.2, 1])


st.divider()

st.subheader("Product Explorer")
st.caption("Select a category, browse top products, and view detailed summary insights.")

# Load summaries (category-keyed)
if ARTICLES_PATH.exists():
    articles = load_articles()
else:
    articles = {}

# Build category list (prefer intersection so dropdown matches both data sources)
df_categories = (
    sorted(df["meta_label"].dropna().unique().tolist())
    if "meta_label" in df.columns
    else []
)
json_categories = (
    sorted(list(articles.keys()))
    if isinstance(articles, dict)
    else []
)

categories = (
    sorted(set(df_categories).intersection(set(json_categories)))
    if df_categories and json_categories
    else (df_categories or json_categories)
)

if not categories:
    st.info("No categories found. Check top_products.csv (meta_label) and all_articles_fixed.json (keys).")
else:
    chosen_cat = st.selectbox("Choose category:", categories)

    # Filter products to the chosen category
    df_cat = df[df["meta_label"] == chosen_cat].copy() if "meta_label" in df.columns else df.copy()

    # Show top products table FIRST (browse -> then pick)
    show_cols = [c for c in ["asins", "name"] if c in df_cat.columns]
    st.dataframe(
        df_cat[show_cols] if show_cols else df_cat,
        use_container_width=True,
        height=260
    )

    # Product picker AFTER seeing the table
    if "name" in df_cat.columns:
        product_options = df_cat["name"].fillna("Unknown").tolist()
        chosen_product = st.selectbox("Choose product:", product_options)
    else:
        product_options = df_cat["asins"].fillna("Unknown").tolist()
        chosen_product = st.selectbox("Choose product (ASIN):", product_options)

    # Pull + clean the category summary text
    cat_text = clean_summary_text(articles.get(chosen_cat, ""))

    # Extract rank sections and match selected product
    sections = extract_rank_sections(cat_text)
    selected = pick_best_section_for_product(sections, chosen_product)

    st.markdown("---")
    st.markdown("### Product Summary")

    if selected:
        st.markdown(f"**Rank {selected['rank']}: {selected['title']}**")
        # Scrollable details so the page doesn't become a wall of text
        st.text_area("Summary details", selected["body"], height=300)
    else:
        st.info("Couldn’t match that product to a Rank section. Showing full category summary instead.")
        st.text_area("Category summary", cat_text, height=320)

        st.divider()
st.subheader("Try it live: paste a review")

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

review_text = st.text_area("Paste a review:", key="review_text", height=140)

if (DATA_DIR / "svm_model.joblib").exists() and (DATA_DIR / "tfidf_vectorizer.joblib").exists():

    vectorizer, model = load_sentiment_model()

    pred = None
    stars = None

    if st.button("Predict sentiment"):
        if review_text.strip() == "":
            st.warning("Please paste a review first.")
        else:
            X = vectorizer.transform([review_text])
            pred = model.predict(X)[0]
            stars = suggest_stars(pred)

            st.success(f"Predicted sentiment: {pred}")
            st.info(f"Suggested rating: {stars} ⭐")

    if pred is not None:
        user_stars = st.slider("Choose your final rating:", 1, 5, stars)
        st.write(f"Final selected rating: {user_stars} ⭐")

else:
    st.info("Live sentiment model not saved yet (missing svm_model.joblib or tfidf_vectorizer.joblib).")