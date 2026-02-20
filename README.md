# AI User Review Aggregator  
Ironhack AI Engineering Bootcamp Project

By Benjamin Hunt and Edwin Sentiego 

git mv "Amazon AI Reviews Aggregator. .png" "amazon-ai-reviews-aggregator.png"

**Main Ironhack Project Website:** https://amazon-user-review-aggregator-project.lovable.app/

This project was developed as part of the Ironhack AI Engineering bootcamp.  
The objective was to build an NLP-based review analysis system capable of:

- Classifying customer reviews into positive, neutral, or negative
- Grouping products into meaningful meta-categories
- Generating structured recommendation summaries using an existing LLM
- Comparing AI-derived ratings with human star ratings

All components were implemented and evaluated following the project guidelines.

---

## Project Objectives

1. Implement a supervised sentiment classification model.
2. Apply unsupervised learning to group products into 4–6 meta-categories.
3. Generate structured recommendation-style summaries using an existing LLM.
4. Evaluate models using quantitative metrics.
5. Organize the project in a clean and reproducible structure.

---

## 1. Sentiment Classification (Supervised Learning)

### Model 2 — LinearSVC (Classical ML Baseline)

Text Representation:
- TF-IDF vectorization

Models Tested:
- XGBoost + Word2Vec
- Random Forest (TF-IDF)
- Naive Bayes (TF-IDF)
- Linear Support Vector Machine (LinearSVC)

**Model 2 (Complete Model):**
- LinearSVC

Performance:
- Train Accuracy: 0.9864  
- Validation Accuracy: 0.9368  
- F1 Score: 0.94  

The Linear SVM was selected based on balanced precision and recall performance on validation data.

---

### Model 5 — Final Sentiment Transformer Model (Colab Notebook)

**Model 5 is the final sentiment model** used for the project’s modern transformer pipeline.  
It is implemented in **`Model5.ipynb`** and is intended to be run in **Google Colab (GPU runtime recommended)**.

**What Model 5 does (end-to-end):**
- Clean preprocessing of review text
- **Leakage-safe 80/20 split**: stratified by sentiment label and **group-aware by product (`item_id`)** to prevent product leakage across train/test
- Transformer fine-tuning (supervised learning)
- Evaluation using **macro F1** (plus per-class reporting and confusion matrix)
- Builds exportable, scored predictions for downstream dashboards / website usage

**Dataset → labels (exact logic used):**
- Input text: `reviews.title` + `. ` + `reviews.text` → `text_model`
- Product grouping key: first ASIN from `asins` → `item_id`
- Rating to 3-class label mapping:
  - 1–2 → **negative**
  - 3 → **neutral**
  - 4–5 → **positive**

**Transformer backbone (teacher fine-tune):**
- `microsoft/deberta-v3-base` (3-class sequence classification)
- Tokenization uses **dynamic padding** via `DataCollatorWithPadding`
- `MAX_LEN = 256`

**Training approach (stability + minority sentiment performance):**
- Uses a custom **Focal Loss** trainer:
  - `GAMMA = 2.0`
  - mild class weights computed from train label counts (inverse-sqrt, mean-normalized)
  - guards against NaN/Inf logits or loss
- TrainingArguments (core settings):
  - output: `./teacher_deberta_focal`
  - learning_rate: `1e-6`
  - epochs: `3`
  - batch_size: train `16`, eval `32`
  - warmup_steps: `500`
  - gradient clipping: `max_grad_norm=0.1`
  - early stopping: patience `1`
  - selects best model by `f1_macro`

**Artifacts + outputs saved by Model 5:**
- Saved model + tokenizer + label map + metrics to:
  - `./artifacts_teacher/`
    - `label_map.json`
    - `metrics.json`
- Saves per-review predictions (including probabilities/confidence) to:
  - `/content/scored_test_predictions.csv` (Colab)

**Extra analysis included in Model 5 (ratings comparison):**
- Includes a utility pipeline to compare **AI-derived “star ratings” vs human ratings** using:
  - `cardiffnlp/twitter-roberta-base-sentiment-latest`
- This supports the “AI vs Human Rating” dashboard and dataset-level comparison.

**How to run Model 5 (Colab):**
1. Open `Model5.ipynb` in Google Colab
2. Upload `user_reviews.csv` into the Colab runtime (`/content`)
3. Run the notebook from top to bottom (it pins versions, then asks for a runtime restart)
4. After training, download:
   - `artifacts_teacher/` (model + metadata)
   - `scored_test_predictions.csv` (scored predictions export)

---

## 2. Meta-Category Clustering (Unsupervised Learning)

To reduce category sparsity and improve interpretability:

1. Reviews were aggregated per product.
2. Product-level embeddings were generated using:
   sentence-transformers/all-MiniLM-L6-v2
3. KMeans clustering was applied (k = 4).
4. Clustering quality was evaluated using:
   - Silhouette Score: 0.158  
   - Davies-Bouldin Index: 1.805  
   - Calinski-Harabasz Score: 6.22  
5. Clusters were visualized using PCA and UMAP.
6. Meta-category labels were assigned after manual inspection.

Resulting meta-categories included:
- Kindle E-Readers
- Alexa & Echo Devices
- Computer & Office Accessories
- Power & Charging

---

## 3. LLM-Based Recommendation Generator

Structured recommendation summaries were generated using LLaMA3 via Ollama.

The approach used:
- Structured metric injection (average rating, review count, sentiment distribution)
- Strict formatting constraints
- Grounding rules to avoid unsupported claims
- Controlled generation parameters for consistency

Each generated summary:
- References dataset statistics
- Avoids speculation
- Follows a fixed paragraph and bullet-point format

---

## Evaluation

Classification was evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score

Clustering was evaluated using:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

Cluster coherence was additionally verified through manual inspection.

---

## Deployment Interfaces

Two Streamlit interfaces were developed using the same underlying models.

### Version 1 – Category-Based Review Generator
- Displays clustered meta-categories
- Shows product-level sentiment aggregation
- Generates LLM-based recommendation summaries

### Version 2 – AI vs Human Rating Analysis Dashboard
- Computes AI-derived ratings from sentiment predictions
- Compares AI ratings to human star ratings
- Displays agreement and divergence statistics
- Provides dataset-level summary metrics

Both interfaces operate on the same classification and clustering pipeline.

---

## Project Structure


review-aggregator/
│
├── data/
├── notebooks/
│ ├── sentiment_model.ipynb
│ ├── clustering_model.ipynb
│ ├── llm_generator.ipynb
│ ├── Model5.ipynb
│
├── app/
│ ├── review_generator_app.py
│ ├── ai_vs_human_dashboard.py
│
├── models/
├── requirements.txt
└── README.md


---

## Environment Setup

```bash
python -m venv .venv

Windows:

.venv\Scripts\activate

macOS/Linux:

source .venv/bin/activate

Then install dependencies:

pip install -r requirements.txt

---

If you want, paste the **final metrics line(s)** from `./artifacts_teacher/metrics.json` after your best run, and I’ll add a clean **“Model 5 Performance”** subsection (without changing any wording elsewhere).
