# AI User Review Aggregator  
Ironhack AI Engineering Bootcamp Project

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

Text Representation:
- TF-IDF vectorization

Models Tested:
- XGBoost + Word2Vec
- Random Forest (TF-IDF)
- Naive Bayes (TF-IDF)
- Linear Support Vector Machine (LinearSVC)

Final Model:
- LinearSVC

Performance:
- Train Accuracy: 0.9864  
- Validation Accuracy: 0.9368  
- F1 Score: 0.94  

The Linear SVM was selected based on balanced precision and recall performance on validation data.

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

```
review-aggregator/
│
├── data/
├── notebooks/
│   ├── sentiment_model.ipynb
│   ├── clustering_model.ipynb
│   ├── llm_generator.ipynb
│
├── app/
│   ├── review_generator_app.py
│   ├── ai_vs_human_dashboard.py
│
├── models/
├── requirements.txt
└── README.md
```

---

## Environment Setup

```
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app/review_generator_app.py
```
