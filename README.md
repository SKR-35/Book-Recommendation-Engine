![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-Apache--2.0-orange)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![Status](https://img.shields.io/badge/Status-Stable-success)

# Book Recommendation Engine

A scalable hybrid recommendation engine that combines **content-based filtering**, **collaborative filtering**, **popularity models** and **explainable AI** to generate personalized book recommendations from Goodreads libraries.

Built for large datasets and commodity hardware (Windows + 8 GB RAM).

---

## Features

- Hybrid recommendation engine
- Content-based recommendations (TF-IDF + Truncated SVD)
- Collaborative filtering (Implicit ALS)
- Popularity baseline
- Explainable recommendations
- Goodreads library import
- Automatic duplicate exclusion
- Cold-start support using Google Books
- Batch processing
- Memory-efficient pipeline
- Parquet-first architecture
- Production-oriented CLI workflow

---

# Architecture

```text
                    Goodreads Dataset
                           │
                           ▼
                  Raw JSON / CSV Files
                           │
                           ▼
                Data Cleaning & Catalog
                           │
                           ▼
                 Popularity Statistics
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
 Content-Based Pipeline             Collaborative Pipeline
(TF-IDF → SVD → Embeddings)        (Implicit ALS Factors)
          │                                 │
          └────────────────┬────────────────┘
                           ▼
                  Hybrid Embedding Space
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
 Goodreads Library Import            Single Book Matching
                           │
                           ▼
                 Personalized User Profile
                           │
                           ▼
               Hybrid Recommendation Engine
                           │
                           ▼
               Explainable Recommendations
```

---

# Pipeline

| Step | Script | Description |
|-------|----------|-------------|
|01|Inspect Raw|Validate downloaded Goodreads dataset|
|02|Convert to Parquet|Chunked conversion|
|03|Build Catalog|Master catalog creation|
|04|Filter Interactions|Prepare collaborative data|
|05|Popularity Baseline|Bayesian & Wilson rankings|
|06|Build Content Features|TF-IDF + SVD|
|07|Content Index|Approximate nearest neighbour index|
|08|Collaborative Model|Implicit ALS|
|09|Hybrid Model|Merge content + collaborative embeddings|
|10|Google Books Client|Cold-start metadata retrieval|
|11|Match Book Input|Local fuzzy book matching|
|12|Import Goodreads Library|Import personal library|
|13|Build User Profile|Create 192-dimensional hybrid profile|
|14|Recommend|Generate personalized recommendations|
|15|Explain Recommendation|Explain recommendation reasoning|

---

# Recommendation Methods

The engine combines several independent signals.

## Content-Based

- TF-IDF
- Truncated SVD
- Cosine similarity

---

## Collaborative Filtering

- Implicit ALS
- User-item latent factors

---

## Hybrid

```
Content Embedding (128)
+
Collaborative Embedding (64)
--------------------------------
Hybrid Embedding (192)
```

---

## Popularity Signals

- Bayesian Rating
- Wilson Lower Bound
- Average Rating
- Ratings Count

---

# Explainability

Each recommendation includes:

- closest books from user's library
- content similarity
- shared genres
- shared authors
- model score
- natural language explanation

Example:

> Recommended because it is closely related to **Crime and Punishment** and **The Death of Ivan Ilych**, while also matching your interest in classic literary fiction.

---

# Cold Start Strategy

If a requested book does not exist locally:

```
Google Books API
        │
        ▼
Metadata Extraction
        │
        ▼
Content Embedding
        │
        ▼
Nearest Neighbours
        │
        ▼
Popularity Re-ranking
```

---

# Project Structure

```
scripts/
processed/
raw/
config/
book_recommender/

README.md
requirements.txt
```

---

# Example Output

```
1  The Catcher in the Rye

Reason:

Related to

• Crime and Punishment
• The Death of Ivan Ilych
• Fathers and Sons

Hybrid similarity:
0.78
```

---

# Performance

Current implementation processes approximately

- 255k books
- 46M interactions
- 670k users

while remaining usable on an ordinary laptop with 8 GB RAM through chunked processing and memory mapping.

---

# Technologies

- Python
- Pandas
- NumPy
- PyArrow
- SciPy
- Implicit ALS
- hnswlib
- SQLite
- Google Books API

---

# Future Enhancements

Planned improvements include:

- Reading plan generator
- Streamlit dashboard
- FastAPI REST API
- Recommendation history
- Interactive explanation viewer
- Feedback-based reranking
- Genre exploration
- Author exploration
- Goodreads synchronization