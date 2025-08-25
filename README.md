This project implements a multi-stage address matching pipeline to reconcile apartment address records between two large datasets:
- CCS (Customer Cloud Service)
- GIS (Geographical Information System)

The system combines fuzzy string matching, semantic similarity with Sentence-BERT, and geospatial filtering to identify high-confidence matches across noisy and inconsistent data. This approach significantly improves data reliability and integration accuracy across systems.

---

🔹 Features:
- Data cleaning and normalization with Pandas + regex
- Address parsing with usaddress
- String similarity (Levenshtein, Jaccard, token-based)
- Semantic similarity using Sentence-Bert embeddings
- Blocking & indexing with MinHash + LSH for scalable comparisons  
- Geospatial validation using BallTree (Haversine distance)

---

🔹 Results:
- Successfully matched 74K+ address records across CCS and GIS datasets.  
- Improved scalability with parallelization in Dask, reducing runtime for large-scale comparisons.  
- Enhanced reliability by reconciling complex apartment information.

---

🔹 Tech Stack:
- Python 3.x
- Pandas, NumPy, Regex  
- RapidFuzz, usaddress  
- Sentence-BERT (Transformers)  
- Datasketch (MinHash, LSH)  
- Dask (parallel execution)  
- Scikit-learn, BallTree, PyProj  

---
