# Breanna Frazier — AI Resume & Role Matcher

An NLP-powered web application that cross-analyzes candidate resumes against target job descriptions or keyword lists using **TF-IDF vectorization** and **cosine similarity**.

🔗 **Live Interactive Demo:** [https://breannafrazier.github.io/ai-resume-matcher/](https://breannafrazier.github.io/ai-resume-matcher/)

---

## Overview & Architecture
This repository includes both a **live client-side web demo** (hosted on GitHub Pages) and a **Python/Flask + Scikit-learn backend** (`app.py`):

* **Text Preprocessing:** Normalizes casing and filters non-alphanumeric characters while preserving technical tokens (such as `C++` and `C#`).
* **N-Gram TF-IDF Vectorization:** Uses English stop-word removal and unigram/bigram tokenization (`ngram_range=(1, 2)`) to weight both single keywords and two-word domain phrases (e.g., *"machine learning"*, *"data analytics"*)[cite: 9].
* **Cosine Similarity Scoring:** Computes vector alignment between the resume and target job description to output an overall percentage match score
* **Keyword Gap Analysis:** Extracts the top-weighted target terms and separates them into **Matched Keywords & Phrases** and **Missing High-Weight Terms**

---

## Tech Stack
* **Backend & NLP:** Python, Flask, Scikit-learn (`TfidfVectorizer`, `cosine_similarity`), NumPy
* **Frontend & Live Demo:** HTML5, CSS3, Vanilla JavaScript (TF-IDF & Cosine Similarity engine)

---

## Running the Flask Backend Locally
```bash
# Clone the repository
git clone https://github.com/BreannaFrazier/ai-resume-matcher.git
cd ai-resume-matcher

# Install dependencies
pip install flask scikit-learn numpy

# Start the local Flask server
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

---

## Author
**Breanna Frazier** — AI Software Engineering Student at Maestro College  
*Focus Areas: Machine Learning, NLP, Healthcare Data Analytics, & Full-Stack AI Integration*
