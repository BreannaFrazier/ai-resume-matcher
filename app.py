from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

app = Flask(__name__)

DEFAULT_RESUME = """Breanna Frazier
AI Engineer
breannahfrazier@gmail.com | (864) 318-6686 | Anderson, SC

Summary
AI engineering (Associates) student at Maestro College. Supported SEO
strategies and kept a customer database behind targeted email campaigns,
part of 2+ years as a Marketing Assistant at SPECTRUM CHARTER
COMMUNICATION. Earlier interpreted results from lab experiments.

Experience
Travel Nursing Manager | Home care. Travel | Jan 2025 - Present
- Assess patient conditions and implement targeted nursing interventions,
improving health outcomes and reducing hospital readmission rates
- Deliver nursing care to patients in hospitals, clinics and long-term care
facilities, ensuring consistent patient satisfaction and safety
- Collaborated with interdisciplinary teams to develop care plans,
tailoring solutions to each patient and improving continuity of care
- Manage a team of 10-20

Home Nurse | Anderson, SC | Jan 2025 - Feb 2026
- Administered medications, treatments and therapies as prescribed by
physicians, maintaining 100% compliance with protocols and accurate
documentation
- Monitored vital signs and assessed patient conditions, reporting changes
to the healthcare team to ensure timely interventions
- Delivered compassionate, personalized care to patients in their homes,
enhancing comfort and well-being through attentive support
- Supported patients with bathing, dressing, grooming and mobility as part
of a team of 5-9, promoting independence and dignity

Marketing Assistant | SPECTRUM CHARTER COMMUNICATION | Jan 2023 - Jan 2025
| Malden, SC
- Supported the development and execution of marketing campaigns across
social media, email and content creation, increasing audience engagement
- Contributed to implementing SEO strategies, improving organic search
rankings on major search engines
- Monitored competitor activity through market analysis, identifying
opportunities for differentiation and strengthening marketing strategies
- Maintained a customer database for targeted marketing, enabling
personalized email campaigns and loyalty programs

Biologistics | LABTECH DIAGNOSTICS | Jan 2019 - Jan 2023 | Anderson, SC
- Assigned medical codes to patient records using ICD-10-CM, CPT and HCPCS
systems, ensuring accuracy and compliance with industry standards
- Collected, analyzed and interpreted data from lab experiments using
advanced scientific equipment, supporting reliable diagnostic outcomes
- Calibrated and maintained laboratory instruments, ensuring accuracy of
test results and minimizing equipment downtime
- Contributed to research report preparation by providing data analysis
support, enhancing the quality and reliability of scientific publications

Education
Associate of Applied Science in AI Software Engineering | Maestro College |
In progress
Medical (Associates) | Greenville Tech | Feb 2020 | Greenville, SC

Skills
AI and data: Machine learning (ML) fundamentals, AI integration, AI,
Generative AI, Data analytics, Data reporting, Data filtering, Database
systems
Tools & Domain: Data entry, Operations, Clerical experience, ICD-10-CM
coding, HCPCS coding, CPT coding, Programming languages, APIs, Project team
coordination, Team management, IT experience within healthcare, Process
design, Home health, Nursing, Quality improvement, Operations management,
Software coding, Technical writing, CRM systems, Software development, SEO,
Microsoft Excel, Operations workflow management, Software engineering,
Agile software development
"""

def preprocess_text(text: str) -> str:
    """Lowercases and keeps letters, numbers, spaces, +, and #."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\+\#]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def analyze_match(resume_text: str, target_text: str) -> dict:
    clean_resume = preprocess_text(resume_text)
    clean_target = preprocess_text(target_text)

    if not clean_resume or not clean_target:
        return {"error": "Please provide both resume content and target keywords or a job description."}

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_target])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_percentage = round(float(similarity) * 100, 2)

    feature_names = np.array(vectorizer.get_feature_names_out())
    resume_scores = tfidf_matrix[0].toarray().flatten()
    target_scores = tfidf_matrix[1].toarray().flatten()

    target_top_indices = np.argsort(target_scores)[::-1]
    missing_keywords = []
    matched_keywords = []

    for idx in target_top_indices:
        if target_scores[idx] > 0:
            term = feature_names[idx]
            if resume_scores[idx] == 0 and len(missing_keywords) < 12:
                missing_keywords.append(term)
            elif resume_scores[idx] > 0 and len(matched_keywords) < 12:
                matched_keywords.append(term)

    return {
        "score": match_percentage,
        "matched": matched_keywords,
        "missing": missing_keywords
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    resume_input = DEFAULT_RESUME
    target_input = ""

    if request.method == "POST":
        resume_input = request.form.get("resume", DEFAULT_RESUME)
        target_input = request.form.get("target_input", "")
        result = analyze_match(resume_input, target_input)

    return render_template(
        "index.html",
        result=result,
        resume_input=resume_input,
        target_input=target_input
    )

if __name__ == "__main__":
    app.run(debug=True)
