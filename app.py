import streamlit as st
import random
import joblib
import pdfplumber
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import spacy

# Load NLP model
nlp = spacy.load("en_core_web_sm")


# Set page config and background styling
st.set_page_config(page_title="AI Learning Platform", layout="wide")

page_bg = '''
<style>
    body {
        background-color: #f4f4f4;
    }
    .stSidebar {
        background-color: #2E3B4E !important;
    }
    .stButton>button {
        background-color: #008CBA;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }
</style>
'''
st.markdown(page_bg, unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

# Function to extract meaningful skills using NLP
def extract_skills(resume_text):
    doc = nlp(resume_text)
    extracted_skills = set()
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
            extracted_skills.add(token.text.lower())
    return extracted_skills

# Simulated Resume Data for Better Training
resume_data = {
    "Job Role": [
        "Data Scientist", "Software Engineer", "AI Engineer", "DevOps Engineer", "Product Manager", "UI Designer", "HR Manager"
    ],
    "Skills": [
        ["python", "machine learning", "deep learning", "sql", "pandas", "data visualization", "big data"],
        ["java", "spring boot", "microservices", "sql", "git", "javascript", "react", "rest apis", "data structures"],
        ["python", "tensorflow", "pytorch", "nlp", "computer vision", "deep learning", "mlops"],
        ["docker", "kubernetes", "ci/cd", "terraform", "aws", "linux", "ansible", "cloud computing", "monitoring"],
        ["agile", "scrum", "market research", "roadmap planning", "business strategy", "a/b testing", "ux research"],
        ["figma", "sketch", "adobe xd", "wireframing", "prototyping", "user research", "html/css", "design systems"],
        ["recruitment", "employee engagement", "conflict resolution", "hr policies", "onboarding", "talent management"]
    ]
}

df = pd.DataFrame(resume_data)

# Mock test links
mock_test_links = {
    "Data Scientist": "https://www.testdome.com/tests/python-online-test/25",
    "Software Engineer": "https://www.hackerrank.com/domains/tutorials/10-days-of-javascript",
    "AI Engineer": "https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187",
    "DevOps Engineer": "https://www.hackerrank.com/domains/tutorials/devops",
    "Product Manager": "https://www.interviewcake.com/product-manager-interview-questions",
    "UI Designer": "https://www.interaction-design.org/literature/topics/ui-design",
    "HR Manager": "https://resources.workable.com/hr-interview-questions"
}

# Tutorial links
tutorial_links = {
    "Data Scientist": ["https://www.kaggle.com/learn", "https://www.coursera.org/specializations/data-science"],
    "Software Engineer": ["https://www.geeksforgeeks.org/", "https://www.udemy.com/topic/software-engineering/"],
    "AI Engineer": ["https://www.tensorflow.org/tutorials", "https://pytorch.org/tutorials/"],
    "DevOps Engineer": ["https://www.udemy.com/course/devops/", "https://www.kubernetes.io/docs/tutorials/"],
    "Product Manager": ["https://www.productschool.com/", "https://www.coursera.org/courses?query=product%20management"],
    "UI Designer": ["https://www.adobe.com/", "https://www.figma.com/resources/learn-design/"],
    "HR Manager": ["https://www.shrm.org/", "https://www.coursera.org/specializations/human-resource-management"]
}

# Resume skill analyzer
def analyze_resume(resume_text, job_role):
    extracted_skills = extract_skills(resume_text)
    required_skills = set(df[df["Job Role"] == job_role]["Skills"].iloc[0])
    matching_skills = extracted_skills.intersection(required_skills)
    missing_skills = required_skills - extracted_skills
    return list(matching_skills), list(missing_skills)

# Sidebar UI
st.sidebar.title("AI Learning Platform")
menu_options = ["Resume Checker", "Mock Test", "Interview Preparation"]
menu = st.sidebar.radio("Navigation", menu_options, key="main_menu")

if menu == "Resume Checker":
    st.title("📄 Resume Matcher")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume_upload")
    job_role = st.selectbox("Select Job Role", df["Job Role"].tolist(), key="resume_role")
    if st.button("Check Match", key="resume_check_button") and uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        matched, missing = analyze_resume(resume_text, job_role)
        match_percentage = (len(matched) / (len(matched) + len(missing))) * 100 if matched or missing else 0
        
        # Pie Chart
        if not matched and not missing:
            st.warning("No skills detected in the resume.")
        else:
            fig, ax = plt.subplots()
            ax.pie(
                [max(len(matched), 1), max(len(missing), 1)],  # Ensure nonzero values
                labels=["Matched Skills", "Missing Skills"],
                colors=["green", "red"],
                autopct='%1.1f%%'
            )
            st.pyplot(fig)
        
        st.write(f"### ✅ Match Percentage: {match_percentage:.2f}%")
        st.write(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
        st.write(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")
        
        if missing:
            st.info("### 🔹 Suggested Skills to Add:")
            for skill in missing:
                st.markdown(f"- {skill}")


if menu == "Mock Test":
    st.title("🎯 Mock Interview Test")
    job_role = st.selectbox("Select Job Role", list(mock_test_links.keys()), key="mock_test_role")
    if st.button("Start Test", key="mock_test_button"):
        st.success(f"[Click here to take the mock test]({mock_test_links[job_role]})")

elif menu == "Interview Preparation":
    st.title("📚 Interview Preparation Resources")
    job_role = st.selectbox("Select Job Role", list(tutorial_links.keys()), key="prep_role")
    if st.button("Get Resources", key="prep_button"):
        st.write("### Useful Resources:")
        for link in tutorial_links[job_role]:
            st.markdown(f"- [{link}]({link})")
