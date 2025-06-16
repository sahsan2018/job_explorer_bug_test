import streamlit as st
import sqlite3
import pandas as pd
import gdown
import os

# Constants
MAJOR_DB_PATH = "data/majors.db"
MAP_DB_PATH = "data/map.db"
JOBS_DB_PATH = "data/jobs.db"
GDRIVE_URL = st.secrets["DB_URL"]

# Ensure job DB exists locally
def download_jobs_db():
    if not os.path.exists(JOBS_DB_PATH):
        st.info("Downloading job postings database...")
        gdown.download(GDRIVE_URL, JOBS_DB_PATH, quiet=False)

# Load hierarchical structure from majors.db
def load_major_hierarchy():
    conn = sqlite3.connect(MAJOR_DB_PATH)
    df = pd.read_sql("SELECT DISTINCT School, Department, [Major Name] AS Major FROM majors;", conn)
    conn.close()
    return df

# Get keywords for selected major
def load_major_keywords(major):
    conn = sqlite3.connect(MAP_DB_PATH)
    df = pd.read_sql("SELECT * FROM major_job_map WHERE major = ?;", conn, params=(major,))
    conn.close()
    job_keywords = df[df["job_title_keyword"].notnull()]["job_title_keyword"].tolist()
    return job_keywords

# Build SQL query to match job title only
def build_query(keywords):
    conditions = " OR ".join(["title LIKE ?" for _ in keywords])
    sql = f"SELECT * FROM job_postings WHERE {conditions} LIMIT 100;"
    params = [f"%{kw}%" for kw in keywords]
    return sql, params

# Run query on jobs.db
def query_jobs(sql_query, params):
    conn = sqlite3.connect(JOBS_DB_PATH)
    df = pd.read_sql(sql_query, conn, params=params)
    conn.close()
    return df

# Streamlit UI
st.set_page_config(page_title="Major-to-Job Explorer", layout="wide")
st.title("🎓 Major-to-Job Postings Explorer")

# Download job DB if needed
download_jobs_db()

# Load hierarchy
hierarchy_df = load_major_hierarchy()

# Step 1: Select School
schools = sorted(hierarchy_df["School"].unique())
selected_school = st.selectbox("Select a School:", schools)

if selected_school:
    departments = sorted(hierarchy_df[hierarchy_df["School"] == selected_school]["Department"].unique())
    selected_department = st.selectbox("Select a Department:", departments)

    if selected_department:
        majors = sorted(hierarchy_df[
            (hierarchy_df["School"] == selected_school) &
            (hierarchy_df["Department"] == selected_department)
        ]["Major"].unique())
        selected_major = st.selectbox("Select a Major:", majors)

        if selected_major:
            job_keywords = load_major_keywords(selected_major)

            if not job_keywords:
                st.warning("No job title keywords found for this major.")
            else:
                st.success(f"Searching job titles with {len(job_keywords)} keywords...")
                query, params = build_query(job_keywords)
                results = query_jobs(query, params)

                st.subheader(f"Job Postings for: {selected_major}")
                st.write(f"Matched using job title keywords only.")
                st.dataframe(results)

                if not results.empty:
                    st.download_button(
                        "Download Results as CSV",
                        data=results.to_csv(index=False),
                        file_name="job_results.csv",
                        mime="text/csv"
                    )
