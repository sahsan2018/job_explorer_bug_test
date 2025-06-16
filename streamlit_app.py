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
@st.cache_data
def load_major_hierarchy():
    conn = sqlite3.connect(MAJOR_DB_PATH)
    df = pd.read_sql(
        "SELECT DISTINCT School, Department, [Major Name] AS Major FROM majors;",
        conn,
    )
    conn.close()
    return df

# Get keywords for selected major
@st.cache_data
def load_major_keywords(major):
    """Return job title and skill keywords for the selected major."""
    conn = sqlite3.connect(MAP_DB_PATH)
    df = pd.read_sql(
        "SELECT job_title_keyword, skill_keyword FROM major_job_map WHERE major = ?;",
        conn,
        params=(major,),
    )
    conn.close()
    job_keywords = df[df["job_title_keyword"].notnull()]["job_title_keyword"].tolist()
    skill_keywords = df[df["skill_keyword"].notnull()]["skill_keyword"].tolist()
    return job_keywords, skill_keywords

# Build SQL query combining title and skill filters
def build_query(title_keywords, skill_keywords):
    """Return SQL and parameters to search by job title and skill keywords."""
    conditions = []
    params = []

    if title_keywords:
        # Match any of the provided title keywords
        title_cond = " OR ".join(["title LIKE ?" for _ in title_keywords])
        conditions.append(f"({title_cond})")
        params.extend([f"%{kw}%" for kw in title_keywords])

    if skill_keywords:
        # Search the job description for skill keywords
        skill_cond = " OR ".join(["description LIKE ?" for _ in skill_keywords])
        conditions.append(f"({skill_cond})")
        params.extend([f"%{kw}%" for kw in skill_keywords])

    if not conditions:
        # No keywords found; return a simple query
        return "SELECT * FROM job_postings LIMIT 100;", params

    if title_keywords and skill_keywords:
        # Require at least one title keyword AND one skill keyword for higher relevancy
        where_clause = " AND ".join(conditions)
    else:
        # Only one type of keyword provided; use OR within that set
        where_clause = conditions[0]

    sql = f"SELECT * FROM job_postings WHERE {where_clause} LIMIT 100;"
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
            title_keywords, skill_keywords = load_major_keywords(selected_major)

            if not title_keywords and not skill_keywords:
                st.warning("No keywords found for this major.")
            else:
                st.success(
                    f"Searching with {len(title_keywords)} title and {len(skill_keywords)} skill keywords..."
                )
                query, params = build_query(title_keywords, skill_keywords)
                results = query_jobs(query, params)

                st.subheader(f"Job Postings for: {selected_major}")
                if title_keywords and skill_keywords:
                    st.write("Matched using job title and skill keywords.")
                elif title_keywords:
                    st.write("Matched using job title keywords only.")
                else:
                    st.write("Matched using skill keywords only.")

                st.dataframe(results)

                if not results.empty:
                    st.download_button(
                        "Download Results as CSV",
                        data=results.to_csv(index=False),
                        file_name="job_results.csv",
                        mime="text/csv"
                    )
