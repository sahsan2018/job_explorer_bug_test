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
    """Return SQL and parameters to search by job title and skill keywords with relevancy scoring."""
    select_parts = ["0"] # Start with 0 for relevancy_score
    where_conditions = []
    params = []

    # Add scoring for title keywords
    for kw in title_keywords:
        select_parts.append(f"(CASE WHEN title LIKE ? THEN 10 ELSE 0 END)")
        where_conditions.append("title LIKE ?")
        params.extend([f"%{kw}%", f"%{kw}%"]) # One for CASE, one for WHERE

    # Add scoring for skill keywords
    for kw in skill_keywords:
        select_parts.append(f"(CASE WHEN description LIKE ? THEN 5 ELSE 0 END)")
        where_conditions.append("description LIKE ?")
        params.extend([f"%{kw}%", f"%{kw}%"]) # One for CASE, one for WHERE

    if not title_keywords and not skill_keywords:
        # No keywords found; return a simple query
        return "SELECT * FROM job_postings LIMIT 100;", []

    # Combine select parts for relevancy score
    select_clause = " + ".join(select_parts)
    sql = f"SELECT *, ({select_clause}) AS relevancy_score FROM job_postings"

    # Combine where conditions with OR for broader matching
    if where_conditions:
        sql += f" WHERE {' OR '.join(where_conditions)}"

    sql += " ORDER BY relevancy_score DESC LIMIT 100;"
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

        search_button = st.button("Search Jobs")

        if search_button and selected_major:
            # Reset pagination when a new search is initiated
            st.session_state.current_page = 0
            st.session_state.last_selected_major = selected_major # Store the major that triggered the search

            title_keywords, skill_keywords = load_major_keywords(selected_major)

            if not title_keywords and not skill_keywords:
                st.warning("No keywords found for this major.")
                st.session_state.search_results = pd.DataFrame() # Clear results if no keywords
            else:
                st.success(
                    f"Searching with {len(title_keywords)} title and {len(skill_keywords)} skill keywords..."
                )
                query, params = build_query(title_keywords, skill_keywords)
                st.session_state.search_results = query_jobs(query, params) # Store results in session state

        # Display results if they exist in session state
        if 'search_results' in st.session_state and not st.session_state.search_results.empty:
            results = st.session_state.search_results
            current_major_display = st.session_state.get('last_selected_major', 'Selected Major') # Use stored major for display

            st.subheader(f"Job Postings for: {current_major_display}")
            st.write("Results are ranked by relevancy, considering both job title and skill keywords.")

            # Pagination setup
            JOBS_PER_PAGE = 10
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 0 # Initialize if not already set (e.g., first load)

            total_jobs = len(results)
            total_pages = (total_jobs + JOBS_PER_PAGE - 1) // JOBS_PER_PAGE

            start_index = st.session_state.current_page * JOBS_PER_PAGE
            end_index = min(start_index + JOBS_PER_PAGE, total_jobs)

            results_page = results.iloc[start_index:end_index]

            # Display navigation buttons
            nav_cols = st.columns([1, 1, 1], vertical_alignment='center', gap='large', border=True)
            with nav_cols[0]:
                if st.session_state.current_page > 0:
                    if st.button("Previous"):
                        st.session_state.current_page -= 1
                        st.rerun()
            with nav_cols[0]:
                if st.session_state.current_page > 0:
                    if st.button("First Page"):
                        st.session_state.current_page = 0
                        st.rerun()
            with nav_cols[1]:
                # Page number selector
                page_options = [i + 1 for i in range(total_pages)]
                selected_page_display = st.selectbox(
                    "Go to Page:",
                    options=page_options,
                    index=st.session_state.current_page,
                    key="page_selector"
                )
                # Update current_page if selection changes
                if selected_page_display - 1 != st.session_state.current_page:
                    st.session_state.current_page = selected_page_display - 1
                    st.rerun()
            with nav_cols[2]:
                if st.session_state.current_page < total_pages - 1:
                    if st.button("Next"):
                        st.session_state.current_page += 1
                        st.rerun()
            with nav_cols[2]:
                if st.session_state.current_page < total_pages - 1:
                    if st.button("Last Page"):
                        st.session_state.current_page = total_pages - 1
                        st.rerun()

            st.write(f"Displaying jobs {start_index + 1}-{end_index} of {total_jobs}")

            if results_page.empty:
                st.info("No job postings found for this page.")
            else:
                for index, row in results_page.iterrows():
                    st.subheader(f"{row['title']} at {row['company_name']}")
                    st.write(f"**Location:** {row['location']} | **Experience Level:** {row['formatted_experience_level']} | **Relevancy Score:** {row['relevancy_score']:.2f}")

                    with st.expander("View Details"):
                        st.write(f"**Description:**")
                        st.markdown(row['description'])

                        if pd.notna(row['skills_desc']) and row['skills_desc']:
                            st.write(f"**Skills:**")
                            st.markdown(row['skills_desc'])

                        st.write(f"**Listed Time:** {row['listed_time']}")
                        st.write(f"**Work Type:** {row['formatted_work_type']}")
                        st.write(f"**Remote Allowed:** {'Yes' if row['remote_allowed'] else 'No'}")

                        salary_info = []
                        if pd.notna(row['min_salary']) and pd.notna(row['max_salary']):
                            salary_info.append(f"{row['currency']} {row['min_salary']:.2f} - {row['max_salary']:.2f} {row['pay_period']}")
                        elif pd.notna(row['normalized_salary']):
                            salary_info.append(f"Normalized Salary: {row['currency']} {row['normalized_salary']:.2f}")
                        if salary_info:
                            st.write(f"**Salary:** {', '.join(salary_info)}")
                        else:
                            st.write("**Salary:** Not specified")

                        if pd.notna(row['job_posting_url']) and row['job_posting_url']:
                            st.markdown(f"**Job Posting URL:** [Link]({row['job_posting_url']})")
                        if pd.notna(row['application_url']) and row['application_url']:
                            st.markdown(f"**Application URL:** [Link]({row['application_url']})")

                        st.write(f"**Views:** {row['views']} | **Applies:** {row['applies']}")
                    st.markdown("---") # Separator for readability

            # Display navigation buttons (Top)
            nav_cols_top = st.columns([1, 1, 1], vertical_alignment='center', gap='large', border=True)
            with nav_cols_top[0]:
                if st.session_state.current_page > 0:
                    if st.button("Previous", key="prev_top"):
                        st.session_state.current_page -= 1
                        st.rerun()
            with nav_cols_top[0]:
                if st.session_state.current_page > 0:
                    if st.button("First Page", key="first_top"):
                        st.session_state.current_page = 0
                        st.rerun()
            with nav_cols_top[1]:
                # Page number selector (Top)
                page_options = [i + 1 for i in range(total_pages)]
                selected_page_display_top = st.selectbox(
                    "Go to Page:",
                    options=page_options,
                    index=st.session_state.current_page,
                    key="page_selector_top"
                )
                # Update current_page if selection changes
                if selected_page_display_top - 1 != st.session_state.current_page:
                    st.session_state.current_page = selected_page_display_top - 1
                    st.rerun()
            with nav_cols_top[2]:
                if st.session_state.current_page < total_pages - 1:
                    if st.button("Next", key="next_top"):
                        st.session_state.current_page += 1
                        st.rerun()
            with nav_cols_top[2]:
                if st.session_state.current_page < total_pages - 1:
                    if st.button("Last Page", key="last_top"):
                        st.session_state.current_page = total_pages - 1
                        st.rerun()

            st.write(f"Displaying jobs {start_index + 1}-{end_index} of {total_jobs}")

            if not results.empty:
                st.download_button(
                    "Download Results as CSV",
                    data=results.to_csv(index=False),
                    file_name="job_results.csv",
                    mime="text/csv"
                )
