import streamlit as st
import sqlite3
import pandas as pd
import gdown
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Constants
MAJOR_DB_PATH = "data/majors.db"
MAP_DB_PATH = "data/map.db"
JOBS_DB_PATH = "data/jobs.db"
FAISS_INDEX_PATH = "data/job_embeddings.faiss" # New constant for FAISS index path
JOBS_GDRIVE_URL = st.secrets["JOB_URL"] # Renamed for clarity
FAISS_GDRIVE_URL = st.secrets["FAISS_URL"] # New constant for FAISS index Google Drive URL

# New Constant for job posting limit
MAX_JOB_POSTINGS_FETCH = 100

# Semantic search constants
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
SEMANTIC_SCORE_SCALE = 100.0 # Scale semantic similarity (0-1) to 0-100
RELEVANCY_THRESHOLD = 0.1 # Minimum relevancy score to display a job

# Ensure job DB exists locally
def download_jobs_db():
    if not os.path.exists(JOBS_DB_PATH):
        st.info("Downloading job postings database...")
        gdown.download(JOBS_GDRIVE_URL, JOBS_DB_PATH, quiet=False)

# Ensure FAISS index exists locally
def download_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        st.info("Downloading FAISS index...")
        gdown.download(FAISS_GDRIVE_URL, FAISS_INDEX_PATH, quiet=False)

def ensure_embedding_metadata_table_exists():
    conn = None
    try:
        conn = sqlite3.connect(JOBS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error during schema check: {e}")
    finally:
        if conn:
            conn.close()

# Load hierarchical structure from majors.db
@st.cache_data
def load_major_hierarchy():
    conn = sqlite3.connect(MAJOR_DB_PATH)
    df = pd.read_sql(
        "SELECT DISTINCT School, Department, [Major Name] AS Major, [Degree Level] AS DegreeLevel FROM majors;",
        conn,
    )
    conn.close()
    return df

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# Generate embeddings for job postings and build FAISS index
@st.cache_resource
def build_job_embeddings_and_faiss_index(current_job_count, embedding_model_name_param):
    model = load_embedding_model()
    
    conn = sqlite3.connect(JOBS_DB_PATH)
    cursor = conn.cursor()

    # Get stored metadata
    stored_embedding_model = cursor.execute("SELECT value FROM embedding_metadata WHERE key = 'embedding_model';").fetchone()
    stored_total_jobs_at_embedding_time = cursor.execute("SELECT value FROM embedding_metadata WHERE key = 'total_jobs_at_embedding_time';").fetchone()

    stored_embedding_model = stored_embedding_model[0] if stored_embedding_model else None
    stored_total_jobs_at_embedding_time = int(stored_total_jobs_at_embedding_time[0]) if stored_total_jobs_at_embedding_time else 0

    regenerate_embeddings = False
    if (not os.path.exists(FAISS_INDEX_PATH) or
        stored_embedding_model != embedding_model_name_param or
        stored_total_jobs_at_embedding_time < current_job_count):
        regenerate_embeddings = True

    if regenerate_embeddings:
        df = pd.read_sql("SELECT job_id, title, description, skills_desc FROM job_postings;", conn)

        # Combine relevant text fields for embedding
        df['combined_text'] = df['title'].fillna('') + " " + \
                              df['description'].fillna('') + " " + \
                              df['skills_desc'].fillna('')
        
        # Generate embeddings
        embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Normalize embeddings for cosine similarity (inner product)
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) # Inner Product for cosine similarity
        index.add(embeddings)

        # Save FAISS index to disk
        faiss.write_index(index, FAISS_INDEX_PATH)

        # Update metadata
        cursor.execute("REPLACE INTO embedding_metadata (key, value) VALUES (?, ?);", ('embedding_model', embedding_model_name_param))
        cursor.execute("REPLACE INTO embedding_metadata (key, value) VALUES (?, ?);", ('total_jobs_at_embedding_time', str(current_job_count)))
        conn.commit()
    else:
        index = faiss.read_index(FAISS_INDEX_PATH)
        df = pd.read_sql("SELECT job_id, title, description, skills_desc FROM job_postings;", conn) # Still need df for id_map

    conn.close()

    # Create a mapping from FAISS index ID to original job ID
    id_map = {i: job_id for i, job_id in enumerate(df['job_id'].tolist())}
    
    return index, id_map, df[['job_id', 'title', 'description', 'skills_desc']] # Return df for later use

# Generate embedding for a major
@st.cache_data
def get_major_embedding(major_name):
    model = load_embedding_model()
    major_embedding = model.encode(major_name)
    major_embedding = np.array(major_embedding).astype('float32')
    faiss.normalize_L2(major_embedding.reshape(1, -1)) # Normalize for cosine similarity
    return major_embedding

# Perform semantic search using FAISS
@st.cache_data
def perform_semantic_search(major_embedding, _faiss_index, job_id_map, k_results):
    D, I = _faiss_index.search(major_embedding.reshape(1, -1), k_results)
    
    results = []
    for i, score in zip(I[0], D[0]):
        if i != -1: # Ensure valid index
            job_id = job_id_map[i]
            results.append({'job_id': job_id, 'semantic_score': float(score)})
    return pd.DataFrame(results)

# Fetch jobs by ID and calculate relevancy in Python
def get_jobs_with_semantic_scores(job_ids_with_scores):
    if job_ids_with_scores.empty:
        return pd.DataFrame()

    job_ids = job_ids_with_scores['job_id'].tolist()
    placeholders = ','.join(['?' for _ in job_ids])
    
    conn = sqlite3.connect(JOBS_DB_PATH)
    sql = f"SELECT * FROM job_postings WHERE job_id IN ({placeholders});"
    jobs_df = pd.read_sql(sql, conn, params=job_ids)
    conn.close()

    # Merge with semantic scores
    merged_df = pd.merge(jobs_df, job_ids_with_scores, left_on='job_id', right_on='job_id', how='inner')
    
    # Calculate relevancy score
    merged_df['relevancy_score'] = merged_df['semantic_score'] * SEMANTIC_SCORE_SCALE
    
    # Filter by relevancy threshold and limit
    filtered_df = merged_df[merged_df['relevancy_score'] > RELEVANCY_THRESHOLD]
    sorted_df = filtered_df.sort_values(by='relevancy_score', ascending=False)
    
    return sorted_df.head(MAX_JOB_POSTINGS_FETCH)


# Run query on jobs.db
def query_jobs(sql_query, params):
    conn = sqlite3.connect(JOBS_DB_PATH)
    df = pd.read_sql(sql_query, conn, params=params)
    conn.close()
    return df

# Streamlit UI
st.set_page_config(page_title="Major-to-Job Explorer", layout="wide")
st.title("🎓 Major-to-Job Postings Explorer")

# Download job DB and FAISS index if needed
download_jobs_db()
download_faiss_index()

# Ensure embedding metadata table exists
ensure_embedding_metadata_table_exists()

# Load hierarchy
hierarchy_df = load_major_hierarchy()

# Step 1: Select School
schools = sorted(hierarchy_df["School"].unique())
selected_school = st.selectbox("Select a School:", schools)

if selected_school:
    departments = sorted(hierarchy_df[hierarchy_df["School"] == selected_school]["Department"].unique())
    selected_department = st.selectbox("Select a Department:", departments)

    if selected_department:
        majors_df = hierarchy_df[
            (hierarchy_df["School"] == selected_school) &
            (hierarchy_df["Department"] == selected_department)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Create a display string for the selectbox
        majors_df["Display"] = majors_df["Major"] + " (" + majors_df["DegreeLevel"] + ")"
        
        # Create a mapping from display string to actual major name
        major_display_to_name = dict(zip(majors_df["Display"], majors_df["Major"]))

        # Sort the display names
        display_majors = sorted(majors_df["Display"].unique())
        
        selected_major_display = st.selectbox("Select a Major:", display_majors)

        # Get the actual major name from the display name
        selected_major = major_display_to_name.get(selected_major_display)

        search_button = st.button("Search Jobs")

        if search_button and selected_major:
            # Reset pagination when a new search is initiated
            st.session_state.current_page = 0
            st.session_state.last_selected_major = selected_major # Store the major that triggered the search

            # Get current job count for cache busting
            conn = sqlite3.connect(JOBS_DB_PATH)
            cursor = conn.cursor()
            current_job_count = cursor.execute("SELECT COUNT(*) FROM job_postings;").fetchone()[0]
            conn.close()

            with st.spinner("Preparing semantic data..."):
                faiss_index, job_id_map, all_jobs_df = build_job_embeddings_and_faiss_index(current_job_count, EMBEDDING_MODEL_NAME)
            
            with st.spinner(f"Generating embedding for {selected_major}..."):
                major_embedding = get_major_embedding(selected_major)

            with st.spinner("Performing semantic search..."):
                # Search for more results than MAX_JOB_POSTINGS_FETCH to allow for filtering by relevancy_threshold
                semantic_results = perform_semantic_search(major_embedding, faiss_index, job_id_map, MAX_JOB_POSTINGS_FETCH * 2)
                
                # Fetch the actual job postings using the IDs and apply relevancy filter
                st.session_state.search_results = get_jobs_with_semantic_scores(semantic_results) # Store results in session state

            if not st.session_state.search_results.empty:
                st.success(f"Search complete! Found {len(st.session_state.search_results)} relevant job postings.")
            else:
                st.warning("No relevant job postings found for this major.")
                st.session_state.search_results = pd.DataFrame() # Clear results if no keywords

        # Display results if they exist in session state
        if 'search_results' in st.session_state and not st.session_state.search_results.empty:
            results = st.session_state.search_results
            current_major_display = st.session_state.get('last_selected_major', 'Selected Major') # Use stored major for display

            st.subheader(f"Job Postings for: {current_major_display}")
            st.write("Results are ranked by semantic relevancy.")

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
