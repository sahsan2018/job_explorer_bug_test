import streamlit as st
import sqlite3
import pandas as pd
import gdown
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from wordcloud import WordCloud
import altair as alt
import textwrap
import plotly.express as px
from sklearn.cluster import KMeans
import torch

# Constants
MAJOR_DB_PATH = "data/majors.db"
MAP_DB_PATH = "data/map.db"
JOBS_DB_PATH = "data/jobs.db"
FAISS_INDEX_PATH = "data/job_embeddings.faiss"
JOBS_GDRIVE_URL = st.secrets["JOB_URL"]
FAISS_GDRIVE_URL = st.secrets["FAISS_URL"]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
dimensions=384

# New Constant for job posting limit
MAX_JOB_POSTINGS_FETCH = 100

# Semantic search constants
SEMANTIC_SCORE_SCALE = 100.0
RELEVANCY_THRESHOLD = 0.1

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
    return SentenceTransformer('mixedbread-ai/mxbai-embed-xsmall-v1', device=DEVICE, truncate_dim=dimensions)


# Load FAISS index and job ID map
@st.cache_resource
def load_faiss_index():
    download_faiss_index()
    index = faiss.read_index(FAISS_INDEX_PATH)
    return index

# Generate embedding for a major
@st.cache_data
def get_major_embedding(major_display: str):
    """
    major_display is of the form "Major Name (DegreeLevel)".
    We parse out both pieces, lookup description, and encode all three.
    """
    model = load_embedding_model()

    # 1) parse the display into name & degree
    if "(" in major_display and major_display.endswith(")"):
        name, degree = major_display.rsplit("(", 1)
        name = name.strip()
        degree = degree[:-1]  # drop trailing ")"
    else:
        name, degree = major_display, ""

    # 2) fetch the rich description from majors.db
    conn = sqlite3.connect(MAJOR_DB_PATH)
    row = conn.execute(
        "SELECT description FROM majors WHERE [Major Name]=? AND [Degree Level]=?",
        (name, degree)
    ).fetchone()
    conn.close()
    desc = row[0] if row and row[0] else ""

    # 3) build the full prompt
    full_text = f"{name} ({degree}). {desc}"

    # 4) embed
    emb = model.encode(full_text, prompt_name="query", convert_to_numpy=True)
    emb = np.array(emb, dtype='float32')
    faiss.normalize_L2(emb.reshape(1, -1))
    return emb

@st.cache_data
def get_major_query_text(major_display: str) -> str:
    # parse out name & degree exactly like get_major_embedding
    if "(" in major_display and major_display.endswith(")"):
        name, degree = major_display.rsplit("(", 1)
        name = name.strip()
        degree = degree[:-1]
    else:
        name, degree = major_display, ""
    # fetch the same description
    conn = sqlite3.connect(MAJOR_DB_PATH)
    row = conn.execute(
        "SELECT description FROM majors WHERE [Major Name]=? AND [Degree Level]=?",
        (name, degree)
    ).fetchone()
    conn.close()
    desc = row[0] if row and row[0] else ""
    # rebuild the exact query text
    return f"{name} ({degree}). {desc}"

# Perform semantic search using FAISS
@st.cache_data
def perform_semantic_search(major_embedding, _faiss_index, k_results):
    D, I = _faiss_index.search(major_embedding.reshape(1, -1), k_results)
    results = []
    for idx64, score in zip(I[0], D[0]):
        if idx64 == -1:
            continue
        job_id = int(idx64)                # this *is* your job_id
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
    
    # Calculate the 25th percentile of relevancy scores
    # Only calculate if there are enough scores to make sense, otherwise use a default low threshold
    if not merged_df.empty and len(merged_df) >= 4: # Ensure at least 4 elements for 25th percentile
        percentile_threshold = np.percentile(merged_df['relevancy_score'], 25)
    else:
        percentile_threshold = 0.0 # Fallback to a very low threshold if not enough data

    # Filter by the dynamic percentile threshold
    filtered_df = merged_df[merged_df['relevancy_score'] >= percentile_threshold]
    
    # Sort by relevancy score and limit to MAX_JOB_POSTINGS_FETCH
    sorted_df = filtered_df.sort_values(by='relevancy_score', ascending=False)
    
    return sorted_df.head(MAX_JOB_POSTINGS_FETCH)


# Run query on jobs.db
def query_jobs(sql_query, params):
    conn = sqlite3.connect(JOBS_DB_PATH)
    df = pd.read_sql(sql_query, conn, params=params)
    conn.close()
    return df

# Streamlit UI
st.set_page_config(
    page_title="Major-to-Job Explorer", 
    layout="centered",
    initial_sidebar_state="expanded"
)
st.title("🎓 Major-to-Job Postings Explorer")

# Download job DB and FAISS index if needed
download_jobs_db()
download_faiss_index()

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
        ].copy()

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
            st.session_state.last_selected_major = selected_major

            with st.spinner("Loading semantic data..."):
                faiss_index = load_faiss_index()
            
            with st.spinner(f"Generating embedding for {selected_major}..."):
                # use the “Major Name (DegreeLevel)” that the user selected
                major_embedding = get_major_embedding(selected_major_display)


            with st.spinner("Performing semantic search..."):
                # Fetch more results initially to allow for percentile filtering
                semantic_results = perform_semantic_search(major_embedding, faiss_index, MAX_JOB_POSTINGS_FETCH * 4)
                st.session_state.search_results = get_jobs_with_semantic_scores(semantic_results)

            if not st.session_state.search_results.empty:
                st.success(f"Search complete! Found {len(st.session_state.search_results)} relevant job postings.")
            else:
                st.warning("No relevant job postings found for this major.")
                st.session_state.search_results = pd.DataFrame()

        # Display results if they exist in session state
        if 'search_results' in st.session_state and not st.session_state.search_results.empty:
            results = st.session_state.search_results
            current_major_display = st.session_state.get('last_selected_major', 'Selected Major')

            # ── Cross‐Encoder Rerank using major description ──
            cross_encoder  = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")

            # 1) Grab the exact same query text we used for FAISS
            query_text = get_major_query_text(selected_major_display)

            # 2) Build (query_text, job_desc) pairs
            pairs = [(query_text, jd) for jd in results["description"].tolist()]

            # 3) Cross-encode as before
            cross_scores = cross_encoder.predict(pairs)
            results["cross_score"] = cross_scores
            results = results.sort_values("cross_score", ascending=False).reset_index(drop=True)

            # 4) (Optional) Truncate to top‐N for display
            TOP_N = st.sidebar.slider("Results to show", 5, 100, 50)
            results = results.head(TOP_N).copy()
            
            # ── Dynamic Role Clustering ──
            # 1) Re-encode titles into embeddings
            model  = load_embedding_model()
            titles = results["title"].tolist()
            embs   = model.encode(titles, convert_to_numpy=True)  

            # 2) Cluster into up to 8 roles
            n_roles = min(8, len(titles))
            kmeans  = KMeans(n_clusters=n_roles, random_state=0).fit(embs)
            results["cluster_id"] = kmeans.labels_

            # 3) Build human-readable names
            centroids = kmeans.cluster_centers_
            role_names = []
            for cid, center in enumerate(centroids):
                idxs = np.where(results["cluster_id"] == cid)[0]      # positional indices
                cluster_embs = embs[idxs]
                # get the positional index of the closest embedding
                winner_pos = idxs[np.argmin(np.linalg.norm(cluster_embs - center, axis=1))]
                # use iloc to fetch by positional index
                role_names.append(results.iloc[winner_pos]["title"])

            # 4) Map into new column
            cluster_to_role = {i: name for i, name in enumerate(role_names)}
            results["role_name"] = results["cluster_id"].map(cluster_to_role)
            
            # ----------Beginning of "Visualization" section-----------------
            viz = st.sidebar.selectbox(
                "Choose a visualization",
                ["None", "Word Cloud", "Top-10 Bar Chart", "Treemap"],
                index = 2
            )
            if viz == "None":
                st.info("No visualization selected. Use the sidebar to choose one.")
            else:
                st.header("🔍 At-a-Glance: Top Job Roles")
                
                if viz == "Word Cloud":
                    # Sum relevancy by role
                    role_weights = (
                        results
                        .groupby("role_name")["relevancy_score"]
                        .sum()
                        .to_dict()
                    )

                    # Generate cloud
                    wc = WordCloud(
                        width=800, height=400,
                        background_color="white",
                        max_words=50
                    ).generate_from_frequencies({r: int(s*100) for r, s in role_weights.items()})

                    st.subheader("Role-Level Word Cloud")
                    st.image(wc.to_array(), use_container_width=True)

                elif viz == "Top-10 Bar Chart":
                    # Let user pick metric
                    metric = st.sidebar.radio("Rank by:", ["Count", "Avg Relevancy"])
                    field  = "count" if metric=="Count" else "avg_rel"

                    # Aggregate on role_name
                    df_role = (
                        results
                        .groupby("role_name")
                        .agg(count=("role_name","size"), avg_rel=("relevancy_score","mean"))
                        .reset_index()
                        .sort_values(field, ascending=False)
                        .head(10)
                    )

                    chart = (
                        alt.Chart(df_role)
                        .mark_bar()
                        .encode(
                            x=alt.X(f"{field}:Q", title=metric),
                            y=alt.Y("role_name:N", sort='-x', title="Role"),
                            tooltip=["role_name","count","avg_rel"]
                        )
                        .properties(title=f"Top-10 Roles by {metric}", height=400)
                    )

                    st.altair_chart(chart, use_container_width=True)

                    with st.expander("View Data Table"):
                        st.table(
                            df_role.rename(columns={
                            "role_name":"Role","count":"Count","avg_rel":"Avg. Relevancy"
                            })
                        )

                elif viz == "Treemap":
                    # 1) Prepare a two-level DataFrame
                    df_tree = (
                        results
                        .groupby(["role_name", "title"])
                        .agg(
                            count=("title", "size"),
                            avg_rel=("relevancy_score", "mean")
                        )
                        .reset_index()
                    )
                    
                    # 2) Prune children: keep top 5 titles per role, aggregate the rest
                    def prune_children(df, top_n=5):
                        pieces = []
                        for role, grp in df.groupby("role_name"):
                            # pick the top N by count
                            top = grp.nlargest(top_n, "count")
                            rest = grp.drop(top.index)
                            pieces.append(top)
                            if not rest.empty:
                                pieces.append(pd.DataFrame({
                                    "role_name": [role],
                                    "title":       ["Other Titles"],
                                    "count":       [rest["count"].sum()],
                                    "avg_rel":     [rest["avg_rel"].mean()]
                                }))
                        return pd.concat(pieces, ignore_index=True)
                    
                    # apply pruning
                    df_tree = prune_children(df_tree, top_n=5)

                    # 3) Build a treemap showing both levels at once
                    fig = px.treemap(
                        df_tree,
                        path=["role_name", "title"],     # level-0=role_name, level-1=title
                        values="count",
                        color="avg_rel",
                        color_continuous_scale="Viridis",
                        hover_data=["count", "avg_rel"],
                        title="Jobs Treemap (Roles → Titles)",
                        maxdepth=2                       # always draw both levels
                    )

                    # 4) Improve padding & fonts for clarity
                    fig.update_traces(
                        tiling=dict(pad=3),             # inner padding
                        outsidetextfont=dict(size=18, color="white"),  # role labels
                        insidetextfont=dict(size=12, color="white"),   # title labels
                        textinfo="label+value"          # show name + count on each rectangle
                    )

                    # 5) Add breathing room and a clear colorbar title
                    fig.update_layout(
                        margin=dict(t=50, l=25, r=25, b=25),
                        coloraxis_colorbar=dict(title="Avg. Relevancy")
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    
                # -----------------End of "Visualization" section-----------------



            st.subheader(f"Job Postings for: {current_major_display}")
            st.write("Results are ranked by semantic relevancy.")

            # Pagination setup
            JOBS_PER_PAGE = 10
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 0

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
                    st.markdown("---")
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
                page_options = [i + 1 for i in range(total_pages)]
                selected_page_display_top = st.selectbox(
                    "Go to Page:",
                    options=page_options,
                    index=st.session_state.current_page,
                    key="page_selector_top"
                )
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
