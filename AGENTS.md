# Repository Guidelines for Agents

## Setup and Testing
- Install dependencies with `python -m pip install -r requirements.txt`.
- Run `pytest -q` before committing if tests are present.

## Application Overview
- Streamlit app for exploring job postings by college major.
- Databases:
  - `data/majors.db` for School → Department → Major hierarchy.
  - `data/map.db` for major-to-job-title keywords.
  - `data/jobs.db` is downloaded via `gdown` using `st.secrets["DB_URL"]`.
- Job search currently matches **job titles only** using `LIKE` queries.

## Coding Guidelines
- Use `st.cache_data` for read-only database functions.
- Do **not** hardcode the `jobs.db` path; rely on the download step.
- Keep SQL query construction well commented and easy to follow.
- Future features (skill-based matching, FTS5, filters) should only be implemented when explicitly requested.
- Organize reusable code under a `utils/` directory when the codebase grows.

## Pull Request Notes
- Summarize changes clearly in the PR body.
- Cite modified lines using the `F:path` syntax in summaries when referencing code.
