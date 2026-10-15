# FM LMS Service — Production (Hosted Edition, Gemini)

Goal-driven Facility Management LMS backend: zero-loss knowledge ingestion, Ask FM (RAG search
with mandatory citations), and Build My Learning Path (goal → staged competency path).

Architecture: FastAPI runtime (this repo, deployed on Render from GitHub), Supabase Postgres with
pgvector as the embedding index, and the Google Gemini API (free tier): gemini-2.5-flash for
classification and grounded answers (temperature 0), gemini-embedding-001 at 768 dimensions for
embeddings.

Environment variables: DATABASE_URL (Supabase connection string — run supabase_schema.sql first),
GEMINI_API_KEY (from Google AI Studio).
