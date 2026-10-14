# FM LMS Service — Production (Hosted Edition)

Goal-driven Facility Management LMS backend: zero-loss knowledge ingestion, Ask FM (RAG search
with mandatory citations), and Build My Learning Path (goal → staged competency path).

Architecture: FastAPI runtime (this repo, deployed as a managed workflow from GitHub), Supabase
Postgres with pgvector as the embedding index, Mistral AI API for classification, grounded answers
(mistral-small-latest, temperature 0) and embeddings (mistral-embed).

Environment variables: DATABASE_URL (Supabase connection string — run supabase_schema.sql first),
MISTRAL_API_KEY (from console.mistral.ai).
