-- FM LMS — Supabase schema (run once in Supabase SQL Editor)
-- v2: 768-dim embeddings (Google gemini-embedding-001)
-- Already ran v1 (1024)? Run the migration at the bottom instead.
create extension if not exists vector;

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  name text, size_mb real, sha256 text,
  category text, subcategory text,
  status text default 'processing',   -- processing|quarantined|error|published
  coverage_pct real default 100, toc_count int default 0,
  review_note text,
  created_at timestamptz default now()
);

create table if not exists elements (
  id bigserial primary key,
  doc_id uuid references documents(id) on delete cascade,
  kind text, raw text, position int,
  quarantined boolean default false, review_note text
);

create table if not exists chunks (
  id bigserial primary key,
  doc_id uuid references documents(id) on delete cascade,
  toc_path text, section text, page int, element_kinds text[],
  risk_score int default 2, body text,
  embedding vector(768)
);

create index if not exists chunks_vec on chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);

create table if not exists unanswered (
  id bigserial primary key, question text, asked_at timestamptz default now()
);

-- MIGRATION (only if you already ran the v1 schema with vector(1024)):
-- drop index if exists chunks_vec;
-- alter table chunks alter column embedding type vector(768);
-- create index chunks_vec on chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);
