create table if not exists public.ranking (
 student_id text primary key,
 score integer not null,
 submitted_at timestamptz not null,
 case_1 text not null,
 case_2 text not null,
 detail jsonb
);
alter table public.ranking enable row level security;
-- No public policies: use the service-role key only in Streamlit Secrets.
