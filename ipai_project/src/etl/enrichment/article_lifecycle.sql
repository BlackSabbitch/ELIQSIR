-- 1. Rename the old column for architectural consistency
ALTER TABLE fact_bioactivity 
RENAME COLUMN date_key TO publication_date_key;

-- 2. Add the new role-playing dimension keys
ALTER TABLE fact_bioactivity
ADD COLUMN received_date_key INT DEFAULT 19000101,
ADD COLUMN revised_date_key INT DEFAULT 19000101,
ADD COLUMN accepted_date_key INT DEFAULT 19000101,
ADD COLUMN epub_date_key INT DEFAULT 19000101,
ADD COLUMN ppub_date_key INT DEFAULT 19000101;

CREATE TABLE eliqsir_dwh.stg_article_dates_enrichment (
    pubmed_id VARCHAR(50),
    received_date_key INT,
    revised_date_key INT,
    accepted_date_key INT,
    epub_date_key INT,
    ppub_date_key INT,
    PRIMARY KEY (pubmed_id) 
);