-- =============================================================================
-- ELIQSIR Data Warehouse — MySQL DDL
-- =============================================================================
-- Star schema for drug-target interaction data.
-- Grain of FactBioactivity: one experimental bioactivity measurement per
--   (drug, protein, publication) triple.
--
-- Run this file against an empty database to create the full schema:
--   mysql -u <user> -p eliqsir_dw < src/database/schema.sql
--
-- Safe to re-run: all statements use IF NOT EXISTS / COMMENT conventions.
-- Foreign key checks are disabled during DDL so tables can be created in any order.
-- =============================================================================

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- Drop existing tables to cleanly recreate the schema (Commented out for safety)
-- DROP TABLE IF EXISTS fact_bioactivity;
-- DROP TABLE IF EXISTS dim_structure;
-- DROP TABLE IF EXISTS dim_article;
-- DROP TABLE IF EXISTS dim_drug;
-- DROP TABLE IF EXISTS dim_protein;
-- DROP TABLE IF EXISTS dim_date;

-- -----------------------------------------------------------------------------
-- DimDate (Temporal Dimension)
-- Grain: one row per day.
-- Required for ML Temporal Split (Out-of-Distribution validation)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_date (
    date_key        INT           NOT NULL COMMENT 'Surrogate key format YYYYMMDD',
    full_date       DATE          NOT NULL COMMENT 'Standard SQL date format',
    full_date_desc  VARCHAR(100)  COMMENT 'Human readable format (e.g., January 14, 1996, Tuesday)',
    year            SMALLINT      COMMENT 'Calendar year',
    month_name      VARCHAR(20)   COMMENT 'Calendar month name (e.g., January)',
    day             SMALLINT      COMMENT 'Day of the month (1-31)',
    quarter         SMALLINT      COMMENT 'Calendar quarter (1-4)',
    day_name        VARCHAR(20)   COMMENT 'Day of week (e.g., Tuesday)',
    is_weekend      VARCHAR(20)   COMMENT 'Text indicator: weekend or non-weekend',
    fractional_year DOUBLE        COMMENT 'Continuous time feature for GNNs (e.g., 2023.5)',
    epoch_time      BIGINT        COMMENT 'Unix epoch time in seconds',

    PRIMARY KEY (date_key),
    INDEX ix_dim_date_year (year)
)
ENGINE = InnoDB
DEFAULT CHARSET = utf8mb4
COLLATE = utf8mb4_unicode_ci
COMMENT = 'Temporal dimension for rolling window ML validation';

-- -----------------------------------------------------------------------------
-- DimProtein
-- Grain: one row per unique UniProt accession (Swiss-Prot reviewed).
-- Source: UniProt REST API  /uniprotkb/stream
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_protein (
    protein_key     INT             NOT NULL AUTO_INCREMENT COMMENT 'ETL-assigned surrogate key',
    uniprot_id      VARCHAR(20)     NOT NULL  COMMENT 'UniProt primary accession (e.g. P04637)',
    gene_names      TEXT                      COMMENT 'Space-separated gene name list from UniProt',
    protein_name    TEXT                      COMMENT 'Recommended protein name from UniProt',
    sequence        TEXT            NOT NULL  COMMENT 'Full amino acid sequence',
    sequence_length INT             NOT NULL  COMMENT 'Number of amino acids (critical for GPU filtering)',
    protein_families VARCHAR(500)             COMMENT 'Protein family classifications',

    PRIMARY KEY (protein_key),
    UNIQUE  KEY uq_dim_protein_uniprot (uniprot_id),
    INDEX   ix_dim_protein_uniprot     (uniprot_id)
)
ENGINE = InnoDB
DEFAULT CHARSET = utf8mb4
COLLATE = utf8mb4_unicode_ci
COMMENT = 'Protein dimension - sourced from UniProt Swiss-Prot';

-- -----------------------------------------------------------------------------
-- DimDrug
-- Grain: one row per unique ChEMBL molecule ID.
-- Source: ChEMBL 36 SQLite  (molecule_dictionary)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_drug (
    drug_key           INT             NOT NULL AUTO_INCREMENT COMMENT 'ETL-assigned surrogate key',
    drug_chembl_id     VARCHAR(20)     NOT NULL  COMMENT 'ChEMBL molecule ID (e.g. CHEMBL25)',
    drug_name          VARCHAR(500)              COMMENT 'Preferred molecule name from ChEMBL',
    molecule_type      VARCHAR(100)              COMMENT 'Type of molecule (e.g., Small molecule) for ML filtering',
    molecular_weight   DECIMAL(10, 2)            COMMENT 'Molecular weight to filter out large biologics for GNNs',
    canonical_smiles   TEXT            NOT NULL  COMMENT 'Chemical structure for ML graph generation',
    standard_inchi_key VARCHAR(27)               COMMENT 'Unique chemical hash to prevent duplicates',

    PRIMARY KEY (drug_key),
    UNIQUE  KEY uq_dim_drug_chembl (drug_chembl_id),
    INDEX   ix_dim_drug_chembl     (drug_chembl_id),
    INDEX   ix_dim_drug_inchi      (standard_inchi_key)
)
ENGINE = InnoDB
DEFAULT CHARSET = utf8mb4
COLLATE = utf8mb4_unicode_ci
COMMENT = 'Drug / molecule dimension - sourced from ChEMBL';


-- -----------------------------------------------------------------------------
-- DimArticle
-- Grain: one row per unique PubMed article.
-- Sources: ChEMBL 36 (title, journal, year)  +  NCBI E-utils (abstract)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_article (
    article_key     INT             NOT NULL AUTO_INCREMENT COMMENT 'ETL-assigned surrogate key',
    pubmed_id       BIGINT                    COMMENT 'NCBI PubMed identifier',
    article_title   TEXT                      COMMENT 'Full article title',
    journal         VARCHAR(300)              COMMENT 'Journal name',
    year            SMALLINT                  COMMENT 'Publication year',
    abstract        TEXT                      COMMENT 'Full abstract text - primary IR corpus',
    doi             VARCHAR(300)              COMMENT 'Digital Object Identifier (when available)',
    authors         TEXT                      COMMENT 'Full list of authors separated by semicolons',

    PRIMARY KEY (article_key),
    UNIQUE  KEY uq_dim_article_pubmed (pubmed_id),
    INDEX   ix_dim_article_pubmed     (pubmed_id),
    -- Full-text index on abstract to accelerate keyword search
    FULLTEXT KEY ft_dim_article_abstract (abstract, article_title)
)
ENGINE = InnoDB
DEFAULT CHARSET = utf8mb4
COLLATE = utf8mb4_unicode_ci
COMMENT = 'Publication dimension - metadata from ChEMBL, abstract from PubMed';


-- -----------------------------------------------------------------------------
-- DimStructure
-- Grain: one row per (PDB entry ID, chain ID) pair.
-- Source: PDBe Graph API  /mappings/best_structures/{uniprot_id}
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_structure (
    structure_key   INT             NOT NULL AUTO_INCREMENT COMMENT 'ETL-assigned surrogate key',
    protein_key     INT             NOT NULL  COMMENT 'FK to dim_protein (Surrogate key replaces business key for DW standard)',
    pdb_id          VARCHAR(10)     NOT NULL  COMMENT '4-character PDB entry ID (e.g. 1TUP)',
    chain_id        VARCHAR(10)               COMMENT 'Chain identifier within the PDB entry',
    resolution      FLOAT                     COMMENT 'Crystallographic resolution in Angstroms',
    coverage        FLOAT                     COMMENT 'Fraction of UniProt sequence covered (0-1)',
    method          VARCHAR(100)              COMMENT 'Experimental method (X-RAY, EM, NMR, ...)',
    unp_start       INT                       COMMENT 'First residue of the mapped UniProt sequence',
    unp_end         INT                       COMMENT 'Last residue of the mapped UniProt sequence',

    PRIMARY KEY (structure_key),
    UNIQUE  KEY uq_dim_structure_pdb_chain (pdb_id, chain_id),
    INDEX   ix_dim_structure_pdb           (pdb_id),
    INDEX   ix_dim_structure_protein       (protein_key),

    CONSTRAINT fk_structure_protein
        FOREIGN KEY (protein_key)
        REFERENCES  dim_protein (protein_key)
        ON DELETE CASCADE
        ON UPDATE CASCADE
)
ENGINE = InnoDB
DEFAULT CHARSET = utf8mb4
COLLATE = utf8mb4_unicode_ci
COMMENT = '3-D structure dimension - sourced from PDBe Graph API';


-- -----------------------------------------------------------------------------
-- FactBioactivity
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_bioactivity (
    activity_id         BIGINT          NOT NULL  COMMENT 'ChEMBL activity_id - natural PK',

    -- Foreign keys
    protein_key         INT             NOT NULL  COMMENT 'FK -> dim_protein',
    drug_key            INT             NOT NULL  COMMENT 'FK -> dim_drug',
    article_key         INT                       COMMENT 'FK -> dim_article (nullable)',
    date_key            INT             NOT NULL DEFAULT 19000101 COMMENT 'FK -> dim_date',

    -- Measures
    date_precision      ENUM('Exact', 'Month', 'Year', 'Unknown') NOT NULL DEFAULT 'Unknown' COMMENT 'Precision of the temporal anchor',
    standard_type       VARCHAR(100)              COMMENT 'Activity type: IC50, Ki, Kd, EC50, ...',
    standard_value      DOUBLE                    COMMENT 'Measured numerical value',
    standard_units      VARCHAR(50)               COMMENT 'Units of standard_value (e.g. nM)',
    pchembl_value       FLOAT                     COMMENT '-log10(molar potency) - comparable across assays',
    confidence_score    TINYINT UNSIGNED          COMMENT 'ChEMBL target confidence 0-9',

    -- Degenerate dimensions (assay context)
    assay_type          VARCHAR(10)               COMMENT 'B=biochemical, F=functional, A=ADMET, ...',
    assay_description   TEXT                      COMMENT 'Free-text description of the assay setup',
    assay_organism      VARCHAR(255)              COMMENT 'Organism of the assay (critical for ML target species filtering)',

    PRIMARY KEY (activity_id),
    INDEX ix_fact_protein  (protein_key),
    INDEX ix_fact_drug     (drug_key),
    INDEX ix_fact_article  (article_key),
    INDEX ix_fact_date     (date_key),

    CONSTRAINT fk_fact_protein
        FOREIGN KEY (protein_key)
        REFERENCES  dim_protein (protein_key)
        ON DELETE RESTRICT
        ON UPDATE CASCADE,

    CONSTRAINT fk_fact_drug
        FOREIGN KEY (drug_key)
        REFERENCES  dim_drug (drug_key)
        ON DELETE RESTRICT
        ON UPDATE CASCADE,

    CONSTRAINT fk_fact_article
        FOREIGN KEY (article_key)
        REFERENCES  dim_article (article_key)
        ON DELETE SET NULL
        ON UPDATE CASCADE,

    CONSTRAINT fk_fact_date
        FOREIGN KEY (date_key)
        REFERENCES  dim_date (date_key)
        ON DELETE RESTRICT
        ON UPDATE CASCADE
)
ENGINE = InnoDB
DEFAULT CHARSET = utf8mb4
COLLATE = utf8mb4_unicode_ci
COMMENT = 'Fact table - one bioactivity measurement per (drug, protein, publication)';


SET FOREIGN_KEY_CHECKS = 1;