# ELIQSIR – Domain-Specific Search Engine for Drug–Target Interactions

> **Course:** Information Integration and Analytic Data Processing 2025/2026  
> **Domain:** Human Proteome × Drug Bioactivity

ELIQSIR integrates four heterogeneous data sources into a MySQL star-schema data warehouse and exposes a search engine for querying drug–protein interaction evidence backed by PubMed literature.

---

## Table of Contents

- [ELIQSIR – Domain-Specific Search Engine for Drug–Target Interactions](#eliqsir--domain-specific-search-engine-for-drugtarget-interactions)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Architecture](#architecture)
  - [Quick Start](#quick-start)
    - [1 – Clone and create a virtual environment](#1--clone-and-create-a-virtual-environment)
    - [2 – Configure the environment](#2--configure-the-environment)
    - [3 – Download ChEMBL](#3--download-chembl)
    - [4 – Create the MySQL database](#4--create-the-mysql-database)
    - [5 – Run the pipeline](#5--run-the-pipeline)
  - [Configuration](#configuration)
  - [Data Sources](#data-sources)
  - [Star Schema](#star-schema)
  - [Running the Pipeline](#running-the-pipeline)
  - [Running Tests](#running-tests)
  - [Notebooks](#notebooks)
  - [Project Structure](#project-structure)

---

## Project Overview

| Phase | What it does |
|-------|-------------|
| **Extraction** | Pulls proteins from UniProt, bioactivity data from a local ChEMBL MySQL DB (auto-loaded from the bundled dump), 3-D structures from PDBe, and article abstracts from NCBI PubMed. |
| **Transformation** | Cleans and normalises each dataset, then builds star-schema dimension and fact DataFrames. |
| **Loading** | Upserts the DataFrames into a MySQL data warehouse using idempotent `INSERT … ON DUPLICATE KEY UPDATE` statements. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      src/pipeline.py                    │
│   (orchestrates the three stages below)                 │
└────────────┬────────────────────────────────────────────┘
             │
    ┌────────▼────────┐    ┌──────────────────┐    ┌────────────────────┐
    │  src/extraction │───►│ src/transformation│───►│   src/loading      │
    │  UniProt        │    │  DataCleaner      │    │  WarehouseLoader   │
    │  ChEMBL         │    │  DimensionalModel │    │  (MySQL upserts)   │
    │  PDBe           │    │  Builder          │    └────────────────────┘
    │  PubMed         │    └──────────────────┘
    └─────────────────┘
             │
    ┌────────▼────────┐
    │  src/database   │
    │  schema.py      │  ← SQLAlchemy ORM (DDL)
    │  connection.py  │  ← Engine / Session factory
    └─────────────────┘
```

---

## Quick Start

### 1 – Clone and create a virtual environment

```bash
git clone https://github.com/BlackSabbitch/ELIQSIR.git
cd ELIQSIR
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# May help in the notebooks to have the kernel registered in Jupyter:
python -m ipykernel install --user --name=eliqsir-venv --display-name="Python (ELIQSIR .venv)"
```

### 2 – Configure the environment

```bash
cp .env.example .env
# Edit .env and fill in the required values (see Configuration section)
```

### 3 – Download ChEMBL

Download the MySQL dump from the [ChEMBLdb FTP server](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/)
([MySQL dump ~19 GB unpacked](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_mysql.tar.gz),
[schema docs](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/schema_documentation.html))
and **extract it into the project folder**:

```
ELIQSIR/
└── data/
    └── ChEMBL/
        └── chembl_36/                   # Extracted folder from the downloaded tar.gz file
            └── chembl_36_mysql/
                ├── chembl_36_mysql.dmp   ← place it here
                └── INSTALL_mysql
```

The pipeline will automatically create the `chembl_36` database and load the
dump on first run — no manual `mysql` commands needed.  Subsequent runs detect
that the database is already populated and skip the import entirely.

> **Manual import (optional):** If you prefer to load the dump yourself, follow
> the two-step process in `data/ChEMBLE/chembl_36/chembl_36_mysql/INSTALL_mysql`:
>
> ```sql
> -- Step 1 (MySQL shell):
> CREATE DATABASE chembl_36 DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;
> ```
> ```bash
> # Step 2 (shell):
> mysql -uUSERNAME -pPASSWORD [-hHOST -PPORT] chembl_36 < chembl_36_mysql.dmp
> ```

**If dumping is delayed try:** adding the following inside a new file like `sudo nano /etc/my.cnf.d/chembl-optimizations.cnf`
```bash
[mysqld]
innodb_buffer_pool_size = 2G
innodb_log_buffer_size = 64M
innodb_flush_log_at_trx_commit = 2
innodb_doublewrite = 0
bulk_insert_buffer_size = 512M
max_allowed_packet = 1G
```

### 4 – Create the MySQL database

```sql
CREATE DATABASE eliqsir_dw CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 5 – Run the pipeline

```bash
python -c "from src.pipeline import run_pipeline; run_pipeline()"
```

---

## Configuration

All settings live in `.env` (copied from `.env.example`). The `src/config.py` module reads these values at import time:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NCBI_EMAIL` | **Yes** | – | Your e-mail for NCBI API compliance |
| `MYSQL_USER` | **Yes** | – | MySQL username (shared by warehouse **and** ChEMBL) |
| `MYSQL_PASSWORD` | **Yes** | – | MySQL password |
| `MYSQL_HOST` | No | `localhost` | MySQL host |
| `MYSQL_PORT` | No | `3306` | MySQL port |
| `MYSQL_DB` | No | `eliqsir_dw` | Target warehouse database name |
| `CHEMBL_MYSQL_DB` | No | `chembl_36` | ChEMBL source database name |
| `CHEMBL_DUMP_DIR` | No | `data/ChEMBLE/chembl_36/chembl_36_mysql` | Directory containing `chembl_36_mysql.dmp` |
| `DATA_DIR` | No | `data/` | Directory for intermediate CSVs |
| `LOG_LEVEL` | No | `INFO` | Python logging level |
| `UNIPROT_ORGANISM_ID` | No | `9606` | NCBI taxonomy ID (9606 = Human) |
| `PDBE_REQUEST_DELAY_S` | No | `0.1` | Seconds between PDBe requests |
| `NCBI_BATCH_SIZE` | No | `200` | PubMed IDs per request batch |

---

## Data Sources

| # | Source | Type | Records (approx.) | Key Attributes |
|---|--------|------|-------------------|----------------|
| 1 | UniProt Swiss-Prot | REST API | ~20 000 | `accession`, `gene_names`, `protein_name` |
| 2 | ChEMBL 36 | Local MySQL | ~millions | `drug_chembl_id`, `standard_value`, `assay_type`, `pubmed_id` |
| 3 | PDBe Graph API | REST API | variable | `pdb_id`, `resolution`, `coverage` |
| 4 | NCBI PubMed | E-utils API | variable | `pubmed_id`, `abstract` |

---

## Star Schema

```
                        ┌──────────────┐
                        │  DimProtein  │
                        │─────────────│
                        │ protein_key  │◄──────────────┐
                        │ uniprot_id   │               │
                        │ gene_names   │        ┌──────┴──────────────┐
                        │ protein_name │        │  DimStructure       │
                        └──────┬───────┘        │─────────────────────│
                               │                │ structure_key        │
                               │                │ pdb_id, chain_id     │
              ┌────────────────▼───────────────┐│ resolution, coverage │
              │       FactBioactivity          ││ method, unp_start/end│
              │───────────────────────────────┐└─────────────────────┘
              │ activity_id (PK)              │
              │ protein_key  (FK)             │
              │ drug_key     (FK)             │   ┌─────────────┐
              │ article_key  (FK, nullable)   ├──►│   DimDrug   │
              │ standard_type                 │   │─────────────│
              │ standard_value  ← measure     │   │ drug_key     │
              │ pchembl_value   ← measure     │   │ drug_chembl_id│
              │ confidence_score← measure     │   │ drug_name    │
              │ assay_type (degenerate dim.)  │   └─────────────┘
              │ assay_description             │
              └───────────────────────────────┘   ┌───────────────┐
                               │                  │  DimArticle   │
                               └─────────────────►│───────────────│
                                                  │ article_key   │
                                                  │ pubmed_id     │
                                                  │ article_title │
                                                  │ journal, year │
                                                  │ abstract ← IR corpus
                                                  └───────────────┘
```

**Grain:** One unique bioactivity measurement per (drug, protein, publication) triple.

---

## Running the Pipeline

```python
from src.pipeline import run_pipeline

# Dry run – saves Parquet files, no DB load  (default format = parquet)
summary = run_pipeline(load_to_db=False, save_csv=True)

# Save as Pickle (preserves nullable dtypes exactly, Python-only)
summary = run_pipeline(load_to_db=False, save_csv=True, save_format="pickle")

# Save as CSV (human-readable, loses typed dtypes on round-trip)
summary = run_pipeline(load_to_db=False, save_csv=True, save_format="csv")

# Full run with MySQL
summary = run_pipeline(load_to_db=True, save_csv=True)

print(summary["row_counts"])
summary["dim_protein"].head()
```

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/IPAI_step1.ipynb` | Original procedural extraction notebook |
| `notebooks/01_extraction_demo.ipynb` | **Step 1** — OOP extractors demo; saves raw Parquet to `data/raw/` |
| `notebooks/02_transformation_and_loading.ipynb` | **Step 2** — Clean, build star schema, save staging Parquet, load to warehouse |

---

## Project Structure

```
ELIQSIR/
├── .env                     # Secrets (not committed)
├── .env.example             # Template for .env
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                 # Raw Parquet from Step 1
│   ├── staging/             # Cleaned Parquet (dimensions + fact) from Step 2
│   ├── PDB/
│   ├── PubMedAbstracts/
│   └── UniProt/
│
├── notebooks/
│   ├── IPAI_step1.ipynb
│   ├── 01_extraction_demo.ipynb
│   └── 02_transformation_and_loading.ipynb
│
├── src/
│   ├── config.py            # Settings singleton (reads .env)
│   ├── pipeline.py          # ETL orchestrator (parquet/pickle/csv output)
│   ├── extraction/
│   │   ├── uniprot_extractor.py
│   │   ├── chembl_extractor.py
│   │   ├── pdbe_extractor.py
│   │   └── pubmed_extractor.py
│   ├── transformation/
│   │   ├── cleaner.py
│   │   └── dimensional_builder.py
│   ├── loading/
│   │   └── warehouse_loader.py
│   ├── database/
│   │   ├── connection.py
│   │   ├── schema.py        # SQLAlchemy ORM
│   │   └── schema.sql       # MySQL DDL (authoritative source)
│   └── utils/
│       └── logging_config.py
│
└── tests/
    ├── test_extraction.py
    ├── test_transformation.py
    └── test_loading.py
```
