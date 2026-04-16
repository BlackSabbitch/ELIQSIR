import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

class EnrichmentLoader:
    """
    Handles the transition of enriched article data from CSV 
    to the AWS RDS Data Warehouse staging and fact tables.
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        load_dotenv()
        self.data_dir = os.getenv('DATA_DIR', 'data')
        self.csv_path = os.path.join(self.data_dir, "pubmed_lifecycle_complete.csv")
        
    def _prepare_data(self) -> list:
        """
        Reads CSV and strict-casts data types to native Python types 
        to avoid numpy.int64 interface errors in SQL.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Enriched CSV not found at {self.csv_path}")
            
        df = pd.read_csv(self.csv_path)
        data_to_load = []
        
        # Iterate and explicitly cast to prevent numpy type leakage
        for _, row in df.iterrows():
            # Ensure PMID is handled as string/int consistently
            pubmed_id = str(int(row['pubmed_id'])) 
            
            # Using our proven integer mapping
            rec = int(row['received_date_key'])
            rev = int(row['revised_date_key'])
            acc = int(row['accepted_date_key'])
            epub = int(row['epub_date_key'])
            ppub = int(row['ppub_date_key'])
            
            data_to_load.append((pubmed_id, rec, rev, acc, epub, ppub))
            
        return data_to_load

    def load_to_staging(self):
        """Uploads sanitized data into the stg_article_dates_enrichment table."""
        data_to_load = self._prepare_data()
        print(f"Loading {len(data_to_load)} records into staging layer...")
        
        insert_sql = """
        INSERT INTO stg_article_dates_enrichment 
        (pubmed_id, received_date_key, revised_date_key, accepted_date_key, epub_date_key, ppub_date_key)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        with self.db_manager.get_dwh_connection() as conn:
            cursor = conn.cursor()
            # TRUNCATE ensures we don't double-count data on retries
            cursor.execute("TRUNCATE TABLE stg_article_dates_enrichment")
            
            chunk_size = 5000 
            for i in range(0, len(data_to_load), chunk_size):
                chunk = data_to_load[i:i + chunk_size]
                cursor.executemany(insert_sql, chunk)
                conn.commit()
                
        print("Staging load successful.")

    def run_update(self):
        """
        Executes the final SQL UPDATE-JOIN to enrich the fact_bioactivity table.
        This propagates dates from staging -> dim_article -> fact_bioactivity.
        """
        print("Synchronizing fact_bioactivity with enriched dates...")
        
        update_sql = """
        UPDATE fact_bioactivity f
        JOIN dim_article a ON f.article_key = a.article_key
        JOIN stg_article_dates_enrichment stg ON a.pubmed_id = stg.pubmed_id
        SET 
            f.received_date_key = stg.received_date_key,
            f.revised_date_key = stg.revised_date_key,
            f.accepted_date_key = stg.accepted_date_key,
            f.epub_date_key = stg.epub_date_key,
            f.ppub_date_key = stg.ppub_date_key;
        """
        
        with self.db_manager.get_dwh_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(update_sql)
            conn.commit()
            print(f"Fact synchronization complete. Rows affected: {cursor.rowcount}")
