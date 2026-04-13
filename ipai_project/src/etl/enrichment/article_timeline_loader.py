import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

class EnrichmentLoader:
    """
    Handles the transition of enriched article data from CSV 
    to the AWS RDS Data Warehouse staging and fact tables.
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        load_dotenv()
        
        # Define paths
        self.data_dir = os.getenv('DATA_DIR')
        self.csv_path = os.path.join(self.data_dir, "csv_cleaned", "article_enriched_dates.csv")
        
    def _prepare_data(self) -> pd.DataFrame:
        """Reads CSV and ensures data types are SQL-friendly."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Enriched CSV not found at {self.csv_path}")
            
        df = pd.read_csv(self.csv_path)
        # Convert NaN to None for proper SQL NULL insertion
        df = df.replace({np.nan: None})
        return df

    def load_to_staging(self):
        """Uploads CSV data into the stg_article_dates_enrichment table."""
        df = self._prepare_data()
        print(f"Loading {len(df)} records into staging table...")
        
        insert_sql = """
        INSERT INTO stg_article_dates_enrichment 
        (pubmed_id, received_date_key, revised_date_key, accepted_date_key, epub_date_key, ppub_date_key)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        data_to_load = [tuple(x) for x in df.to_numpy().tolist()]
        
        with self.db_manager.get_dwh_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("TRUNCATE TABLE stg_article_dates_enrichment")
            
            chunk_size = 10000
            for i in range(0, len(data_to_load), chunk_size):
                chunk = data_to_load[i:i + chunk_size]
                cursor.executemany(insert_sql, chunk)
                conn.commit()
        print("Staging load successful.")

    def run_update(self):
        """Executes the final SQL UPDATE to enrich the fact_bioactivity table."""
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
            print(f"Update complete. Rows affected: {cursor.rowcount}")

    def get_validation_summary(self) -> pd.DataFrame:
        """
        Fetches enrichment statistics from staging and returns a 
        formatted DataFrame for clear reporting.
        """
        query = """
        SELECT 
            COUNT(*) as total_rows,
            SUM(CASE WHEN received_date_key != 19000101 THEN 1 ELSE 0 END) as received,
            SUM(CASE WHEN revised_date_key != 19000101 THEN 1 ELSE 0 END) as revised,
            SUM(CASE WHEN accepted_date_key != 19000101 THEN 1 ELSE 0 END) as accepted,
            SUM(CASE WHEN epub_date_key != 19000101 THEN 1 ELSE 0 END) as epub,
            SUM(CASE WHEN ppub_date_key != 19000101 THEN 1 ELSE 0 END) as ppub
        FROM stg_article_dates_enrichment
        """
        df_raw = self.db_manager.fetch_to_dataframe(query)
        
        if df_raw.empty or df_raw['total_rows'][0] == 0:
            return pd.DataFrame(columns=['date_type', 'found_count', 'coverage_pct'])

        total = int(df_raw['total_rows'][0])
        
        report = df_raw.drop(columns=['total_rows']).T.reset_index()
        report.columns = ['date_type', 'found_count']
        
        report['coverage_pct'] = (report['found_count'] / total * 100).round(2)
        
        return report