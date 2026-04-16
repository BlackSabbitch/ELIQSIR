import os
import time
import pandas as pd
from Bio import Entrez
from tqdm.notebook import tqdm
from pathlib import Path
from dotenv import load_dotenv

class PubMedDataEnricher_Omni:
    """
    A comprehensive tool to fetch article lifecycle dates from NCBI PubMed.
    Captures Submission, Revision, Acceptance, and Publication milestones.
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        load_dotenv()
        
        # Path configuration
        self.data_dir = os.getenv('DATA_DIR', 'data')
        self.output_dir = os.path.join(self.data_dir, "csv_cleaned")
        self.output_file = os.path.join(self.output_dir, "pubmed_lifecycle_complete.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Entrez API setup
        self.email = os.getenv('NCBI_EMAIL', 'admin@eliqsir-pipeline.com')
        Entrez.email = self.email
        Entrez.tool = "ELIQSIR_Data_Warehouse"
        
        self.default_date = 19000101
        
        # Status Mapping: Connecting PubMed XML tags to our DWH columns
        self.status_map = {
            'received': 'received_date_key',
            'revised': 'revised_date_key',
            'accepted': 'accepted_date_key',
            'pubmed': 'epub_date_key',    # Electronic publication fallback
            'medline': 'ppub_date_key',   # Print publication / Indexing
            'entrez': 'epub_date_key'     # System entry date
        }

    def get_pubmed_ids(self) -> list:
        """Retrieves unique PubMed IDs from the dim_article table in RDS."""
        print("Connecting to DWH to fetch target IDs...")
        query = "SELECT DISTINCT pubmed_id FROM dim_article WHERE pubmed_id IS NOT NULL"
        df = self.db_manager.fetch_to_dataframe(query)
        return df['pubmed_id'].tolist()

    def _format_date(self, date_dict: dict) -> int:
        """Converts Entrez date dictionary components into a YYYYMMDD integer."""
        try:
            year = str(date_dict.get('Year', '1900'))
            month = str(date_dict.get('Month', '1')).zfill(2)
            day = str(date_dict.get('Day', '1')).zfill(2)
            return int(f"{year}{month}{day}")
        except Exception:
            return self.default_date

    def fetch_batch(self, pmid_chunk: list) -> list:
        """Fetches and parses a batch of IDs from PubMed E-utilities."""
        # Initialize batch container with default values
        batch_results = {str(pid): {
            'pubmed_id': int(pid),
            'received_date_key': self.default_date, 'revised_date_key': self.default_date,
            'accepted_date_key': self.default_date, 'epub_date_key': self.default_date, 
            'ppub_date_key': self.default_date
        } for pid in pmid_chunk}
        
        try:
            id_string = ",".join(map(str, pmid_chunk))
            handle = Entrez.efetch(db="pubmed", id=id_string, retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            for article in records.get('PubmedArticle', []):
                pmid = str(article['MedlineCitation']['PMID'])
                history = article.get('PubmedData', {}).get('History', [])
                
                for history_date in history:
                    status = history_date.attributes.get('PubStatus')
                    if status in self.status_map:
                        col_name = self.status_map[status]
                        date_val = self._format_date(history_date)
                        
                        # Only update if we don't have a date yet (prioritizes first match)
                        if batch_results[pmid][col_name] == self.default_date:
                            batch_results[pmid][col_name] = date_val
                            
        except Exception as e:
            # Handle specific network/API hiccups
            if "IncompleteRead" in str(e):
                print(f"Network timeout (IncompleteRead) for batch starting {pmid_chunk[0]}")
            else:
                print(f"Batch Error: {e}")
                
        return list(batch_results.values())

    def run(self, batch_size: int = 250):
        """Orchestrates the enrichment process with incremental disk saving."""
        pmid_list = self.get_pubmed_ids()
        
        if not pmid_list:
            print("No IDs found. Aborting.")
            return

        print(f"Starting Omni-Enrichment for {len(pmid_list)} articles...")
        
        # 1. Initialize CSV with headers (overwrites old files)
        header_df = pd.DataFrame(columns=[
            'pubmed_id', 'received_date_key', 'revised_date_key', 
            'accepted_date_key', 'epub_date_key', 'ppub_date_key'
        ])
        header_df.to_csv(self.output_file, index=False)

        # 2. Loop through batches
        for i in tqdm(range(0, len(pmid_list), batch_size), desc="Processing Batches"):
            chunk = pmid_list[i:i + batch_size]
            batch_data = self.fetch_batch(chunk)
            
            # 3. Append results to CSV immediately to prevent data loss
            pd.DataFrame(batch_data).to_csv(
                self.output_file, mode='a', index=False, header=False
            )
            time.sleep(0.4) 

        print(f"\nAll set! Results saved to: {self.output_file}")