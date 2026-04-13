import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm
from dotenv import load_dotenv

class PMCDataEnricher:
    """
    A class to fetch missing article timeline dates from the Europe PMC API 
    and save them for DWH staging, using the custom ConnectionManager.
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        load_dotenv()
        
        self.data_dir = os.getenv('DATA_DIR')
        self.output_dir = os.path.join(self.data_dir, "csv_cleaned")
        self.output_file = os.path.join(self.output_dir, "article_enriched_dates.csv")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.email = os.getenv('NCBI_EMAIL')
        self.headers = {'User-Agent': f'ELIQSIR_Data_Pipeline ({self.email})'}

    def get_pubmed_ids(self) -> list:
        """Retrieves unique PubMed IDs using the ConnectionManager's fetch_to_dataframe."""
        print("Connecting to AWS RDS DWH via ConnectionManager...")
        query = "SELECT DISTINCT pubmed_id FROM dim_article WHERE pubmed_id IS NOT NULL"
        df = self.db_manager.fetch_to_dataframe(query)
        return df['pubmed_id'].tolist()

    def fetch_article_data(self, pubmed_id: str) -> dict:
        """Fetches the timeline history for a single PubMed ID from Europe PMC."""
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pubmed_id}&resultType=core&format=json"
        
        result_template = {
            'pubmed_id': pubmed_id,
            'received_date_key': 19000101,
            'revised_date_key': 19000101,
            'accepted_date_key': 19000101,
            'epub_date_key': 19000101,
            'ppub_date_key': 19000101
        }

        def format_key(d_str):
            if not d_str: return 19000101
            return int(d_str.replace('-', '').replace('/', ''))

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('hitCount', 0) > 0:
                    core = data['resultList']['result'][0]
                    hist = core.get('history', {})
                    
                    if 'received' in hist: result_template['received_date_key'] = format_key(hist['received'].get('date'))
                    if 'revised' in hist: result_template['revised_date_key'] = format_key(hist['revised'].get('date'))
                    if 'accepted' in hist: result_template['accepted_date_key'] = format_key(hist['accepted'].get('date'))
                    if 'electronicPublicationDate' in core: result_template['epub_date_key'] = format_key(core['electronicPublicationDate'])
                    if 'firstPublicationDate' in core: result_template['ppub_date_key'] = format_key(core['firstPublicationDate'])
        except Exception:
            pass 
        
        return result_template

    def run(self, max_workers: int = 10):
        pmid_list = self.get_pubmed_ids()
        
        if not pmid_list:
            print("No PubMed IDs found. Aborting.")
            return

        print(f"Successfully loaded {len(pmid_list)} unique PubMed IDs. Starting enrichment...")
        
        enriched_data = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.fetch_article_data, str(pmid)): pmid for pmid in pmid_list}
            
            for future in tqdm(as_completed(futures), total=len(pmid_list), desc="Fetching Dates"):
                enriched_data.append(future.result())

        # Save to CSV
        df_final = pd.DataFrame(enriched_data)
        df_final.to_csv(self.output_file, index=False)
        
        print(f"\nPipeline Complete! Data saved to: {self.output_file}")