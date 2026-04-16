from Bio import Entrez
import time

def fetch_full_article_timeline(pubmed_id: str, email: str = "samosudova@gmail.com") -> dict:
    # Initialize the default timeline dictionary with 19000101
    timeline = {
        'pubmed_id': pubmed_id,
        'received_date_key': 19000101,
        'revised_date_key': 19000101,
        'accepted_date_key': 19000101,
        'epub_date_key': 19000101,
        'ppub_date_key': 19000101
    }

    # NCBI requires an email address for API usage
    Entrez.email = email

    try:
        # Fetch data from NCBI PubMed in XML format
        handle = Entrez.efetch(db="pubmed", id=pubmed_id, retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        # Check if we got any valid article records
        if not records.get('PubmedArticle'):
            return timeline

        article = records['PubmedArticle'][0]
        pubmed_data = article.get('PubmedData', {})
        history = pubmed_data.get('History', [])

        # Helper function to convert Entrez date objects to YYYYMMDD integer
        def format_date_key(date_obj):
            try:
                year = date_obj.get('Year', '1900')
                # NCBI returns months and days as strings without leading zeros
                month = str(date_obj.get('Month', '1')).zfill(2)
                day = str(date_obj.get('Day', '1')).zfill(2)
                return int(f"{year}{month}{day}")
            except Exception:
                return 19000101

        # Map PubMed statuses directly to our Data Warehouse column names
        status_mapping = {
            'received': 'received_date_key',
            'revised': 'revised_date_key',
            'accepted': 'accepted_date_key',
            'epublish': 'epub_date_key',
            'ppublish': 'ppub_date_key'
        }

        # Iterate through the history block and populate existing dates
        for history_date in history:
            status = history_date.attributes.get('PubStatus')
            if status in status_mapping:
                db_key = status_mapping[status]
                timeline[db_key] = format_date_key(history_date)

    except Exception as e:
        # Log the error and add sleep to avoid hammering the API on fail
        print(f"Error fetching data for PMID {pubmed_id}: {e}")
        time.sleep(1) 

    return timeline