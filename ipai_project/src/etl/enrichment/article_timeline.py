import requests
import time

def fetch_full_article_timeline(pubmed_id: str) -> dict:
    # Set default keys to 19000101 (Unknown date in dim_date)
    timeline = {
        'pubmed_id': pubmed_id,
        'received_date_key': 19000101,
        'revised_date_key': 19000101,
        'accepted_date_key': 19000101,
        'epub_date_key': 19000101,
        'ppub_date_key': 19000101
    }

    # Europe PMC REST API URL
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pubmed_id}&resultType=core&format=json"

    try:
        response = requests.get(url)
        data = response.json()

        # Check if we got any results for this PubMed ID
        if data.get('hitCount', 0) == 0:
            return timeline

        result = data['resultList']['result'][0]
        history = result.get('history', {})

        # Helper function to convert 'YYYY/MM/DD' to YYYYMMDD integer
        def format_date_key(date_str):
            if not date_str: return 19000101
            return int(date_str.replace('/', '').replace('-', ''))

        # Extract dates from the history block if they exist
        if 'received' in history and 'date' in history['received']:
            timeline['received_date_key'] = format_date_key(history['received']['date'])
            
        if 'revised' in history and 'date' in history['revised']:
            timeline['revised_date_key'] = format_date_key(history['revised']['date'])
            
        if 'accepted' in history and 'date' in history['accepted']:
            timeline['accepted_date_key'] = format_date_key(history['accepted']['date'])

        # Electronic publication date (Online)
        if 'electronicPublicationDate' in result:
            timeline['epub_date_key'] = format_date_key(result['electronicPublicationDate'])

        # Print publication date (First publication)
        if 'firstPublicationDate' in result:
            timeline['ppub_date_key'] = format_date_key(result['firstPublicationDate'])

    except Exception as e:
        # Log the error and continue
        print(f"Error fetching data for PMID {pubmed_id}: {e}")

    return timeline