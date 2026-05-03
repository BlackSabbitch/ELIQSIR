import json
import random
from pathlib import Path
from typing import List, Dict

class TrecPoolBuilder:
    """
    Implements TREC-style depth-K pooling for Eliqsir v2.
    Blends results from BM25, VSM (TF-IDF), and Boolean search to create 
    an unbiased, high-quality Ground Truth dataset.
    """
    def __init__(self, retriever, gt_manager):
        """
        :param retriever: An instance of LexicalRetriever.
        :param gt_manager: An instance of GroundTruthManager.
        """
        self.retriever = retriever
        self.gt = gt_manager
        # Pre-load metadata to provide a rich UI for the human annotator
        self.docs_info = self._load_article_data()

    def _load_article_data(self) -> Dict[str, dict]:
        """
        Parses the searchable_corpus.jsonl once to extract clean snippets 
        for the human annotator.
        """
        data = {}
        if not self.gt.corpus_path.exists():
            print(f"WARNING: Corpus not found at {self.gt.corpus_path}")
            return data
            
        print("Loading and indexing metadata for annotation interface...")
        with open(self.gt.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                key = str(doc['article_key'])
                
                # Extract data using the Eliqsir v2 JSON schema
                meta = doc.get('metadata', {})
                title = meta.get('title', 'No Title')
                
                # Parse the search_corpus to separate entities from abstract text
                full_text = doc.get('search_corpus', '')
                try:
                    # Split by the 'Abstract:' marker to isolate prefixed entities
                    entities_part = full_text.split('Abstract:')[0]
                    # Remove the first occurrence of the title (boosted in the index)
                    clean_entities = entities_part.replace(title, "", 1).strip()
                    # Get a preview of the abstract
                    abstract_preview = full_text.split('Abstract:')[-1][:400].strip() + "..."
                except Exception:
                    clean_entities = "Metadata parsing error"
                    abstract_preview = full_text[:400]

                data[key] = {
                    "title": title,
                    "journal": meta.get('journal', 'N/A'),
                    "year": meta.get('year', 'N/A'),
                    "entities": clean_entities,
                    "abstract": abstract_preview
                }
        print(f"Successfully cached {len(data)} document snippets.")
        return data

    def generate_blind_pool(self, qid: str, top_k: int = 5):
        """
        Executes search via three models, merges results into a pool,
        shuffles them to hide the source model, and displays them for review.
        """
        query = self.gt.queries.get(qid)
        if not query:
            print(f"ERROR: Query ID '{qid}' not found in GroundTruthManager.")
            return

        print(f"\nBLIND ANNOTATION POOL | ID: {qid}")
        print(f"QUERY: '{query}'")
 
        # 1. Collect results from three independent search engines
        bm25_res = self.retriever.search_bm25(query, top_k=top_k)
        tfidf_res = self.retriever.search_vsm(query, top_k=top_k)
        # Use Boolean OR with a 0.5 similarity threshold (at least 50% words matched)
        bool_res = self.retriever.search_boolean(query, operator="OR", min_similarity=0.5)
        
        # Ensure all keys are strings for consistent comparison
        bm25_keys = [str(r['article_key']) for r in bm25_res]
        tfidf_keys = [str(r['article_key']) for r in tfidf_res]
        bool_keys = [str(r['article_key']) for r in bool_res[:top_k]]

        # 2. Create the unified Pool
        # Using a set automatically handles overlapping results (deduplication)
        pool = list(set(bm25_keys + tfidf_keys + bool_keys))

        # 3. Shuffle the pool to prevent annotation bias (Blind Test)
        # Fixed seed ensures the order remains the same if the cell is re-run
        random.seed(42) 
        random.shuffle(pool)

        print(f"\nPool Stats: {len(pool)} unique documents collected.")
        print(f"Sources: BM25({len(bm25_keys)}), TF-IDF({len(tfidf_keys)}), Boolean({len(bool_keys)})")
        print("\nINSTRUCTIONS: Review the snippets below. If relevant, use: gt_manager.add_relevant(qid, key)\n")

        # 4. Display "Article Cards" for the annotator
        for i, key in enumerate(pool, 1):
            info = self.docs_info.get(key)
            if not info:
                print(f"DOCUMENT {i}/{len(pool)} | KEY: [ {key} ] (MISSING IN CACHE)")
                continue

            print(f"DOCUMENT {i}/{len(pool)} | KEY: [ {key} ]")
            print(f"TITLE:    {info['title']}")
            print(f"SOURCE:  {info['journal']} | YEAR: {info['year']}")
            print(f"ENTITIES: {info['entities']}")
            print(f"ABSTRACT: {info['abstract']}")
