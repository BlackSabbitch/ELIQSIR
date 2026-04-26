import os
import json
import pandas as pd
import warnings
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from src.database.connection_manager import db_manager

warnings.filterwarnings('ignore', category=UserWarning)

class DocumentFactory:
    def __init__(self, chunk_size: int = 500):
        self.data_dir = Path(os.getenv('DATA_DIR', 'data'))
        self.output_dir = self.data_dir / "documents"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "searchable_corpus.jsonl"
        
        self.chunk_size = chunk_size
        self.date_map = self._load_date_map()

    def _load_date_map(self) -> dict:
        """Fetch all dates from the DWH date dimension."""
        print("Pre-loading date dimension...")
        query = "SELECT date_key, full_date FROM dim_date"
        with db_manager.get_dwh_connection() as conn:
            df_dates = pd.read_sql(query, conn)
        return df_dates.set_index('date_key')['full_date'].astype(str).to_dict()

    def _get_all_article_keys(self) -> List[int]:
        query = "SELECT article_key FROM dim_article WHERE pubmed_id IS NOT NULL AND TRIM(pubmed_id) != ''"
        with db_manager.get_dwh_connection() as conn:
            df_keys = pd.read_sql(query, conn)
        return df_keys['article_key'].tolist()

    def _get_data_chunk(self, keys_chunk: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch expanded metadata and facts."""
        keys_str = ",".join(map(str, keys_chunk))
        
        # Extended metadata from dim_article
        q_articles = f"""
            SELECT article_key, pubmed_id, doi, article_title, abstract, journal, authors, year AS pub_year
            FROM dim_article 
            WHERE article_key IN ({keys_str})
        """
        
        # Extended facts including all lifecycle date keys and measurement types
        q_facts = f"""
            SELECT 
                f.article_key, 
                f.publication_date_key, f.accepted_date_key, f.received_date_key,
                f.revised_date_key, f.epub_date_key, f.ppub_date_key,
                f.assay_organism, f.confidence_score, f.pchembl_value, f.standard_type, f.assay_description,
                d.drug_name, d.drug_chembl_id, d.standard_inchi_key, d.molecule_type,
                p.protein_name, p.uniprot_id, p.gene_names, p.protein_families,
                s.pdb_id, s.method as structure_method
            FROM fact_bioactivity f
            JOIN dim_drug d ON f.drug_key = d.drug_key
            JOIN dim_protein p ON f.protein_key = p.protein_key
            LEFT JOIN dim_structure s ON p.protein_key = s.protein_key
            WHERE f.article_key IN ({keys_str})
        """
        
        with db_manager.get_dwh_connection() as conn:
            df_articles = pd.read_sql(q_articles, conn)
            df_facts = pd.read_sql(q_facts, conn)
            
        return df_articles, df_facts

    def _process_dataframe(self, df_articles: pd.DataFrame, df_facts: pd.DataFrame) -> List[dict]:
        """Deduplicates facts and assembles the final JSON structure."""
        
        # Map all lifecycle dates
        date_cols = ['publication_date_key', 'accepted_date_key', 'received_date_key', 
                     'revised_date_key', 'epub_date_key', 'ppub_date_key']
        
        for col in date_cols:
            new_col_name = col.replace('_date_key', '_date')
            df_facts[new_col_name] = df_facts[col].map(self.date_map)

        get_unique_list = lambda x: list(set(x.dropna()))

        # Aggregate facts by article_key to prevent Cartesian product duplication
        facts_grouped = df_facts.groupby('article_key', dropna=False).agg({
            'drug_chembl_id': get_unique_list,
            'standard_inchi_key': get_unique_list,
            'uniprot_id': get_unique_list,
            'pdb_id': get_unique_list,
            'drug_name': get_unique_list,
            'molecule_type': get_unique_list,
            'protein_name': get_unique_list,
            'gene_names': get_unique_list,
            'protein_families': get_unique_list,
            'assay_organism': get_unique_list,
            'standard_type': get_unique_list,
            'assay_description': get_unique_list,
            'structure_method': get_unique_list,
            'confidence_score': list,
            'pchembl_value': list,
            # Dates: take the first non-null value for the article
            'publication_date': 'first',
            'accepted_date': 'first',
            'received_date': 'first',
            'revised_date': 'first',
            'epub_date': 'first',
            'ppub_date': 'first'
        }).reset_index()

        final_df = pd.merge(df_articles, facts_grouped, on='article_key', how='inner')

        docs = []
        for _, row in final_df.iterrows():
            # Constructing Searchable Prefix for BM25
            prefix_parts = []
            
            proteins = ", ".join(row['protein_name'])
            genes = ", ".join(row['gene_names'])
            families = ", ".join(row['protein_families'])
            organisms = ", ".join(row['assay_organism'])
            drugs = ", ".join(row['drug_name'])
            mol_types = ", ".join(row['molecule_type'])
            measurements = ", ".join(row['standard_type'])
            methods = ", ".join(row['structure_method'])
            assays = " ".join(row['assay_description'])

            if proteins: prefix_parts.append(f"Target: {proteins}.")
            if genes: prefix_parts.append(f"Genes: {genes}.")
            if families: prefix_parts.append(f"Family: {families}.")
            if organisms: prefix_parts.append(f"Organism: {organisms}.")
            if drugs: prefix_parts.append(f"Compound: {drugs}.")
            if mol_types: prefix_parts.append(f"Type: {mol_types}.")
            if measurements: prefix_parts.append(f"Measurement: {measurements}.")
            if methods: prefix_parts.append(f"Structure method: {methods}.")
            if row['journal']: prefix_parts.append(f"Journal: {row['journal']}.")
            if row['authors']: prefix_parts.append(f"Authors: {row['authors']}.")
            if assays: prefix_parts.append(f"Assay context: {assays}.")
            
            prefix_str = " ".join(prefix_parts)
            
            # Final corpus string (Title is repeated for weight boost)
            corpus = f"{row['article_title']} {row['article_title']} {prefix_str} Abstract: {row['abstract']}"
            corpus = " ".join(corpus.split()) # Clean double spaces
            
            docs.append({
                "article_key": int(row['article_key']),
                "natural_keys": {
                    "pubmed_id": str(row['pubmed_id']),
                    "doi": str(row['doi']) if pd.notna(row['doi']) else None,
                    "chembl_ids": row['drug_chembl_id'],
                    "uniprot_ids": row['uniprot_id'],
                    "pdb_ids": row['pdb_id']
                },
                "metadata": {
                    "title": row['article_title'],
                    "year": int(row['pub_year']) if pd.notna(row['pub_year']) else None,
                    "journal": row['journal'],
                    "authors": row['authors'],
                    "lifecycle_dates": {
                        "received": row['received_date'],
                        "revised": row['revised_date'],
                        "accepted": row['accepted_date'],
                        "epub": row['epub_date'],
                        "ppub": row['ppub_date'],
                        "published_primary": row['publication_date']
                    },
                    "scientific_context": {
                        "molecule_types": row['molecule_type'],
                        "measurement_types": row['standard_type'],
                        "structure_methods": row['structure_method']
                    }
                },
                "search_corpus": corpus,
                "metrics": {
                    "max_confidence": max(row['confidence_score'], default=0),
                    "avg_pchembl": round(pd.Series(row['pchembl_value']).mean(), 3) if row['pchembl_value'] else 0
                }
            })
        return docs

    def build_and_save(self):
        all_keys = self._get_all_article_keys()
        print(f"Building expanded corpus for {len(all_keys)} articles...")

        with open(self.output_file, 'w', encoding='utf-8') as f:
            for i in tqdm(range(0, len(all_keys), self.chunk_size)):
                keys_chunk = all_keys[i : i + self.chunk_size]
                df_articles, df_facts = self._get_data_chunk(keys_chunk)
                if df_articles.empty: continue
                
                for doc in self._process_dataframe(df_articles, df_facts):
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        
        print(f"Success! JSONL saved to: {self.output_file}")
        return self.output_file
    
    def get_document_statistics(self) -> dict:
        """Parses the generated JSONL file line-by-line to extract statistics."""
        if not self.output_file.exists():
            print(f"Error: Corpus file not found at {self.output_file}")
            return {}

        total_docs = 0
        sample_doc = None

        print(f"\nAnalyzing corpus: {self.output_file.name}...")
        
        with open(self.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if total_docs == 0:
                    sample_doc = json.loads(line)
                total_docs += 1

        stats = {
            "total_documents": total_docs,
            "file_path": str(self.output_file),
            "file_size_mb": round(self.output_file.stat().st_size / (1024 * 1024), 2),
            "sample": sample_doc
        }

        print(f"\nCORPUS STATISTICS")
        print(f"TOTAL DOCUMENTS: {stats['total_documents']}")
        print(f"FILE SIZE: {stats['file_size_mb']} MB")
        
        if sample_doc:
            print(f"\nSAMPLE DOCUMENT")
            print(json.dumps(sample_doc, indent=2, ensure_ascii=False))

        return stats