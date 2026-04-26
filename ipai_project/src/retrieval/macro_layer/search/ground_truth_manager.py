import os
import json
from pathlib import Path

class GroundTruthManager:
    """
    Manages the creation, storage, and retrieval of the Ground Truth (QRELS) dataset.
    Updated for Eliqsir v2: Semantic Prefixes & Entity-Driven Queries.
    """
    def __init__(self):
        env_data_dir = os.getenv("DATA_DIR")
        if not env_data_dir:
            raise ValueError("DATA_DIR is not set. Make sure dotenv is loaded!")
            
        self.data_dir = Path(env_data_dir)
        self.corpus_path = self.data_dir / "documents" / "searchable_corpus.jsonl"
        self.file_path = self.data_dir / "documents" / "qrels.json"
        
        # 10 Refined, Complex Scientific Queries
        self.queries = {
            "q1": "EGFR kinase inhibitors in Homo sapiens", # Target + Organism
            "q2": "X-ray crystal structure of protein-ligand complexes", # Structure Method
            "q3": "Cyclic nucleotide phosphodiesterase family inhibitors", # Protein Family
            "q4": "potency of Imatinib mesylate measured by IC50 or Ki", # Compound + Measurement
            "q5": "Small molecule antagonists for GPCR targets", # Molecule Type + Target
            "q6": "treatment of triple-negative breast cancer with doxorubicin", # Clinical + Drug
            "q7": "PDE2A and P2RY12 dual target inhibition", # Multiple Genes
            "q8": "toxicity and cell viability assays in mitochondrial stress", # Assay Context
            "q9": "mechanism of resistance to platinum-based chemotherapy", # Complex Biological Process
            "q10": "binding affinity of monoclonal antibodies after 2020" # Type + Temporal context
        }
        self.qrels = self._load_qrels()

    def _load_qrels(self) -> dict:
        """Loads existing annotations from disk, or initializes empty lists."""
        if self.file_path.exists():
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {qid: [] for qid in self.queries}

    def save_qrels(self):
        """Saves current annotations to disk."""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.qrels, f, indent=4)
        print(f"Ground Truth successfully saved to {self.file_path.name}")

    def add_relevant(self, qid: str, article_key: str):
        """Adds a document key to a specific query's ground truth."""
        if qid in self.qrels:
            # Ensure article_key is stored as a string for JSON consistency
            ak_str = str(article_key)
            if ak_str not in self.qrels[qid]:
                self.qrels[qid].append(ak_str)
                print(f"  [+] Added {ak_str} to {qid}")

    def remove_relevant(self, qid: str, article_key: str):
        """Removes a document key in case of a manual annotation mistake."""
        if qid in self.qrels:
            ak_str = str(article_key)
            if ak_str in self.qrels[qid]:
                self.qrels[qid].remove(ak_str)
                print(f"  [-] Removed {ak_str} from {qid}")

    def get_annotation_stats(self):
        """Prints a summary of how many documents have been judged relevant per query."""
        print("\n=== Annotation Progress ===")
        total = 0
        for qid, docs in self.qrels.items():
            count = len(docs)
            total += count
            print(f"  {qid}: {count} relevant documents")
        print(f"Total Annotations: {total}\n")

    def bootstrap_from_model(self, search_function, top_k: int = 5):
        """
        Automatically populates the Ground Truth draft using a provided search function.
        Useful for building a baseline that will be manually refined later.
        """
        print(f"Bootstrapping Ground Truth for {len(self.queries)} optimized queries...")
        total_added = 0
        
        for qid, query in self.queries.items():
            results = search_function(query, top_k=top_k)
            for res in results:
                self.add_relevant(qid, res['article_key'])
                total_added += 1
                
        self.save_qrels()
        print(f"Success! Bootstrapped {total_added} relevance judgments.")