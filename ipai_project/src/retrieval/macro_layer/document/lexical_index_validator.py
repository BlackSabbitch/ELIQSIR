import os
import joblib
from pathlib import Path
import scipy.sparse as sp

class LexicalIndexValidator:
    """
    Validates the integrity, alignment, dimensionality, and vocabulary 
    of the generated ELIQSIR lexical indexes.
    """
    def __init__(self):
        env_data_dir = os.getenv("DATA_DIR")
        if not env_data_dir:
            raise ValueError("DATA_DIR is not set. Make sure dotenv is loaded!")
            
        self.data_dir = Path(env_data_dir)
        self.index_path = self.data_dir / "indexes" / "lexical"
        self.corpus_path = self.data_dir / "documents" / "searchable_corpus.jsonl"
        
        self.expected_docs = self._get_expected_doc_count()

    def _get_expected_doc_count(self) -> int:
        if not self.corpus_path.exists():
            return 0
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def run_all_tests(self):
        """Executes the standard QA suite on the generated artifacts."""
        if self.expected_docs == 0:
            print(f"CRITICAL ERROR: Source corpus not found or empty at {self.corpus_path}")
            return

        print(f"STARTING INDEX QUALITY ASSURANCE (Expected Docs: {self.expected_docs})\n")
        
        # 1. Structural Checks
        self._test_file_existence()
        self._test_streaming_corpus()
        self._test_matrix_dimensions()
        
        # 2. Semantic Checks (The new integration)
        entities_to_test = [
            "target egfr", 
            "compound imatinib", 
            "measurement ic50", 
            "structure method"
        ]
        self.inspect_vocabulary(sample_size=10, test_terms=entities_to_test)
        
        print("ALL TESTS COMPLETED SUCCESSFULLY")

    def _test_file_existence(self):
        """Checks if all 8 required artifacts were created and have non-zero size."""
        print("Test 1: File Existence & Size")
        expected_files = [
            "processed_corpus.txt", 
            "article_keys.txt", 
            "article_keys.joblib", 
            "tfidf_vectorizer.joblib", 
            "tfidf_matrix.joblib", 
            "bow_vectorizer.joblib", 
            "bow_matrix.joblib",
            "bm25_index.joblib"
        ]
        
        for filename in expected_files:
            filepath = self.index_path / filename
            assert filepath.exists(), f"CRITICAL: Missing file -> {filename}"
            size_mb = filepath.stat().st_size / (1024 * 1024)
            assert size_mb > 0, f"CRITICAL: File is empty -> {filename}"
            print(f"  [OK] {filename} found ({size_mb:.2f} MB)")

    def _test_streaming_corpus(self):
        """Verifies that the text files have the exact expected number of lines."""
        print("\nTest 2: Streaming Corpus Alignment")
        with open(self.index_path / "processed_corpus.txt", 'r', encoding='utf-8') as f:
            corpus_lines = sum(1 for _ in f)
        with open(self.index_path / "article_keys.txt", 'r', encoding='utf-8') as f:
            key_lines = sum(1 for _ in f)

        assert corpus_lines == self.expected_docs, f"Expected {self.expected_docs} docs, got {corpus_lines}"
        assert corpus_lines == key_lines, "CRITICAL: Mismatch between text lines and keys!"
        print("  [OK] Text corpus and keys are aligned.")

    def _test_matrix_dimensions(self):
        """Verifies that matrix shapes match the vocabulary and document counts."""
        print("\nTest 3: Matrix & Vocabulary Dimensions")
        tfidf_mat = joblib.load(self.index_path / "tfidf_matrix.joblib")
        tfidf_vec = joblib.load(self.index_path / "tfidf_vectorizer.joblib")
        keys = joblib.load(self.index_path / "article_keys.joblib")

        assert sp.issparse(tfidf_mat), "TF-IDF matrix is not sparse!"
        assert tfidf_mat.shape[0] == len(keys), "Row count mismatch!"
        assert tfidf_mat.shape[1] == len(tfidf_vec.vocabulary_), "Col count mismatch!"
        print(f"  [OK] Matrices properly shaped: {tfidf_mat.shape[0]} docs x {tfidf_mat.shape[1]} terms")

    def inspect_vocabulary(self, sample_size: int = 10, test_terms: list = None):
        """
        Prints vocabulary statistics, a random sample of n-grams, and 
        verifies if specific key entities survived the indexing thresholds.
        """
        print("\nTest 4: Term Vocabulary Inspection")
        try:
            tfidf_vec = joblib.load(self.index_path / "tfidf_vectorizer.joblib")
        except FileNotFoundError:
            print("  [SKIP] tfidf_vectorizer.joblib not found. Run the Indexer first.")
            return

        vocab = tfidf_vec.vocabulary_
        vocab_size = len(vocab)
        print(f"  Total Unique Terms (Unigrams & Bigrams): {vocab_size:,}\n")
        
        print(f"Alphabetical Sample ({sample_size} terms)")
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[0])
        
        # Take a slice from the middle to avoid numbers/symbols at the very beginning
        mid_point = max(0, vocab_size // 2)
        for term, idx in sorted_vocab[mid_point : mid_point + sample_size]:
            print(f"    '{term}': {idx}")
            
        if test_terms:
            print("\nSpecific Entity Check")
            for term in test_terms:
                if term in vocab:
                    print(f"    [FOUND] '{term}' -> Index: {vocab[term]}")
                else:
                    print(f"    [MISSING] '{term}'")