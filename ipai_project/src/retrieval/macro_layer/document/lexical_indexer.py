import os
import json
import re
import joblib
from pathlib import Path
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from functools import lru_cache

class LexicalIndexer:
    """
    Builds TF-IDF, Bag-of-Words (BoW), and BM25 representations for the corpus.
    Memory-optimized for processing large JSONL files using generators and batched I/O.
    """
    
    SCIENTIFIC_STOP_WORDS = {
        'study', 'result', 'using', 'significantly', 'effect', 'activity', 
        'showed', 'data', 'research', 'analysis', 'method', 'concentration',
        'dose', 'values', 'compound', 'derivatives', 'inhibitor', 'active',
        'tested', 'based', 'potential', 'evaluated', 'clinical', 'treatment'
    }

    def __init__(self, corpus_path: str = None):
        # 1. Ensure NLTK resources are available
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK resources...")
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            
        # 2. Setup Paths
        env_data_dir = os.getenv("DATA_DIR")
        if not env_data_dir:
            raise ValueError("DATA_DIR is not set. Make sure dotenv is loaded!")
        
        self.data_dir = Path(env_data_dir)
        
        if corpus_path:
            self.corpus_path = Path(corpus_path)
        else:
            self.corpus_path = self.data_dir / "documents" / "searchable_corpus.jsonl"
            
        self.output_dir = self.data_dir / "indexes" / "lexical"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Setup NLP Pipeline Components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')).union(self.SCIENTIFIC_STOP_WORDS)
        self.lemmatize_cached = lru_cache(maxsize=10000)(self.lemmatizer.lemmatize)
        
        # Replace non-alphanumeric characters (including hyphens) with spaces safely
        self.clean_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        
        # 4. Setup Scikit-Learn Vectorizers
        self.params = {
            "lowercase": False, 
            "ngram_range": (1, 2), 
            "min_df": 2,          
            "max_df": 0.85,        
            "token_pattern": r'(?u)\b\w\w+\b'
        }

        self.tfidf_vectorizer = TfidfVectorizer(**self.params)
        self.bow_vectorizer = CountVectorizer(**self.params)
        
        # Calculate total documents dynamically for accurate progress bars
        self.total_docs = self._count_total_docs()

    def _count_total_docs(self) -> int:
        """Dynamically counts the number of lines in the corpus file."""
        if not self.corpus_path.exists():
            return 0
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def _preprocess_pipeline(self, text: str) -> str:
        """Standard NLP Pipeline optimized with pre-compiled regex and LRU Cache."""
        if not text:
            return ""

        text = self.clean_pattern.sub(' ', text.lower())
        tokens = text.split()

        cleaned_tokens = [
            self.lemmatize_cached(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 1
        ]
        
        return " ".join(cleaned_tokens)

    def build_index(self):
        """
        Builds the Lexical Index (TF-IDF, BoW, and BM25) with memory-safe streaming, 
        I/O batching, and automatic resumption from crashes.
        """
        if self.total_docs == 0:
            raise FileNotFoundError(f"Corpus file not found or empty: {self.corpus_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_file_path = self.output_dir / "processed_corpus.txt"
        keys_file_path = self.output_dir / "article_keys.txt"

        # PHASE 1: Checkpoint Recovery & Streaming
        processed_keys = set()
        if keys_file_path.exists():
            with open(keys_file_path, 'r', encoding='utf-8') as kf:
                processed_keys = {line.strip() for line in kf}
            print(f"Resuming indexing... Found {len(processed_keys)} completed documents.")

        print(f"Streaming and preprocessing {self.total_docs} documents to disk with batched I/O...")
        
        BATCH_SIZE = 1000
        batch_texts = []
        batch_keys = []

        with open(self.corpus_path, 'r', encoding='utf-8') as f_in, \
             open(processed_file_path, 'a', encoding='utf-8') as f_text, \
             open(keys_file_path, 'a', encoding='utf-8') as f_keys:
            
            for line in tqdm(f_in, total=self.total_docs, desc="Processing"):
                data = json.loads(line)
                key = str(data['article_key'])
                
                if key in processed_keys:
                    continue
                    
                processed_text = self._preprocess_pipeline(data.get('search_corpus', ''))
                
                batch_texts.append(processed_text + '\n')
                batch_keys.append(key + '\n')
                
                if len(batch_texts) >= BATCH_SIZE:
                    f_text.writelines(batch_texts)
                    f_keys.writelines(batch_keys)
                    f_text.flush()
                    f_keys.flush()
                    
                    batch_texts.clear()
                    batch_keys.clear()
            
            if batch_texts:
                f_text.writelines(batch_texts)
                f_keys.writelines(batch_keys)
                f_text.flush()
                f_keys.flush()

        # PHASE 2: Prepare Final Artifacts for Scikit-Learn
        with open(keys_file_path, 'r', encoding='utf-8') as kf:
            final_keys = [line.strip() for line in kf]
        joblib.dump(final_keys, self.output_dir / "article_keys.joblib")

        class StreamCorpus:
            def __init__(self, filepath):
                self.filepath = filepath
            def __iter__(self):
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    for doc_line in f:
                        yield doc_line.strip()

        corpus_stream = StreamCorpus(processed_file_path)

        print("\nFitting TF-IDF model...")
        tfidf_mat = self.tfidf_vectorizer.fit_transform(corpus_stream)
        joblib.dump(self.tfidf_vectorizer, self.output_dir / "tfidf_vectorizer.joblib")
        joblib.dump(tfidf_mat, self.output_dir / "tfidf_matrix.joblib")

        print("Fitting BoW model...")
        bow_mat = self.bow_vectorizer.fit_transform(corpus_stream)
        joblib.dump(self.bow_vectorizer, self.output_dir / "bow_vectorizer.joblib")
        joblib.dump(bow_mat, self.output_dir / "bow_matrix.joblib")

        # PHASE 3: Compile BM25
        self.build_bm25_index()

        print("\nINDEXING PIPELINE COMPLETED SUCCESSFULLY")
        return tfidf_mat, bow_mat
    
    def build_bm25_index(self):
        """
        Reads the processed streaming corpus and pre-compiles the BM25 index
        to save memory and initialization time during retrieval.
        """
        print("\n--- Phase 3: Building BM25 Index ---")
        from rank_bm25 import BM25Okapi 
        
        corpus_file = self.output_dir / "processed_corpus.txt"
        output_file = self.output_dir / "bm25_index.joblib"
        
        if not corpus_file.exists():
            print("ERROR: processed_corpus.txt not found. Run Phase 1 first.")
            return

        print("Tokenizing corpus for BM25...")
        tokenized_corpus = []
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=self.total_docs, desc="Tokenizing"):
                tokenized_corpus.append(line.strip().split())
                
        print("Compiling BM25 Okapi Engine...")
        bm25_engine = BM25Okapi(tokenized_corpus)
        
        joblib.dump(bm25_engine, output_file)
        
        size_mb = output_file.stat().st_size / (1024*1024)
        print(f"BM25 index saved to {output_file} ({size_mb:.2f} MB)")