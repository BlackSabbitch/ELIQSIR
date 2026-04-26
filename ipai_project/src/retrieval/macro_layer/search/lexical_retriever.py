import os
import re
import joblib
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

class LexicalRetriever:
    """
    Core search engine utilizing TF-IDF, Boolean, and BM25 (Okapi).
    Optimized for fast inference using pre-built indexing artifacts.
    """
    def __init__(self):
        env_data_dir = os.getenv("DATA_DIR")
        if not env_data_dir:
            raise ValueError("DATA_DIR is not set. Make sure dotenv is loaded!")
            
        self.data_dir = Path(env_data_dir)
        self.index_path = self.data_dir / "indexes" / "lexical"
        
        print("Loading ELIQSIR Indexing Artifacts...")
        
        # 1. Load Scikit-Learn Artifacts
        self.tfidf_vec = joblib.load(self.index_path / "tfidf_vectorizer.joblib")
        # Keep TF-IDF as CSR (Compressed Sparse Row) format for fast row-wise cosine similarity
        self.tfidf_mat = joblib.load(self.index_path / "tfidf_matrix.joblib")
        
        self.bow_vec = joblib.load(self.index_path / "bow_vectorizer.joblib")
        # Convert BoW matrix to CSC (Compressed Sparse Column) format.
        # This makes column slicing in search_boolean() exponentially faster.
        self.bow_mat = joblib.load(self.index_path / "bow_matrix.joblib").tocsc()
        
        self.article_keys = joblib.load(self.index_path / "article_keys.joblib")
        
        # Load the pre-compiled BM25 engine 
        self.bm25_engine = joblib.load(self.index_path / "bm25_index.joblib")
        
        # 2. Setup NLP pipeline (Must perfectly match LexicalIndexer)
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.lemmatize_cached = lru_cache(maxsize=10000)(self.lemmatizer.lemmatize)
        
        # Replacing hyphens with spaces safely, matching indexer
        self.clean_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        
        self.SCIENTIFIC_STOP_WORDS = {
            'study', 'result', 'using', 'significantly', 'effect', 'activity', 
            'showed', 'data', 'research', 'analysis', 'method', 'concentration', 
            'dose', 'values', 'compound', 'derivatives', 'inhibitor', 'active', 
            'tested', 'based', 'potential', 'evaluated', 'clinical', 'treatment'
        }
        self.stop_words = set(stopwords.words('english')).union(self.SCIENTIFIC_STOP_WORDS)
        
        print("Commencing countdown, engines ON!")

    def _preprocess_query(self, query: str) -> str:
        """Cleans and lemmatizes the query."""
        if not query: return ""
        query = self.clean_pattern.sub(' ', query.lower())
        tokens = [self.lemmatize_cached(t) for t in query.split() 
                  if t not in self.stop_words and len(t) > 1]
        return " ".join(tokens)

    def search_bm25(self, query: str, top_k=10):
        """Standard BM25Okapi retrieval using precompiled rank_bm25 engine."""
        processed_query = self._preprocess_query(query)
        tokens = processed_query.split()
        if not tokens: return []

        # Use the pre-compiled engine to get scores for all documents
        scores = self.bm25_engine.get_scores(tokens)
        
        return self._rank_results(scores, top_k)

    def search_vsm(self, query: str, top_k=10):
        """Vector Space Model using Cosine Similarity on TF-IDF."""
        processed_query = self._preprocess_query(query)
        if not processed_query: return []
        
        query_vec = self.tfidf_vec.transform([processed_query])
        similarities = cosine_similarity(query_vec, self.tfidf_mat).flatten()
        return self._rank_results(similarities, top_k)

    def search_boolean(self, query: str, operator="AND", min_similarity=0.0):
        """
        Boolean retrieval with Overlap Similarity scoring.
        If operator="AND", strictly requires 100% term overlap.
        If operator="OR", requires at least min_similarity (0.0 to 1.0) overlap.
        """
        processed_query = self._preprocess_query(query)
        tokens = processed_query.split()
        if not tokens: return []

        vocab = self.bow_vec.vocabulary_
        indices = [vocab[t] for t in tokens if t in vocab]
        if not indices: return []

        # Slicing is now extremely fast due to CSC format conversion in __init__
        term_presence_matrix = self.bow_mat[:, indices] > 0
        match_counts = term_presence_matrix.sum(axis=1).A1 
 
        similarity_scores = match_counts / len(indices)

        if operator == "AND":
            mask = similarity_scores == 1.0
        else: 
            mask = similarity_scores > min_similarity
            
        res_indices = np.where(mask)[0]
        valid_scores = similarity_scores[res_indices]
        
        sorted_order = np.argsort(valid_scores)[::-1]
        best_indices = res_indices[sorted_order]
        best_scores = valid_scores[sorted_order]

        return [
            {"article_key": self.article_keys[i], "score": round(float(score) * 100, 2)}
            for i, score in zip(best_indices, best_scores)
        ]

    def _rank_results(self, scores, top_k):
        """Sorts and formats the top-k results."""
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"article_key": self.article_keys[i], "score": round(float(scores[i]), 4)}
            for i in top_indices if scores[i] > 0
        ]