class HybridRetriever:
    """
    Independent Hybrid Retrieval Engine.
    Merges Lexical (BM25) and Semantic (Dense) results using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, lexical_retriever, neural_indexer, rrf_k=60):
        self.lexical = lexical_retriever
        self.neural = neural_indexer
        self.rrf_k = rrf_k 

    def search(self, query, top_k=10):
        # Fetch a larger candidate pool to ensure robust intersection
        candidate_k = max(60, top_k * 2)
        
        bm25_res = self.lexical.search_bm25(query, top_k=candidate_k)
        neural_res = self.neural.search(query, top_k=candidate_k)
        
        scores = {}
        
        # Calculate RRF scores
        for rank, res in enumerate(bm25_res, 1):
            key = int(res['article_key'])
            scores[key] = scores.get(key, 0) + (1.0 / (self.rrf_k + rank))
            
        for rank, res in enumerate(neural_res, 1):
            key = int(res['article_key'])
            scores[key] = scores.get(key, 0) + (1.0 / (self.rrf_k + rank))
            
        # Sort by merged score descending
        sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [{'article_key': k, 'score': s} for k, s in sorted_keys[:top_k]]