import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

class NeuralIndexer:
    """
    A dense retrieval indexer using Sentence Transformers and FAISS.
    Converts text documents into dense vector embeddings and provides 
    fast semantic search capabilities, with disk persistence.
    """
    
    def __init__(self, model_name: str = 'allenai-specter'):
        print(f"Loading transformer model: {model_name}...")
        
        # Initialize the model. It will automatically utilize the GPU if available.
        self.model = SentenceTransformer(model_name)
        
        # Retrieve the output dimension of the embeddings (e.g., 768 for SPECTER)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded successfully! Embedding dimension: {self.dimension}")
        
        # Initialize FAISS index. 
        # IndexFlatIP (Inner Product) with normalized vectors calculates Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # List to map FAISS internal numeric IDs to our original article keys
        self.doc_ids = []

    def build_index(self, texts: list, ids: list, batch_size: int = 32):
        """
        Encodes a list of texts into dense vectors and adds them to the FAISS index.
        Processes data in batches to prevent GPU Out-Of-Memory (OOM) errors.
        
        Args:
            texts: List of strings (document contents) to encode.
            ids: List of document identifiers corresponding to the texts.
            batch_size: Number of documents to process simultaneously.
        """
        if len(texts) != len(ids):
            raise ValueError("The number of texts and ids must be exactly the same.")
            
        print(f"Encoding {len(texts)} documents in batches of {batch_size}...")
        
        # Encode texts. 'normalize_embeddings=True' is crucial for Cosine Similarity.
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        
        print("Adding generated embeddings to the FAISS index...")
        
        # Convert embeddings to float32 (required by FAISS)
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Double-check normalization (FAISS requirement for accurate Inner Product)
        faiss.normalize_L2(embeddings_np) 
        
        # Add vectors to the index and store their corresponding IDs
        self.index.add(embeddings_np)
        self.doc_ids.extend(ids)
        
        print(f"Index built successfully! Total documents in index: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 10) -> list:
        """
        Performs a semantic search to find the most relevant documents for a given query.
        
        Args:
            query: The search string.
            top_k: The number of top results to return.
            
        Returns:
            A list of dictionaries containing the 'article_key' and 'score'.
        """
        if self.index.ntotal == 0:
            print("Warning: The index is empty. Please build the index first.")
            return []

        # Encode the search query into a vector and normalize it
        query_vector = self.model.encode([query], normalize_embeddings=True)
        query_np = np.array(query_vector).astype('float32')
        faiss.normalize_L2(query_np)
        
        # Search the FAISS index
        scores, indices = self.index.search(query_np, top_k)
        
        results = []
        
        # Iterate through the top_k results
        for i, internal_idx in enumerate(indices[0]):
            # FAISS returns -1 if it cannot find enough results
            if internal_idx != -1: 
                results.append({
                    "article_key": str(self.doc_ids[internal_idx]),
                    "score": round(float(scores[0][i]), 4)
                })
                
        return results

    def save(self, save_dir: str):
        """
        Saves the FAISS index and the document IDs to the hard drive.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Save FAISS index
        index_path = os.path.join(save_dir, "faiss.index")
        faiss.write_index(self.index, index_path)
        
        # 2. Save document IDs mapping
        ids_path = os.path.join(save_dir, "doc_ids.json")
        with open(ids_path, 'w', encoding='utf-8') as f:
            json.dump(self.doc_ids, f)
            
        print(f"Index successfully saved to {save_dir}")

    def load(self, save_dir: str):
        """
        Loads a pre-computed FAISS index and document IDs from the hard drive.
        """
        index_path = os.path.join(save_dir, "faiss.index")
        ids_path = os.path.join(save_dir, "doc_ids.json")
        
        if not os.path.exists(index_path) or not os.path.exists(ids_path):
            raise FileNotFoundError(f"Index files not found in {save_dir}. Build the index first!")
            
        # 1. Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # 2. Load document IDs mapping
        with open(ids_path, 'r', encoding='utf-8') as f:
            self.doc_ids = json.load(f)
            
        print(f"Index loaded successfully! Total documents: {self.index.ntotal}")