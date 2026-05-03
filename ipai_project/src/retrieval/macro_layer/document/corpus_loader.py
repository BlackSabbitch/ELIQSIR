import json
import time
from typing import Tuple, List

class CorpusLoader:
    """
    Utility class responsible for loading and preprocessing 
    datasets before passing them to the indexers.
    """

    @staticmethod
    def load_jsonl_corpus(file_path: str) -> Tuple[List[str], List[str]]:
        """
        Reads a JSONL file and extracts texts and their corresponding IDs.
        
        Args:
            file_path (str): The absolute or relative path to the .jsonl file.
            
        Returns:
            Tuple[List[str], List[str]]: A tuple containing the list of texts 
                                         and the list of article keys.
        """
        texts = []
        ids = []

        print(f"Loading JSONL corpus from: {file_path}")
        start_time = time.time()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    
                    # Extract the required fields
                    article_key = str(doc.get('article_key', ''))
                    search_corpus = doc.get('search_corpus', '')
                    
                    # Ensure we only add valid, non-empty documents to our index
                    if search_corpus and article_key:
                        ids.append(article_key)
                        texts.append(search_corpus)
                        
            print(f"Successfully loaded {len(texts)} documents in {time.time() - start_time:.2f} seconds.")
            
        except FileNotFoundError:
            print(f"Error: The file was not found at {file_path}. Please check the path.")
        except json.JSONDecodeError:
            print("Error: The file contains invalid JSON data.")
        except Exception as e:
            print(f"An unexpected error occurred while loading the corpus: {e}")

        return texts, ids