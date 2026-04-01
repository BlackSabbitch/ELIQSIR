import time
from tqdm import tqdm 
import pandas as pd

class DrugPipeline:
    def __init__(self, db_manager, engine, storage):
        """
        Orchestrates the extraction from DWH, transformation to graphs, and loading to HDF5.
        """
        self.db = db_manager
        self.engine = engine
        self.storage = storage

    def run(self, batch_size: int = 1000, limit: int = None, start_offset: int = 0):
        """
        Runs the ETL pipeline for drug graphs.
        :param batch_size: Number of molecules to process before saving to disk.
        :param limit: Optional limit for testing (e.g., process only first 5000).
        :param start_offset: Row number to start fetching from DWH (useful for resuming).
        """
        offset = start_offset 
        total_processed = 0
        total_skipped = 0
        
        print(f"Starting Drug Graph Generation Pipeline from offset {offset}...")
        start_time = time.time()
        
        while True:
            # Check if we hit the user-defined limit
            if limit and (offset - start_offset) >= limit: 
                print(f"Reached user-defined limit of {limit} molecules.")
                break

            # 1. FETCH DATA
            query = f"""
                SELECT drug_key, canonical_smiles 
                FROM dim_drug 
                WHERE canonical_smiles IS NOT NULL 
                LIMIT {batch_size} OFFSET {offset}
            """
            df_batch = self.db.fetch_to_dataframe(query)
            
            if df_batch.empty:
                print("Reached the end of the database table.")
                break
                
            batch_results = {}
            
            # 2. TRANSFORM (Build Graphs)
            for _, row in tqdm(df_batch.iterrows(), total=len(df_batch), desc=f"Batch (offset {offset})"):
                try:
                    smiles_str = row['canonical_smiles']
                    graph = self.engine.build(smiles_str)
                    
                    if graph is None:
                        total_skipped += 1
                        continue
                        
                    batch_results[row['drug_key']] = {
                        'graph': graph,
                        'smiles': smiles_str
                    }
                except Exception as e:
                    total_skipped += 1
                    continue
            
            # 3. LOAD (Save to HDF5)
            if batch_results:
                try:
                    self.storage.save_graph_batch(batch_results)
                    total_processed += len(batch_results)
                except Exception as e:
                    print(f"\nCritical Error saving batch to HDF5: {e}")
                    break
            else:
                print(f"\nWarning: Batch at offset {offset} yielded 0 valid graphs!")
            
            offset += batch_size

        # FINAL STATISTICS 
        elapsed = time.time() - start_time
        print("\n" + "="*40)
        print("PIPELINE COMPLETE!")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Successfully saved in this run: {total_processed} graphs")
        print(f"Skipped in this run: {total_skipped}")
        print("="*40)