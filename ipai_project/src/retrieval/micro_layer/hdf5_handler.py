import os
import h5py
import numpy as np

class HDF5Handler:
    def __init__(self, filename: str = "advanced_quantum_dataset.h5"):
        # Calculate absolute path to the ELIQSIR root directory
        # Current file is in ELIQSIR/ipai_project/src/retrieval/micro_layer/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        eliqsir_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
        self.file_path = os.path.join(eliqsir_root, 'data', 'datasets', filename)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        # Initialize string data type for saving SMILES in HDF5
        self.string_dt = h5py.string_dtype(encoding='utf-8')
        
        print(f"HDF5 Storage initialized. Target file: {self.file_path}")

    def save_graph_batch(self, batch_data: dict):
        """
        Appends a batch of graph data to the HDF5 file.
        Expects batch_data format: 
        {drug_key: {'graph': {'x': np.array, 'edge_index': np.array, 'u': np.array, ...}, 'smiles': str}}
        """
        # Open file in append mode ('a') so we don't overwrite previous batches
        with h5py.File(self.file_path, "a") as f:
            for drug_key, data in batch_data.items():
                group_name = str(drug_key)
                
                # Prevent duplication errors if the pipeline is restarted
                if group_name in f:
                    continue
                    
                graph = data['graph']
                smiles_str = data['smiles']
                
                group = f.create_group(group_name)
                
                # Save matrices with gzip compression (level 4 for optimal speed/size ratio)
                group.create_dataset("x", data=graph['x'], compression="gzip", compression_opts=4)
                group.create_dataset("edge_index", data=graph['edge_index'], compression="gzip", compression_opts=4)
                group.create_dataset("edge_attr", data=graph['edge_attr'], compression="gzip", compression_opts=4)
                group.create_dataset("pos", data=graph['pos'], compression="gzip", compression_opts=4)
                
                # Save global molecular descriptors (LogP, TPSA, exact_mw, etc.)
                group.create_dataset("u", data=graph['u'], compression="gzip", compression_opts=4)
                
                # Save the raw SMILES string for future debugging and visualization
                group.create_dataset("smiles", data=smiles_str, dtype=self.string_dt)