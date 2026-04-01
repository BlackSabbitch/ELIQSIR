import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from typing import Optional

class QuantumGraphEngine:
    """
    Constructs high-fidelity molecular representations for Quantum Graph Neural Networks (QGNNs).
    Transforms 1D/2D topological data (SMILES) into 3D geometric graphs with quantum-chemical descriptors.
    """
    def __init__(self):
        # Initialized without external extractors to strictly enforce a fixed 
        # 10-feature vector architecture, preventing index shifting.
        print("QuantumGraphEngine initialized: Strict 10-feature mode active.")

    def build(self, smiles: str) -> Optional[dict]:
        """
        Builds a multi-tensor molecular graph. 
        Returns None if 3D embedding or charge calculation fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 1. Add explicit hydrogens for accurate 3D volume and electronics
        mol = Chem.AddHs(mol)
        
        # 2. Compute 3D coordinates (ETKDG v3)
        res = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=50)
        if res != 0:
            return None 
            
        # Optional: Force Field Optimization (UFF)
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass 

        # 3. Compute Gasteiger-Marsili partial charges
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            return None

        # 4. Extract global graph-level features (u) - [1, 4] vector
        try:
            logp = Descriptors.MolLogP(mol)                         # Lipophilicity 
            tpsa = rdMolDescriptors.CalcTPSA(mol)                   # Polar Surface Area
            num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol) # Flexibility
            exact_mw = Descriptors.ExactMolWt(mol)                  # Molecular weight
            
            u = np.array([[logp, tpsa, num_rotatable_bonds, exact_mw]], dtype=np.float32)
        except Exception:
            return None

        # 5. Extract node features (x) and 3D positions (pos)
        node_features = []
        positions = []
        conformer = mol.GetConformer() 
        
        for i, atom in enumerate(mol.GetAtoms()):
            # Safely extract Gasteiger charge
            try:
                gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
            except KeyError:
                gasteiger_charge = 0.0

            # STRICT 10-FEATURE VECTOR: Hardcoded order
            atom_feature_vector = [
                float(atom.GetAtomicNum()),      # [0] Atomic number
                float(atom.GetDegree()),         # [1] Number of bonded neighbors
                float(atom.GetMass()),           # [2] Atomic mass
                float(atom.GetTotalValence()),   # [3] Total valence electrons
                float(atom.GetFormalCharge()),   # [4] Formal charge
                float(atom.GetIsAromatic()),     # [5] Aromaticity (1.0 or 0.0)
                gasteiger_charge,                # [6] Gasteiger partial charge
                float(atom.GetChiralTag()),      # [7] Chirality
                float(atom.GetHybridization()),  # [8] Hybridization state
                float(atom.GetTotalNumHs())      # [9] Attached Hydrogens
            ]
            node_features.append(atom_feature_vector)
            
            # 3D Cartesian coordinates in Angstroms
            pos = conformer.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])

        x = np.array(node_features, dtype=np.float32)
        pos_matrix = np.array(positions, dtype=np.float32)

        # 6. Construct edge_index (topology) and edge_attr (bond physics)
        src_nodes = []
        dst_nodes = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            bond_feat = [
                bond.GetBondTypeAsDouble(),
                float(bond.GetIsConjugated()),
                float(bond.IsInRing()),
                float(bond.GetStereo())
            ]
            
            # Undirected graph: add edges for both directions
            src_nodes.extend([i, j])
            dst_nodes.extend([j, i])
            edge_features.extend([bond_feat, bond_feat])

        edge_index = np.array([src_nodes, dst_nodes], dtype=np.int64)
        edge_attr = np.array(edge_features, dtype=np.float32)

        return {
            "x": x,                   # [N, 10]
            "edge_index": edge_index, # [2, E*2]
            "edge_attr": edge_attr,   # [E*2, 4]
            "pos": pos_matrix,        # [N, 3]
            "u": u                    # [1, 4]
        }