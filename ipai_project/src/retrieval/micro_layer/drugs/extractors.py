import numpy as np
from rdkit import Chem
from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    """
    Abstract base class for all node (atom) feature extractors.
    """
    @abstractmethod
    def extract(self, atom: Chem.Atom) -> list:
        pass

class AdvancedAtomExtractor(FeatureExtractor):
    """
    Extracts advanced physicochemical and quantum properties of an atom.
    Expects Gasteiger charges to be computed on the molecule beforehand.
    """
    def extract(self, atom: Chem.Atom) -> list:
        # 1. Basic structural properties
        atomic_num = atom.GetAtomicNum()
        degree = atom.GetDegree()
        mass = atom.GetMass()
        is_in_ring = float(atom.IsInRing())
        is_aromatic = float(atom.GetIsAromatic())
        
        # 2. Quantum & Physicochemical properties
        formal_charge = float(atom.GetFormalCharge())
        num_radical_electrons = float(atom.GetNumRadicalElectrons())
        
        # Hybridization mapping (cast to float for the tensor)
        hybridization = float(atom.GetHybridization())
        
        # 3. Gasteiger partial charge
        # Wrap in try-except because calculation might fail for exotic ions
        try:
            gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
            if np.isnan(gasteiger_charge) or np.isinf(gasteiger_charge):
                gasteiger_charge = 0.0
        except KeyError:
            gasteiger_charge = 0.0
            
        return [
            atomic_num, degree, mass, is_in_ring, is_aromatic,
            formal_charge, num_radical_electrons, hybridization, gasteiger_charge
        ]