from abc import ABC, abstractmethod

import torch
from ase import Atoms
import numpy as np
from scipy.spatial.transform import Rotation

class BaseRepresentation(ABC):
    @abstractmethod
    def encode(self, atomic_symbols, positions, cell):
        pass

    @abstractmethod
    def decode(self, input_dict: dict):
        pass

    def _translate(self, positions, cell):
        """
        translate the positions of the atoms by a random vector
        """
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)
        if not isinstance(cell, np.ndarray):
            cell = np.array(cell)
        
        positions_t = torch.tensor(positions, dtype=torch.float)
        cell_t = torch.tensor(cell, dtype=torch.float)

        translation_v = cell_t @ torch.rand(3,1)
        positions_t += translation_v.T

        return positions_t.cpu().numpy().tolist()
    
    def _rotate(self, positions):
        """
        rotate the positions of the atoms by a random rotation matrix
        """

        # convert to torch tensors
        positions_t = torch.tensor(positions, dtype=torch.float)

        # rotation matrix
        R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float)
        positions_t @= R 

        return positions_t.cpu().numpy().tolist()
    
    def _permute(self, atomic_symbols, positions):
        """
        permute the atomic symbols and positions of the atoms
        """
        if len(atomic_symbols) == 1:
            return atomic_symbols, positions

        if not isinstance(atomic_symbols, np.ndarray):
            atomic_symbols = np.array(atomic_symbols)
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)
        
        perm = torch.randperm(len(atomic_symbols))
        atomic_symbols = atomic_symbols[perm]
        positions = positions[perm]

        return atomic_symbols.tolist(), positions.tolist()
        

