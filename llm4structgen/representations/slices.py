import math
import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from invcryrep.invcryrep import InvCryRep
from pymatgen.core.structure import Structure

from .base_representation import BaseRepresentation

class SLICES(BaseRepresentation):
    def __init__(
        self, 
        translate: bool = False,
        rotate: bool = False,
        permute: bool = False,
    ):
        super().__init__()
        self.translate = translate
        self.rotate = rotate
        self.permute = permute
        self.backend = InvCryRep()

    def encode(
        self,
        atomic_symbols,
        positions,
        cell,
    ):
        if self.translate:
            positions = self._translate(positions, cell)
        if self.rotate:
            positions = self._rotate(positions)
        if self.permute:
            atomic_symbols, positions = self._permute(atomic_symbols, positions)

        atoms = Atoms(
            symbols=atomic_symbols,
            positions=positions,
            cell=cell,
            pbc=(True, True, True),
        )

        structure = AseAtomsAdaptor.get_structure(atoms)

        return self.backend.structure2SLICES(structure)
    
    def decode(self, input_str: str) -> Atoms:
        structure, final_energy_per_atom_IAP = self.backend.SLICES2structure(input_str)
        atoms = AseAtomsAdaptor.get_atoms(structure)

        return atoms