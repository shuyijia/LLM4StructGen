import tensorflow as tf
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from invcryrep.invcryrep import InvCryRep
from pymatgen.core import Structure

from .base_representation import BaseRepresentation
from llm4structgen.datasets.prompts import SLICES_GENERATION_PROMPT_HEADER

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

        # prevent TensorFlow from allocating all available GPU memory at once
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)

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

        # by default, pymatgen.core.Structure expects fractional coors
        # positions @ np.linalg.inv(cell)
        structure = Structure(
            cell, 
            atomic_symbols, 
            positions, 
            coords_are_cartesian=True
        )

        slices_str = self.backend.structure2SLICES(structure)
        return slices_str
    
    def decode(self, input_str: str) -> Atoms:
        structure, final_energy_per_atom_IAP = self.backend.SLICES2structure(input_str)
        atoms = AseAtomsAdaptor.get_atoms(structure)

        return atoms
    
    @property
    def prompt_header(self):
        return SLICES_GENERATION_PROMPT_HEADER