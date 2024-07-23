import ase
import numpy as np
from ase import Atoms

from .base_representation import BaseRepresentation
from llm4structgen.datasets.prompts import DISTANCE_MATRIX_GENERATION_PROMPT_HEADER

class DistanceMatrix(BaseRepresentation):
    def __init__(
        self, 
        translate: bool = False,
        rotate: bool = False,
        permute: bool = False,
        decimals: int = 2
    ):
        super().__init__()
        self.translate = translate
        self.rotate = rotate
        self.permute = permute
        self.decimals = decimals

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
        
        return struct2distance_matrix(
            atomic_symbols=atomic_symbols,
            positions=positions,
            cell=cell,
            decimals=self.decimals,
        )
    
    def decode(self, input_str: str) -> Atoms:
        return distance_matrix2struct(input_str)
    
    @property
    def prompt_header(self):
        return DISTANCE_MATRIX_GENERATION_PROMPT_HEADER
    
def struct2distance_matrix(
    atomic_symbols,
    positions,
    cell,
    decimals=2,
):
    """
    Given the atomic symbols, positions and cell of a structure,
    return a lower triangular distance matrix representation.
    """

    atoms = Atoms(
        symbols=atomic_symbols,
        positions=positions,
        cell=cell,
        pbc=(True, True, True),
    )

    lattice_params = atoms.cell.cellpar()
    lengths = lattice_params[:3]
    angles = lattice_params[3:]

    out_str = " ".join(["{0:.1f}".format(x) for x in lengths]) + "\n" + \
        " ".join([str(int(x)) for x in angles]) + "\n" + \
        atomic_symbols[0] + "\n"

    pbc_dist_mat = atoms.get_all_distances(mic=True)

    for i, (sym, dists) in enumerate(zip(atomic_symbols[1:], pbc_dist_mat[1:])):
        out_str += sym + "\n"
        dist_str = ""
        for j in range(i+1):
            dist_str += f"{dists[j]:.{decimals}f} "
        out_str += dist_str.strip() + "\n"
    
    return out_str

def distance_matrix2struct(input_str: str) -> Atoms:
    """
    Given a lower triangular distance matrix representation, return the structure.
    """
    pass