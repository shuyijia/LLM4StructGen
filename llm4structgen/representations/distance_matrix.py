import numpy as np
from ase import Atoms
import torch

from llm4structgen.reconstruction.reconstruction import Reconstruction, dotdict
from torchtune import utils

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
    CONFIG = {}
    CONFIG["descriptor"] = "distance"
    CONFIG["all_neighbors"] = True
    CONFIG["perturb"] = False
    CONFIG["load_pos"] = False
    CONFIG["cutoff"] = 10.0
    CONFIG["offset_count"] = 1

    cfg = dotdict(CONFIG)

    lines = [x for x in input_str.split("\n") if len(x) > 0]
    lengths = [float(x) for x in lines[0].split(" ")]
    angles = [float(x) for x in lines[1].split(" ")]
    species = [lines[2]] + [x for x in lines[3::2]]
    lower_matrix = [[0]] + [[float(y) for y in x.split(" ")] for x in lines[4::2]]

    n = len(lower_matrix)
    distance_matrix = torch.zeros((n, n), dtype=torch.float, device=utils.get_device())

    for i in range(n):
        for j in range(i + 1):
            if i != j:
                distance_matrix[i][j] = lower_matrix[i][j]

    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i][j] = distance_matrix[j][i]

    cell_arr = lengths + angles

    atom = Atoms(symbols=species, cell=cell_arr)

    cell = atom.get_cell()
    atomic_numbers = atom.get_atomic_numbers()

    data = {}
    data["atomic_numbers"] = atomic_numbers
    data["cell"] = torch.tensor(np.array(cell), dtype=torch.float, device=utils.get_device())
    features = distance_matrix
    data["representation"] = torch.unsqueeze(features, 0)

    constructor = Reconstruction(cfg)

    best_positions, _ = constructor.basin_hopping(
        data,
        total_trials=5,
        max_hops=5,
        lr=0.02,
        displacement_factor=2,
        max_loss=0.00001,
        max_iter=500,
        verbose=True,
    )

    optimized_structure = Atoms(
        numbers=data["atomic_numbers"],
        positions=best_positions.detach().cpu().numpy(),
        cell=data["cell"].cpu().numpy(),
        pbc=(True, True, True),
    )

    return optimized_structure
