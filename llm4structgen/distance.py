import torch, itertools
import numpy as np
import ase
import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure


def struct2distance_matrix(structure, permute=False):
    atom = AseAtomsAdaptor.get_atoms(structure)
    symbols = atom.get_chemical_symbols()
    lengths = structure.lattice.parameters[:3]
    angles = structure.lattice.parameters[3:]

    positions = atom.get_positions()
    cell = atom.get_cell()
    chemical_symbols = atom.get_chemical_symbols()

    if permute:
        idx = np.random.permutation(len(chemical_symbols))
        chemical_symbols = [chemical_symbols[i] for i in idx]
        positions = positions[idx]

    if isinstance(cell, ase.cell.Cell):
        cell = np.array(cell)

    pbc_offsets = get_pbc_offsets(cell, 3, device="cpu")
    distances, _ = get_distances(positions, pbc_offsets, device="cpu")

    # Create the distance matrix string and add the 3 lattice lengths and angles at the top
    distance_matrix_string = (
        " ".join(["{0:.1f}".format(x) for x in lengths])
        + "\n"
        + " ".join([str(round(x)) for x in angles])
        + "\n"
    )

    # Add the lower triangular portion of the distance matrix along with the atom symbols for each row of the matrix
    for i in range(len(distances)):
        line = symbols[i]
        for j in range(i):
            line += " " + str(round(distances[i][j].item(), 2))
        distance_matrix_string += line + "\n"

    return distance_matrix_string


def get_distances(positions, pbc_offsets, device):
    """
    Get atomic distances

        Parameters:
            positions (numpy.ndarray/torch.Tensor): positions attribute of ase.Atoms
            pbc_offsets (numpy.ndarray/torch.Tensor): periodic boundary condition offsets

        Returns:
            M x M matrix of distances
    """

    if isinstance(positions, np.ndarray):
        positions = torch.tensor(positions, device=device, dtype=torch.float)

    n_atoms = len(positions)
    n_cells = len(pbc_offsets)

    pos1 = positions.view(-1, 1, 1, 3).expand(-1, n_atoms, n_cells, 3)
    pos2 = positions.view(1, -1, 1, 3).expand(n_atoms, -1, n_cells, 3)
    pbc_offsets = pbc_offsets.view(-1, n_cells, 3).expand(pos2.shape[0], n_cells, 3)
    pos2 = pos2 + pbc_offsets

    # calculate the distance between target atom and the periodic images of the other atom
    atom_distance_sqr = torch.linalg.norm(pos1 - pos2, dim=-1)
    # get the minimum distance
    atom_distance_sqr_min, min_indices = torch.min(atom_distance_sqr, dim=-1)

    atom_rij = pos1 - pos2
    min_indices = min_indices[..., None, None].expand(-1, -1, 1, atom_rij.size(3))
    atom_rij = torch.gather(atom_rij, dim=2, index=min_indices).squeeze()

    return atom_distance_sqr_min, atom_rij

def get_pbc_offsets(cell, offset_num, device):
    """
    Get periodic boundary condition (PBC) offsets

        Parameters:
            cell (np.ndarray/torch.Tensor): unit cell vectors of ase.cell.Cell
            offset_num:

        Returns:
            TODO
    """
    if isinstance(cell, np.ndarray):
        cell = torch.tensor(np.array(cell), device=device, dtype=torch.float)

    unit_cell = []
    offset_range = np.arange(-offset_num, offset_num + 1)

    for prod in itertools.product(offset_range, offset_range, offset_range):
        unit_cell.append(list(prod))

    unit_cell = torch.tensor(unit_cell, dtype=torch.float, device=device)

    return torch.mm(unit_cell, cell.to(device))
