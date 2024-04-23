import ase, torch, itertools, warnings
import numpy as np

from torch_scatter import scatter


class Distances:

    def __init__(self, CONFIG) -> None:
        self.CONFIG = CONFIG

        # check GPU availability & set device
        self.device = torch_device_select(self.CONFIG.gpu)
        self.dsize = (self.CONFIG.L + 1) * len(self.CONFIG.eta) * len(self.CONFIG.Rs)

    def get_distances(self, positions, cell, offset_count=None, device=None):
        if offset_count is None:
            offset_count = self.CONFIG.offset_count
        if device is None:
            device = self.device
        if isinstance(cell, ase.cell.Cell):
            cell = np.array(cell)

        offsets = get_pbc_offsets(cell, offset_count, device)
        distances, _ = get_distances(positions, offsets, device)
        return distances

    def get_features(
        self, positions, cell, atomic_numbers, offset_count=None, device=None
    ):
        if offset_count is None:
            offset_count = self.CONFIG.offset_count
        if device is None:
            device = self.device
        if isinstance(cell, ase.cell.Cell):
            cell = np.array(cell)

        distances = self.get_distances(positions, cell)
        distance_max = torch.amax(distances, dim=0)
        distance_min = torch.amin(
            distances + torch.eye(distances.shape[0], device=self.device) * 1e15, dim=0
        )

        distance_mean = torch.zeros(
            (distances.shape[0]), device=self.device, dtype=torch.float
        )
        # TODO: sum and divide by (n-1)
        for i in range(0, distances.shape[0]):
            distance_mean[i] = torch.mean(distances[i][distances[i] > 0], dim=0)

        distance_max = scatter(
            distance_max,
            torch.tensor(atomic_numbers, dtype=int, device=self.device),
            dim=0,
            dim_size=100,
            reduce="max",
        ).flatten()
        distance_min = scatter(
            distance_min,
            torch.tensor(atomic_numbers, dtype=int, device=self.device),
            dim=0,
            dim_size=100,
            reduce="min",
        ).flatten()
        distance_mean = scatter(
            distance_mean,
            torch.tensor(atomic_numbers, dtype=int, device=self.device),
            dim=0,
            dim_size=100,
            reduce="mean",
        ).flatten()

        out = torch.cat((distance_max, distance_min, distance_mean), dim=0)

        return out


def get_distances(positions, pbc_offsets, device):
    """
    Get atomic distances

        Parameters:
            positions (numpy.ndarray/torch.Tensor): positions attribute of ase.Atoms
            pbc_offsets (numpy.ndarray/torch.Tensor): periodic boundary condition offsets

        Returns:
            TODO
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


def get_all_distances(positions, pbc_offsets, device):
    """
    Get atomic distances

        Parameters:
            positions (numpy.ndarray/torch.Tensor): positions attribute of ase.Atoms
            pbc_offsets (numpy.ndarray/torch.Tensor): periodic boundary condition offsets

        Returns:
            TODO
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
    # min_indices = min_indices[..., None, None].expand(-1, -1, 1, atom_rij.size(3))
    # atom_rij = torch.gather(atom_rij, dim=2, index=min_indices).squeeze()

    return atom_distance_sqr, atom_rij


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


# Obtain unit cell offsets for distance calculation
class PBC_offsets:
    def __init__(self, cell, device, supercell_max=4):
        # set up pbc offsets for minimum distance in pbc
        self.pbc_offsets = []

        for offset_num in range(0, supercell_max):
            unit_cell = []
            offset_range = np.arange(-offset_num, offset_num + 1)

            for prod in itertools.product(offset_range, offset_range, offset_range):
                unit_cell.append(list(prod))

            unit_cell = torch.tensor(unit_cell, dtype=torch.float, device=device)
            self.pbc_offsets.append(torch.mm(unit_cell, cell.to(device)))

    def get_offset(self, offset_num):
        return self.pbc_offsets[offset_num]


def torch_device_select(gpu):
    # check GPU availability & return device type
    if torch.cuda.is_available() and not gpu:
        warnings.warn("GPU is available but not used.")
        return "cpu"
    elif not torch.cuda.is_available() and gpu:
        warnings.warn("GPU is not available but set to used. Using CPU.")
        return "cpu"
    elif torch.cuda.is_available() and gpu:
        return "cuda"
    else:
        return "cpu"
