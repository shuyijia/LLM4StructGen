import glob
import ase
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import os

import torch

from llm4structgen.distance import get_distances, get_pbc_offsets

original_directory = "./data/mp20-cif/"
new_directory = "./data/mp20-distance-matrix/"

if not os.path.exists(new_directory):
    os.makedirs(new_directory)

for original_file_path in glob.glob(os.path.join(original_directory, "*.csv")):
    filename = os.path.basename(original_file_path)
    new_file_path = os.path.join(new_directory, filename)

    if not os.path.exists(new_file_path):
        df = pd.concat([pd.read_csv(fn) for fn in glob.glob(original_file_path)])
        df = df.to_dict(orient="records")

        new_df = []

        for i, row in enumerate(df):
            structure = Structure.from_str(row["cif"], fmt="cif")
            atom = AseAtomsAdaptor.get_atoms(structure)
            symbols = atom.get_chemical_symbols()
            lengths = structure.lattice.parameters[:3]
            angles = structure.lattice.parameters[3:]

            positions = atom.get_positions()
            cell = atom.get_cell()

            if isinstance(cell, ase.cell.Cell):
                cell = np.array(cell)

            pbc_offsets = get_pbc_offsets(cell, 3, device="cpu")
            distances, _ = get_distances(positions, pbc_offsets, device="cpu")

            # Create the distance matrix string and add the 3 lattice lengths and angles at the top
            distance_matrix_string = \
                " ".join(["{0:.1f}".format(x) for x in lengths]) + "\n" + \
                " ".join([str(round(x)) for x in angles]) + "\n"

            # Add the lower triangular portion of the distance matrix along with the atom symbols for each row of the matrix
            for i in range(len(distances)):
                line = symbols[i]
                for j in range(i):
                    line += " " + str(round(distances[i][j].item(), 2))
                distance_matrix_string += line + "\n"

            new_df.append({
                "distance_matrix": distance_matrix_string
            })

        new_df = pd.DataFrame(new_df)
        new_df.to_csv(new_file_path, index=False)
