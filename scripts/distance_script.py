import glob
import ase
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import os

import torch

from llm4structgen.distance import get_distances, get_pbc_offsets, struct2distance_matrix

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

        for i, row in enumerate(df):
            structure = Structure.from_str(row["cif"], fmt="cif")
            distance_matrix_string = struct2distance_matrix(structure, permute=False)
            distance_matrix_string_permuted = struct2distance_matrix(structure, permute=True)

            # row["distance_matrix"] = distance_matrix_string
            print("Normal: ", distance_matrix_string)
            print("Permuted: ", distance_matrix_string_permuted)


        # df = pd.DataFrame(df)
        # df.to_csv(new_file_path, index=False)
