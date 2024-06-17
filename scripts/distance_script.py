import glob
import pandas as pd
from pymatgen.core.structure import Structure
import os

from llm4structgen.distance import struct2distance_matrix

original_directory = "./data/mp20-cif/"
new_directory = "./data/mp20-distance-matrix/"

os.makedirs(new_directory, exist_ok=True)

for original_file_path in glob.glob(os.path.join(original_directory, "*.csv")):
    filename = os.path.basename(original_file_path)
    new_file_path = os.path.join(new_directory, filename)

    if not os.path.exists(new_file_path):
        df = pd.concat([pd.read_csv(fn) for fn in glob.glob(original_file_path)])
        df = df.to_dict(orient="records")

        for i, row in enumerate(df):
            structure = Structure.from_str(row["cif"], fmt="cif")
            distance_matrix_string = struct2distance_matrix(structure, permute=False)

            row["distance_matrix"] = distance_matrix_string

        df = pd.DataFrame(df)
        df.to_csv(new_file_path, index=False)
