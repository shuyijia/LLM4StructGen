import argparse
import ase.io
from glob import glob
from tqdm import tqdm
from pathlib import Path
from collections import Counter

import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

def num_elements(atoms_list):
    counter = Counter()

    for atoms in atoms_list:
        counter.update(atoms.get_chemical_symbols())
    
    return counter

def num_atoms(atoms_list):
    counter = Counter()

    for atoms in atoms_list:
        counter.update([len(atoms)])
    
    return counter

# element distribution periodic table plot functions
symbol = [['H', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'He'],
         ['Li', 'Be', '', '', '', '', '', '', '', '', '', '', 'B', 'C', 'N', 'O', 'F', 'Ne'],
         ['Na', 'Mg', '', '', '', '', '', '', '', '', '', '', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
         ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
         ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe' ],
         ['Cs', 'Ba', '', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn' ],
         ['Fr', 'Ra', '', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'],
         ['', '', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', ''],
         ['', '', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', '' ]]

def get_custom_colorscale(code):
    "returns a custom color scale"
    colorscale = getattr(px.colors.sequential, code)[:10]
    assert len(colorscale) == 10

    colorscale = [[i/10, colorscale[i-1]] for i in range(1, 11)]
    colorscale = [[0, "white"]] + colorscale
    return colorscale

def get_color_values(d):
    "dictionary containing the frequency of elements according to the order of the periodic table"
    keys = d.keys()
    vals = np.array(list(d.values()))

    log_vals = np.log(vals + 1)
    new_d = dict(zip(keys, log_vals))

    color_values = []
    for row in symbol:
        row_values = []
        for element in row:
            if element in new_d:
                row_values.append(new_d[element])
            else:
                row_values.append(0)
        color_values.append(row_values)
    
    return color_values

def plot_and_save_element(title, data):
    color = get_color_values(data)
    color = np.where(np.array(symbol) == '', -1, color)
    colorscale = get_custom_colorscale("tempo")

    fig = px.imshow(color, aspect="auto", color_continuous_scale=colorscale, title=title)
    fig.update_traces(text=symbol, texttemplate="%{text}", textfont_size=12)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    # Customize colorbar ticks and labels with actual values, excluding -1 values
    valid_colors = color[color != -1]
    ticks_values = np.linspace(0, np.max(valid_colors), num=5)
    actual_ticks = np.round(np.exp(ticks_values) - 1, decimals=2).astype(int)

    fig.update_coloraxes(
        cmin=0,
        colorbar=dict(
            tickvals=ticks_values,
            ticktext=actual_ticks,
            title="Count",
        )
    )

    fig.write_image(f"n_elements_{title}.png")

# num of atoms histogram plot functions
def plot_and_save_atoms(title, data):
    keys = list(data.keys())
    for i in range(1, max(keys)):
        if i not in keys:
            keys.append(i)
    sorted_keys = sorted(keys)
    values = [data.get(key, 0) for key in sorted_keys]

    print(sorted_keys)
    print(values)

    # Create custom x positions with increased spacing
    spacing_factor = 1.1  # Increase this value to increase spacing
    x_positions = np.arange(len(sorted_keys)) * spacing_factor

    # Plot the bar chart
    plt.bar(x_positions, values, align='center', width=1.2, color='skyblue', edgecolor='black')

    # Set x-ticks to match the custom positions and rotate labels
    plt.xticks(x_positions, sorted_keys, rotation=90, ha='right')
    plt.xlabel('# atoms')
    plt.ylabel('# structures')
    plt.title(title)
    
    plt.tight_layout()  # Adjust layout to fit rotated x-tick labels
    plt.savefig(f"n_atoms_{title}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribution of elements and atoms")
    parser.add_argument("--cif_dir", type=str, help="Path to the directory containing CIF files")
    parser.add_argument("--title", type=str, default=None, help="Title of the plot")

    args = parser.parse_args()

    cif_dir = Path(args.cif_dir)
    cifs = glob(str(cif_dir / "*.cif"))
    assert len(cifs) > 0, "No CIF files found in the directory"

    # title
    title = args.title if args.title else cif_dir.parent.name

    atoms_list = [ase.io.read(cif) for cif in tqdm(cifs)]

    elements = num_elements(atoms_list)
    atoms = num_atoms(atoms_list)

    print(f"Elements:\n {elements}")
    print(f"Number of atoms:\n {atoms}")

    # plot and save element distribution
    plot_and_save_element(title, elements)

    # plot and save number of atoms histogram
    plot_and_save_atoms(title, atoms)