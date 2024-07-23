import math
import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from .base_representation import BaseRepresentation
from llm4structgen.datasets.prompts import CARTESIAN_GENERATION_PROMPT_HEADER

class Cartesian(BaseRepresentation):
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
        
        return struct2cartesian(
            atomic_symbols=atomic_symbols,
            positions=positions,
            cell=cell,
            decimals=self.decimals,
        )
    
    def decode(self, input_str: str) -> Atoms:
        return cartesian2struct(input_str)

    @property
    def prompt_header(self):
        return CARTESIAN_GENERATION_PROMPT_HEADER

def struct2cartesian(
    atomic_symbols, 
    positions, 
    cell, 
    decimals=2,
    fractional_coors=True
):
    """
    Given the atomic symbols, positions and cell of a structure,
    return a string representation of the structure (CIF).

    Args:
        fractional_coors (bool): Whether to use fractional coordinates or not.
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
    coors = atoms.get_scaled_positions() if fractional_coors else atoms.get_positions()

    # Create the CIF string
    cif_str = \
        " ".join(["{0:.1f}".format(x) for x in lengths]) + "\n" + \
        " ".join([str(int(x)) for x in angles]) + "\n" + \
        "\n".join([
            str(t) + "\n" + " ".join([
                "{0:.2f}".format(x) for x in c
            ]) for t,c in zip(atomic_symbols, coors)
        ])
    
    return cif_str

def cartesian2struct(cif_str: str):
    lines = cif_str.split("\n")

    # cell
    cell = []
    for l in lines[:2]:
        cell += [float(x) for x in l.split(" ")]
    
    # atomic symbols and positions
    atomic_symbols = []
    positions = []

    for l in lines[2:]:
        if not l:
            continue
        
        if l.isalpha():
            atomic_symbols.append(l)
        else:
            positions.append([float(x) for x in l.split(" ")])

    # construct atoms object
    atoms = Atoms(
        symbols=atomic_symbols,
        scaled_positions=positions,
        cell=cell,
        pbc=[True, True, True]
    )
    
    return atoms