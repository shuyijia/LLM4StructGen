import math
import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from .base_representation import BaseRepresentation
from llm4structgen.datasets.prompts import Z_MATRIX_GENERATION_PROMPT_HEADER

class ZMatrix(BaseRepresentation):
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

    def encode(self, atomic_symbols, positions, cell):
        if self.translate:
            positions = self._translate(positions, cell)
        if self.rotate:
            positions = self._rotate(positions)
        if self.permute:
            atomic_symbols, positions = self._permute(atomic_symbols, positions)

        z_str = struct2zmatrix(
            atomic_symbols=atomic_symbols,
            positions=positions,
            cell=cell,
            decimals=self.decimals,
        )

        return format_zmatrix_str(z_str)
    
    def decode(self, input_str: str) -> Atoms:
        return zmatrix2struct(input_str, return_ase=True)
    
    @property
    def prompt_header(self):
        return Z_MATRIX_GENERATION_PROMPT_HEADER

def struct2zmatrix(atomic_symbols, positions, cell, decimals=2, string=True):    
    zmatrix = []

    # copy atomic symbols
    _atomic_symbols = atomic_symbols.copy()
    
    #Insert dummy atoms
    temp_atoms_1 = Atoms(
        symbols=atomic_symbols, 
        positions=positions,
        cell=cell,
        pbc=(True, True, True)
    )
    _atomic_symbols.insert(0, "Fm")
    _atomic_symbols.insert(0, "Fm")
    _atomic_symbols.insert(0, "Fm")
    positions = temp_atoms_1.get_scaled_positions()
    positions = np.insert(positions, 0, np.array([0.0,0.3,0.0]), axis=0)
    positions = np.insert(positions, 0, np.array([0.3,0.0,0.0]), axis=0)
    positions = np.insert(positions, 0, np.array([0.0,0.0,0.0]), axis=0)
    
    temp_atoms_2 = Atoms(
        symbols=_atomic_symbols, 
        scaled_positions=positions,
        cell=cell,
        pbc=(True, True, True)
    )

    positions = temp_atoms_2.get_positions()

    #convert cell into format of (a, b, c, alpha, beta, gamma)
    lattice_params = temp_atoms_2.cell.cellpar()
    for item in lattice_params:
        zmatrix.append(round(item, decimals))

    #compute internal coordinates with ASE
    for i in range(0, len(temp_atoms_2)):
        zmatrix.append(_atomic_symbols[i])
        if i > 0:
            zmatrix.append(round(temp_atoms_2.get_distance(i-1, i, mic=False), decimals))
        if i > 1:
            zmatrix.append(round(temp_atoms_2.get_angle(i-2, i-1, i, mic=False), decimals))
        if i > 2:
            try:
                zmatrix.append(round(temp_atoms_2.get_dihedral(i-3, i-2, i-1, i, mic=False), decimals))
            except:
                # add a bit of perturbation to avoid divide by zero error
                perturbed_positions = temp_atoms_2.get_positions()
                perturbed_positions += np.random.normal(0, 0.001, perturbed_positions.shape)
                temp_atoms_3 = Atoms(
                    symbols=temp_atoms_2.get_chemical_symbols(),
                    positions=perturbed_positions,
                    cell=temp_atoms_2.get_cell(),
                )
                zmatrix.append(round(temp_atoms_3.get_dihedral(i-3, i-2, i-1, i, mic=False), decimals))
            
    if string == True:
        zmatrix = [str(element) for element in zmatrix]
        zmatrix = " ".join(zmatrix) 
           
    return zmatrix

def zmatrix2struct(zmatrix, return_ase=False):
    zmatrix = [float(e) if e.replace('.', '', 1).isdigit() else str(e) for e in zmatrix.split()]

    # find length of structure
    len_structure_map = {7: 1, 9: 2, 12: 3}
    len_structures = len_structure_map.get(len(zmatrix), int((len(zmatrix) - 12) / 4 + 3))

    # first 6 entries encodes lattice parameters
    s_cell = zmatrix[0:6]
    
    # map zmatrix to separate lists of distances, angles, dihedrals    
    s_atomic_symbols, s_distances, s_angles, s_dihedrals = [], [], [], []
    for i in range(0, len_structures):
        if i == 0:
            s_atomic_symbols.append(zmatrix[6])
            s_distances.append([None])
            s_angles.append([None])
            s_dihedrals.append([None])
        if i == 1:
            s_atomic_symbols.append(zmatrix[7])
            s_distances.append(zmatrix[2])
            s_angles.append([None])
            s_dihedrals.append([None])            
        if i == 2:
            s_atomic_symbols.append(zmatrix[8])
            s_distances.append(zmatrix[9])
            s_angles.append(zmatrix[10])
            s_dihedrals.append([None]) 
        if i >= 3:
            s_atomic_symbols.append(zmatrix[(i-3)*4+12])
            s_distances.append(zmatrix[(i-3)*4+13])
            s_angles.append(zmatrix[(i-3)*4+14])
            s_dihedrals.append(zmatrix[(i-3)*4+15])       
    
    # first three positions for dummy atoms are hardcoded in
    temp = Atoms(
        symbols=["Fm","Fm","Fm"], scaled_positions=[[0,0,0],[0.3,0,0],[0.0,0.3,0]],
        cell=s_cell,
        pbc=(True, True, True)
    )
    temp_positions = temp.get_positions()

    s_coords = []    
    for i in range(0, len_structures):        
        if i == 0:
            s_coords.append(np.array(temp_positions[0]))
        if i == 1:
            s_coords.append(np.array(temp_positions[1]))
        if i == 2: 
            s_coords.append(np.array(temp_positions[2]))                                          
        if i > 2:          
            avec = np.array(s_coords[i-1])
            bvec = np.array(s_coords[i-2])

            dst = s_distances[i]
            ang = s_angles[i] * math.pi / 180.0                           
            #if i == 2:
            #    tor = 90.0 * pi / 180.0
            #    cvec = np.array([0, 1, 0])                    
            #else:
            #    tor = s_dihedrals[i] * pi / 180.0
            #    cvec = np.array(s_coords[i-3])
            tor = s_dihedrals[i] * math.pi / 180.0
            cvec = np.array(s_coords[i-3])
                
            v1 = avec - bvec
            v2 = avec - cvec

            n = np.cross(v1, v2)
            nn = np.cross(v1, n)

            n /= np.linalg.norm(n)
            nn /= np.linalg.norm(nn)

            n *= -np.sin(tor)
            nn *= np.cos(tor)

            v3 = n + nn
            v3 /= np.linalg.norm(v3)
            v3 *= dst * np.sin(ang)

            v1 /= np.linalg.norm(v1)
            v1 *= dst * np.cos(ang)

            position = avec + v3 - v1  
            s_coords.append(position)                      
    
    # returns ASE Atoms object - minus dummy atoms                   
    structure = Atoms(
        symbols=s_atomic_symbols[3:], positions=s_coords[3:],
        cell=s_cell,
        pbc=(True, True, True)
    )

    if return_ase:
        return structure

    pymatgen_structure = AseAtomsAdaptor.get_structure(structure)
    return pymatgen_structure.to(fmt="cif")


def format_zmatrix_str(zmatrix_str):
    out = []
    tokens = zmatrix_str.split(' ')

    # lattice
    lattice_lens = tokens[:3]
    lattice_angles = [round(float(x)) for x in tokens[3:6]]
    out.append(' '.join(lattice_lens))
    out.append(' '.join(map(str, lattice_angles)))

    tokens = tokens[6:]

    # atoms
    temp = []
    for t in tokens:
        if t.isalpha():
            if temp:
                out.append(' '.join(temp))
                temp = []
            out.append(t)
        else:
            if len(temp) == 0:
                # round to 1 decimals if it is between 0 and 10
                if 0 < float(t) < 10:
                    temp.append(str(round(float(t), 1)))
                else:
                    # round to integer
                    temp.append(str(round(float(t))))
            else:
                # round to integer
                temp.append(str(round(float(t))))

    if temp:
        out.append(' '.join(temp))

    return '\n'.join(out)