import math
import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

def zmatrix2struct(zmatrix):
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

    pymatgen_structure = AseAtomsAdaptor.get_structure(structure)
    return pymatgen_structure.to(fmt="cif")
