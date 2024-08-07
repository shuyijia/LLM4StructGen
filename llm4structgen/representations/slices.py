import tensorflow as tf
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from invcryrep.invcryrep import InvCryRep
from pymatgen.core import Structure
from mace.calculators import mace_mp
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import FIRE
from invcryrep.invcryrep import function_timeout

from .base_representation import BaseRepresentation
from llm4structgen.datasets.prompts import SLICES_GENERATION_PROMPT_HEADER

class SLICES(BaseRepresentation):
    def __init__(
        self, 
        translate: bool = False,
        rotate: bool = False,
        permute: bool = False,
        randomized: bool = False,
    ):
        super().__init__()
        self.translate = translate
        self.rotate = rotate
        self.permute = permute
        self.randomized = randomized

        # prevent TensorFlow from allocating all available GPU memory at once
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)

        self.backend = InvCryRepMace()

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

        # by default, pymatgen.core.Structure expects fractional coors
        # positions @ np.linalg.inv(cell)
        structure = Structure(
            cell, 
            atomic_symbols, 
            positions, 
            coords_are_cartesian=True
        )

        if self.randomized:
            slices_str = self.backend.structure2SLICESAug(structure=structure,num=1)[0]
        else:
            slices_str = self.backend.structure2SLICES(structure)

        return slices_str
    
    def decode(self, input_str: str) -> Atoms:
        structure, final_energy_per_atom_IAP = self.backend.SLICES2structure(input_str)
        atoms = AseAtomsAdaptor.get_atoms(structure)

        return atoms
    
    @property
    def prompt_header(self):
        return SLICES_GENERATION_PROMPT_HEADER

class InvCryRepMace(InvCryRep):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.calculator = mace_mp(model="large", dispersion=False, default_dtype="float32", device='cuda')
        self.adaptor = AseAtomsAdaptor()
    
    @function_timeout(seconds=180)
    def m3gnet_relax(self,struc):
        """Replaces m3gnet with mace optimization 60s time limit"""
        
        struc_ase = self.adaptor.get_atoms(struc)
        struc_ase.calc = self.calculator
        opt = FIRE(atoms=struc_ase,trajectory='out.traj')
        opt.run(fmax=self.fmax,steps=self.steps)

        relaxed_struc = self.adaptor.get_structure(struc_ase)
        
        
        return relaxed_struc, struc_ase.get_potential_energy()

    @function_timeout(seconds=360)
    def m3gnet_relax_large_cell1(self,struc):
        """Replaces m3gnet with mace optimization 360s time limit"""

        struc_ase = self.adaptor.get_atoms(struc)
        struc_ase.calc = self.calculator
        opt = FIRE(atoms=struc_ase,trajectory='out.traj')
        opt.run(fmax=self.fmax,steps=self.steps)
        
        relaxed_struc = self.adaptor.get_structure(struc_ase)
        
        return relaxed_struc, struc_ase.get_potential_energy()

    @function_timeout(seconds=1000)
    def m3gnet_relax_large_cell2(self,struc):
        """Replaces m3gnet with mace optimization 1000s time limit"""

        struc_ase = self.adaptor.get_atoms(struc)
        struc_ase.calc = self.calculator
        opt = FIRE(atoms=struc_ase,trajectory='out.traj')
        opt.run(fmax=self.fmax,steps=self.steps)
        
        relaxed_struc = self.adaptor.get_structure(struc_ase)
        
        return relaxed_struc, struc_ase.get_potential_energy()