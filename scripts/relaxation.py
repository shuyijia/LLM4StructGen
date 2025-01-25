import argparse
import ase.io
import numpy as np
from ase import Atoms
from glob import glob
from pathlib import Path
from tqdm import tqdm
from mace.calculators import mace_mp
from ase.optimize import FIRE

import signal
from functools import wraps
from contextlib import contextmanager

def function_timeout(seconds: int):
    """Define a decorator that sets a time limit for a function call.

    Args:
        seconds (int): Time limit.

    Raises:
        SystemExit: Timeout exception.

    Returns:
        Decorator: Timeout Decorator.
    """
    def decorator(func):
        @contextmanager
        def time_limit(seconds_):
            def signal_handler(signum, frame):  # noqa
                raise SystemExit("Timed out!")  #TimeoutException
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds_)
            try:
                yield
            finally:
                signal.alarm(0)
        @wraps(func)
        def wrapper(*args, **kwargs):
            with time_limit(seconds):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class SafeFIRE(FIRE):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    @function_timeout(seconds=180)
    def step(self, f=None):
        if np.isnan(self.atoms.get_potential_energy()):
            raise Exception("NaN")

        super().step(f)

class Relaxation():
    def __init__(self, cif_dir, calculator="mace", fmax=0.2, steps=100):
        self.cif_dir = Path(cif_dir)
        self.fmax = fmax
        self.steps = steps

        # get atoms
        cifs = glob(str(self.cif_dir / "*.cif"))
        assert len(cifs) > 0, "No CIF files found in the directory"

        self.atoms_list = {}
        for cif in tqdm(cifs):
            _id = Path(cif).stem
            atoms = ase.io.read(cif)
            self.atoms_list[_id] = atoms

        if calculator == 'mace':
            self.calculator = mace_mp(
                model="large",
                dispersion=False,
                default_dtype="float32",
                device="cuda"
            )
        else:
            raise NotImplementedError
    
    @function_timeout(seconds=180)
    def relax_single(self, atoms):
        atoms.calc = self.calculator
        opt = SafeFIRE(atoms=atoms, trajectory=None)
        opt.run(fmax=self.fmax, steps=self.steps)

        return atoms, atoms.get_potential_energy()
    
    def relax_all(self):
        relaxed_atoms_list = {}

        for _id, atoms in tqdm(self.atoms_list.items()):
            try:
                relaxed_atoms, energy = self.relax_single(atoms)
                relaxed_atoms_list[_id] = {
                    "atoms": relaxed_atoms,
                    "energy": energy
                }
            except Exception as e:
                print(f"Relaxation of {_id} timed out. Exception: {e}")

        print(f"Relaxed {len(relaxed_atoms_list)} out of {len(self.atoms_list)} structures")
            
        # save relaxed atoms
        save_dir = Path(args.save_dir) / (str(Path(args.cif_dir).parent.name) + "_relaxed")
        save_dir.mkdir(parents=True, exist_ok=True)

        for _id, relaxed_atoms in relaxed_atoms_list.items():
            ase.io.write(str(save_dir / f"{_id}.cif"), relaxed_atoms["atoms"])
        
        return relaxed_atoms_list
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relax CIF files")
    parser.add_argument("--cif_dir", type=str, help="Directory containing CIF files")
    parser.add_argument("--save_dir", type=str, default="outputs/", help="Directory to save relaxed CIF files")
    parser.add_argument("--calculator", type=str, default="mace", help="Calculator to use")
    parser.add_argument("--fmax", type=float, default=0.2, help="Maximum force")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps")
    args = parser.parse_args()

    relaxation = Relaxation(args.cif_dir, args.calculator, args.fmax, args.steps)
    relaxed_atoms_list = relaxation.relax_all()