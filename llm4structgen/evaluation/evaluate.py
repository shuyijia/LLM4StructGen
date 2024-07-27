import os
import json
import ase.io
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

class Evaluator:
    """
    decode and evaluate the generated string representations
    for crystal structures.
    
    evaluation:
        - CDVAE metrics 
    """

    def __init__(self, args):
        self.args = args

        # load config
        f = open(self.args.config, 'r')
        cfg_dict = json.load(f)
        f.close()

        self.cfg = OmegaConf.create(cfg_dict)

        # load encoder
        if self.cfg.representation_type == "cartesian":
            from llm4structgen.representations.cartesian import Cartesian
            self.encoder = Cartesian()
        elif self.cfg.representation_type == "zmatrix":
            from llm4structgen.representations.z_matrix import ZMatrix
            self.encoder = ZMatrix()
        elif self.cfg.representation_type == "distance":
            from llm4structgen.representations.distance_matrix import DistanceMatrix
            self.encoder = DistanceMatrix()
        elif self.cfg.representation_type == "slices":
            from llm4structgen.representations.slices import SLICES
            self.encoder = SLICES()
        else:
            raise ValueError(f"Invalid representation type: {self.cfg.representation_type}")
        
    def decode_to_cifs(self):
        """
        decode the generated string representations to cifs
        """
        outputs = self.cfg.get("outputs", None)
        if outputs is None:
            raise ValueError("No generated strings found in file")

        for s in tqdm(outputs):
            # retrieve the generated string
            _splits = s.strip().split("\n", 1)
            assert len(_splits) == 2
            generated_str = _splits[1]

            # decode
            decoded = self.encoder.decode(generated_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the generated string representations for crystal structures")
    parser.add_argument("--config", type=str, help="path to the data file")
    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.decode_to_cifs()