import os
import json
import ase.io
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
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

        # load output json
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

        if self.args.save:
            config_path = Path(self.args.config)
            save_dir = config_path.parent / f"{config_path.stem}" / "cifs"
            os.makedirs(save_dir, exist_ok=True)


        failed_list = []
        for i, s in enumerate(tqdm(outputs)):
            # retrieve the generated string
            _splits = s.strip().split("\n", 1)
            assert len(_splits) == 2
            generated_str = _splits[1]

            # decode
            # decoded = self.encoder.decode(generated_str)
            decoded = self.safe_decode(generated_str)

            if decoded is None:
                failed_list.append(generated_str)
                continue

            if self.args.save:
                # save as cif
                cif_path = save_dir / f"{i}.cif"
                ase.io.write(str(cif_path), decoded)

        # get timestamp
        now = datetime.now()
        formatted_time = now.strftime("%d%m%Y_%H%M%S")
        
        info_str = f"""
        Datetime: {formatted_time}\n
        Total: {len(outputs)}\n
        Failed: {len(failed_list)}\n
        Success Rate: {1 - len(failed_list) / len(outputs)}
        """

        # save info_str to file without overwriting the file if the file already exists
        info_file = save_dir / "info.txt"
        with open(info_file, 'a') as f:
            f.write(info_str)
        
        print(info_str)

    def safe_decode(self, generated_str):
        try:
            decoded = self.encoder.decode(generated_str)
            return decoded
        except Exception as e:
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the generated string representations for crystal structures")
    parser.add_argument("--config", type=str, help="path to the data file")
    parser.add_argument("--save", action='store_true', help="save the decoded cifs")
    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.decode_to_cifs()