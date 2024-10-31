import os
import warnings
warnings.filterwarnings("ignore")

import json
import ase.io
import argparse
import warnings
from tqdm import tqdm
from glob import glob
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

from llm4structgen.evaluation.cdvae.eval_utils import *

# Suppress warnings from the pymatgen package
warnings.filterwarnings("ignore", category=FutureWarning, module="pymatgen")

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
                try:
                    ase.io.write(str(cif_path), decoded)
                except:
                    continue;

        # get timestamp
        now = datetime.now()
        formatted_time = now.strftime("%d%m%Y_%H%M%S")
        
        info_str = f"""
        Datetime: {formatted_time}\n
        Total: {len(outputs)}\n
        Failed: {len(failed_list)}\n
        Success Rate: {1 - len(failed_list) / len(outputs)}
        """

        print(info_str)

        # save info_str to file without overwriting the file if the file already exists
        if self.args.save:
            info_file = save_dir.parent / "info.txt"
            with open(info_file, 'a') as f:
                f.write(info_str)

        return save_dir if self.args.save else None

    def safe_decode(self, generated_str):
        try:
            decoded = self.encoder.decode(generated_str)
            return decoded
        except Exception as e:
            return None

    def decode_and_evaluate(self):
        """
        decode and evaluate the generated string representations
        """
        if not self.args.save:
            raise ValueError("Save flag must be set to True for evaluation")

        # decode to cifs and save
        print("Decoding generated strings to cifs...")
        save_dir = self.decode_to_cifs()

        # get cifs
        cifs = glob(f"{save_dir.parent}/cifs/*.cif")
        assert len(cifs) >= 10000, f"Expected 10000 cifs for CDVAE metrics, got {len(cifs)}"

        # get crystals for generated structures
        print("Getting crystals for generated structures...")
        crys_array_list = get_crystals_list(cifs)
        gen_crystals = p_map(lambda x: Crystal(x), crys_array_list)

        # get crystals for testset structures
        print("Getting crystals for test set structures...")
        csv = pd.read_csv(self.args.testset_path)
        test_crystals = p_map(get_gt_crys_ori, csv['cif'])

        print(len(gen_crystals))
        print(len(test_crystals))
        # get CDVAE metrics
        gen_evaluator = GenEval(
            gen_crystals, 
            test_crystals, 
            eval_model_name=self.args.eval_model_name
        )

        print("Calculating CDVAE metrics...")
        all_metrics = {}
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

        # save metrics
        metrics_path = save_dir.parent / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f)

        print(all_metrics)

    def evaluate(self):
        """
        evaluate a directory of cifs
        """
        # get cifs
        cif_dir = Path(self.args.cif_dir)
        cifs = glob(f"{cif_dir}/*.cif")
        assert len(cifs) >= 10000, f"Expected 10000 cifs for CDVAE metrics, got {len(cifs)}"

        # get crystals for generated structures
        print("Getting crystals for generated structures...")
        crys_array_list = get_crystals_list(cifs)
        gen_crystals = p_map(lambda x: Crystal(x), crys_array_list)

        # get crystals for testset structures
        print("Getting crystals for test set structures...")
        csv = pd.read_csv(self.args.testset_path)
        test_crystals = p_map(get_gt_crys_ori, csv['cif'])

        # get CDVAE metrics
        gen_evaluator = GenEval(
            gen_crystals, 
            test_crystals, 
            eval_model_name=self.args.eval_model_name
        )

        print("Calculating CDVAE metrics...")
        all_metrics = {}
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

        # save metrics
        metrics_path = cif_dir.parent / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f)

        print(all_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the generated string representations for crystal structures")
    parser.add_argument("--config", type=str, help="path to the data file")
    parser.add_argument("--save", action='store_true', help="save the decoded cifs")
    parser.add_argument("--testset_path", type=str, help="path to the testset csv", default="data/eval/test.csv")
    parser.add_argument("--eval_model_name", type=str, help="name of the evaluation model", default="mp20")
    parser.add_argument("--cif_dir", type=str, help="path to the directory of cifs", default=None)
    parser.add_argument("--eval", action='store_true', help="evaluate the decoded cifs")
    args = parser.parse_args()

    evaluator = Evaluator(args)

    if args.cif_dir:
        evaluator.evaluate()
    elif args.eval:
        evaluator.decode_and_evaluate()
    else:
        evaluator.decode_to_cifs()
