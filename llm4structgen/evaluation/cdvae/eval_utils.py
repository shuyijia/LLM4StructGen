from glob import glob
import os
import signal
from collections import Counter
import numpy as np
import multiprocessing as mp
import os
import signal
import time  # For simulating delay in get_fingerprints

import threading
import queue

import concurrent.futures

import time

import ase.io
import itertools
from p_tqdm import p_map
import numpy as np
import pandas as pd
from collections import Counter
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition

import smact
from smact.screening import pauling_test

from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist

from llm4structgen.evaluation.cdvae.constants import *
from llm4structgen.evaluation.cdvae.scaler import StandardScaler

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')
CompScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.
)

def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True
    
def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    # if len(list(itertools.product(*ox_combos))) > 1e5:
    #     return False
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False

def compute_cov(crys, gt_crys,
                struc_cutoff, comp_cutoff, num_gen_crystals=None):

    #struc_fps = [c.struct_fp for c in crys]
    #comp_fps = [c.comp_fp for c in crys]
    struc_fps = []
    comp_fps = []
    for c in crys:
        if (not c.valid):
            print("Invalid")
            continue;
        try:
            struc_fps.append(c.struct_fp)
            comp_fps.append(c.comp_fp)
        except:
            continue

    #gt_struc_fps = [c.struct_fp for c in gt_crys]
    #gt_comp_fps = [c.comp_fp for c in gt_crys]
    gt_struc_fps = []
    gt_comp_fps = []

    for c in gt_crys:
        if (not c.valid):
            continue;
        try:
            gt_struc_fps.append(c.struct_fp)
            gt_comp_fps.append(c.comp_fp)
        except:
            continue

    print("Getting length", flush=True)
    print(len(gt_crys), flush=True)
    print(len(crys), flush=True)

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)

    comp_fps = CompScaler.transform(comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
        'amsd_recall': np.mean(struc_recall_dist),
        'amsd_precision': np.mean(struc_precision_dist),
        'amcd_recall': np.mean(comp_recall_dist),
        'amcd_precision': np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        'struc_recall_dist': struc_recall_dist.tolist(),
        'struc_precision_dist': struc_precision_dist.tolist(),
        'comp_recall_dist': comp_recall_dist.tolist(),
        'comp_precision_dist': comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict

def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps

def get_crystals_list(cifs):
    crystal_array_list = []

    i = 0
    for cif in cifs:
        try:
            atoms = ase.io.read(cif)
        except:
            i+=1
            continue;

        crystal_array_list.append(
            {
                "frac_coords": atoms.get_scaled_positions(),
                "atom_types": atoms.get_atomic_numbers(),
                "lengths": atoms.cell.cellpar()[:3],
                "angles": atoms.cell.cellpar()[3:],
            }
        )
    
    print(i)
    return crystal_array_list

def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif,fmt='cif')
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles)
    }
    return Crystal(crys_array_dict)
    

class Crystal(object):
    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)

        try:
            self.get_structure()
            self.get_composition()
            self.get_validity()
            #self.get_fingerprints()
        except:
            pass

        self.valid = False
        self.process_item_with_timeout(timeout=30)  # Set timeout here for each item



    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'            
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self, shared_dict):
        shared_dict['valid'] = True
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        #self.comp_fp = CompFP.featurize(comp)
        shared_dict['comp_fp'] = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            shared_dict['valid'] = False
            shared_dict['comp_fp'] = None
            shared_dict['struct_fp'] = None
            #self.valid = False
            #self.comp_fp = None
            #self.struct_fp = None
            return
        #self.struct_fp = np.array(site_fps).mean(axis=0)
        shared_dict['struct_fp'] = np.array(site_fps).mean(axis=0)

    def process_item_with_timeout(self, timeout):
        manager = mp.Manager()
        shared_dict = manager.dict()

        p = mp.Process(target=self.get_fingerprints, args=(shared_dict,))
        p.start()

        p.join(timeout=timeout)  # Wait for the function to finish

        if p.is_alive():
            print(f"get_fingerprints() timed out after {timeout} seconds. Skipping to the next item...")
            # Forcefully kill the process if it's still alive
            os.kill(p.pid, signal.SIGKILL)
            p.join()  # Clean up the process
            self.valid = False
        else:
            self.valid = shared_dict.get('valid', False)
            self.comp_fp = shared_dict.get('comp_fp', None)
            self.struct_fp = shared_dict.get('struct_fp', None)

            #self.valid = True


class GenEval(object):
    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_v = []
        struct_v = []

        for c in self.crys:
            if (not c.valid):
                print("Invalid")
                continue;
            try:
                comp_v.append(c.comp_valid)
                struct_v.append(c.struct_valid)
            except:
                continue
        #comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        #struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        comp_valid = np.array(comp_v).mean()
        struct_valid = np.array(struct_v).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}


    def get_density_wdist(self):
        pred_densities = []
        gt_densities = []

        for c in self.valid_samples:
            if not c.valid:
                print("Invalid")
                continue
            try:
                pred_densities.append(c.structure.density)
            except:
                continue

        for c in self.gt_crys:
            if not c.valid:
                print("Invalid")
                continue
            try:
                gt_densities.append(c.structure.density)
            except:
                continue


        #pred_densities = [c.structure.density for c in self.valid_samples]
        #gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}


    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples if c.valid]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys if c.valid]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        # if self.eval_model_name is not None:
        #     pred_props = prop_model_eval(self.eval_model_name, [
        #                                  c.dict for c in self.valid_samples])
        #     gt_props = prop_model_eval(self.eval_model_name, [
        #                                c.dict for c in self.gt_crys])
        #     wdist_prop = wasserstein_distance(pred_props, gt_props)
        #     return {'wdist_prop': wdist_prop}
        # else:
        #     return {'wdist_prop': None}
        raise NotImplementedError

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_density_wdist())
        # metrics.update(self.get_prop_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_coverage())
        return metrics




