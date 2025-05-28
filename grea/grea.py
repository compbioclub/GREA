import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy

import grea.rankscore as rankscore
import grea.enrich_test as enrich_test
import grea.library as library
import grea.out as out
from grea import plot as pl

def pheno_prerank_enrich(rank_df, libraries, **kwargs):
    obj = _GREA(libraries, **kwargs)
    sig_names = rank_df.index.to_numpy()
    sig_vals = rank_df.to_numpy()
    obs_names = rank_df.columns.to_list()
    obj._check_sig_shape(sig_names, sig_vals, obs_names, **kwargs)
    obj._enrich()
    return obj    

def obs_prerank_enrich(rank_df, libraries, **kwargs):
    obj = _GREA(libraries, get_pval=False, **kwargs)
    sig_names = rank_df.index.to_numpy()
    sig_vals = rank_df.to_numpy()
    obs_names = rank_df.columns.to_list()
    obj._check_sig_shape(sig_names, sig_vals, obs_names, **kwargs)
    obj._enrich()
    return obj    


class _GREA(object):

    def __init__(self, libraries, seed=0, prob_method='perm', 
                 sig_sep=',', verbose=True, sig_upper=True,
                 add_lib_key=True, min_size=5, max_size=1000, n_process=4,
                 n_perm=1000, symmetric=False,
                 center=True, add_noise=False,
                 get_pval=True, get_lead_sigs=True,
                 save_permutation: bool=False) -> None:

        if type(libraries) == list:
            self.term_dict = library.get_library_from_names(libraries, add_lib_key=add_lib_key, min_size=min_size, max_size=max_size)
        elif type(libraries) == dict:
            self.term_dict = libraries
        else:
            raise ValueError(f'libraries must be list of library names, or a Python dictionary, where each key is a pathway name and the corresponding value is a list of genes.')

        if seed is None:
            seed = np.random.randint(-10000000, 100000000)
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed

        if prob_method not in ['perm', None]:
            raise ValueError("prob_method '{prob_method}' must be 'perm', 'None'.")
        self.prob_method = prob_method

        symmetric = self._check_n_perm(n_perm, symmetric)
        self.n_perm = n_perm
        self.symmetric = symmetric

        self.n_process = n_process
        self.sig_sep = sig_sep
        self.verbose = verbose
        self.save_permutation = save_permutation
        self.get_pval = get_pval
        self.get_lead_sigs = get_lead_sigs
        self.sig_upper = sig_upper
        self.center = center
        self.add_noise = add_noise
        
    def _check_sig_shape(self, sig_names, sig_vals, obs_names, center=True, add_noise=False, **kwargs):
        #2dim    
        if sig_names.ndim == 1:
            sig_names = sig_names.reshape(-1, 1)
        if sig_vals.ndim == 1:
            sig_vals = sig_vals.reshape(-1, 1)

        # Ensure that sig_name and sig_val have the same number of rows
        if sig_names.shape[0] != sig_vals.shape[0]:
            raise ValueError("sig_name and sig_val must be same number of rows.")

        #IF the column of sig_name is 1 and the column of sig_val is greater than 1, 
        #repeat sig_name to match the number of columns of sig_val
        if sig_names.shape[1] == 1 and sig_vals.shape[1] > 1:
            sig_names = np.repeat(sig_names, sig_vals.shape[1], axis=1)
        sig_names = deepcopy(sig_names)
        sig_vals = deepcopy(sig_vals)
        # sig_name: array (n_sig x n_sample), the name of signature
        # sig_vals: array (n_sig x n_sample), the value of signature, used for ranking the sig_name
        sig_vals = rankscore.process_signature(sig_vals, center, add_noise)
        
        self.sig_names = sig_names
        self.sig_vals = sig_vals
        self.obs_names = obs_names
        self.center = center
        self.add_noise = add_noise

    def _enrich(self):
        res = enrich_test.enrich(self)
        return res
    
    def get_enrich_results(self, metric):
        self._check_metric(metric)
        return out.enrich_long_df(self, metric)
    
    def get_enrich_score(self, metric):
        self._check_metric(metric)
        return out.enrich_wide_df(self, metric)
    
    def pl_running_sum(self, metric, term, obs_id, **kwargs):
        self._check_metric(metric)
        self._check_term(term)
        self._check_obs(obs_id)
        return pl.running_sum(self, metric, term, obs_id, **kwargs)

    def _check_n_perm(self, n_perm, symmetric):
        if n_perm is None:
            return None
        if n_perm < 1000 and not symmetric:
            print('Low numer of permutations can lead to inaccurate p-value estimation. Symmetric Gamma distribution enabled to increase accuracy.')
            symmetric = True
        elif n_perm < 500:
            print('Low numer of permutations can lead to inaccurate p-value estimation. Consider increasing number of permutations.')
            symmetric = True
        return symmetric
    
    def _check_metric(self, metric):
        if metric not in ['KS-ES', 'KS-ESD', 'RC-AUC', 'RC-nAUC', 'nRC-AUC']:
            raise ValueError(f"Metric '{metric}' must be 'KS-ES', 'KS-ESD', 'RC-nAUC', 'RC-nAUC', 'nRC-AUC'.")
    
    def _check_term(self, term):
        if term not in self.term_names:
            raise ValueError(f"The geneset '{term}' is not enriched.")
        
    def _check_obs(self, obs_id):
        if obs_id not in self.obs_names:
            raise ValueError(f"The obs_id '{obs_id}' is not in input rank_df.")