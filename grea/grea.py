import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy

import grea.rankscore as rankscore
import grea.enrich_test as enrich_test

class GREA(object):

    def __init__(self, seed=0,
                 processes = 4,
                 verbose= True,
                 ) -> None:
        self.verbose = verbose
        self.processes = processes
        if seed is None:
            seed = np.random.randint(-10000000, 100000000)
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed


    def fit(self, sig_name, sig_val, library, 
            metric='ESD',
            prob_method='perm', sig_sep=',',
            n_perm=1000,
            add_noise=False, center=True,
            verbose: bool=False, symmetric=False,
            save_permutation: bool=False,
            batch = True
            ):
        
        if metric not in ['KS-ES', 'KS-ESD', 'RC-AUC']:
            raise ValueError("metic must be 'KS-ES', 'KS-ESD', or 'RC-AUC'.")

        if prob_method not in ['perm', 'gamma']:
            raise ValueError("metic must be 'perm', or 'gamma'.")

        if n_perm < 1000 and not symmetric:
            if verbose:
                print('Low numer of permutations can lead to inaccurate p-value estimation. Symmetric Gamma distribution enabled to increase accuracy.')
            symmetric = True
        elif n_perm < 500:
            if verbose:
                print('Low numer of permutations can lead to inaccurate p-value estimation. Consider increasing number of permutations.')
            symmetric = True

        #2dim    
        if sig_name.ndim == 1:
            sig_name = sig_name.reshape(-1, 1)
        if sig_val.ndim == 1:
            sig_val = sig_val.reshape(-1, 1)

        # Ensure that sig_name and sig_val have the same number of rows
        if sig_name.shape[0] != sig_val.shape[0]:
            raise ValueError("sig_name and sig_val must be same number of rows.")


        #IF the column of sig_name is 1 and the column of sig_val is greater than 1, 
        #repeat sig_name to match the number of columns of sig_val
        if sig_name.shape[1] == 1 and sig_val.shape[1] > 1:
            sig_name = np.repeat(sig_name, sig_val.shape[1], axis=1)
        sig_name = deepcopy(sig_name)
        sig_val = deepcopy(sig_val)
        # sig_name: array (n_sig x n_sample), the name of signature
        # sig_val: array (n_sig x n_sample), the value of signature, used for ranking the sig_name

        sig_val = rankscore.process_signature(sig_val, center=center, add_noise=add_noise)
        if batch:
            res = enrich_test.sig_enrich_batch(
                    sig_name, sig_val, library, 
                    sig_sep=sig_sep,
                    metric=metric, 
                    n_perm=n_perm, 
                    prob_method=prob_method,
                    processes=self.processes,
                    verbose=verbose, 
                    save_permutation=save_permutation,
            )
        else:
            res = enrich_test.sig_enrich(
                    sig_name, sig_val, library, 
                    sig_sep=sig_sep,
                    metric=metric, 
                    n_perm=n_perm, 
                    prob_method=prob_method,
                    processes=self.processes,
                    verbose=verbose, 
                    save_permutation=save_permutation,
            )
        return res