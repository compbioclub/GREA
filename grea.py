import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy

import enrich
import genesig
import sigtest

class GREA(object):

    def __init__(self, seed=0,
                 perm_n = 1000, processes = 4,
                 verbose= True,
                 ) -> None:
        self.verbose = verbose
        self.processes = processes
        if seed is None:
            seed = np.random.randint(-10000000, 100000000)
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed

        if perm_n < 1000 and not symmetric:
            if verbose:
                print('Low numer of permutations can lead to inaccurate p-value estimation. Symmetric Gamma distribution enabled to increase accuracy.')
            symmetric = True
        elif perm_n < 500:
            if verbose:
                print('Low numer of permutations can lead to inaccurate p-value estimation. Consider increasing number of permutations.')
            symmetric = True
    
    def fit(self, sig_name, sig_val, library, method='KS',
            sig_type='ss', sample_type='ss', sig_sep=',',
            n_perm=1000,
            add_noise=False, center=True,
            verbose: bool=False,
            min_size: int=5, max_size: int=4000,
            accuracy: int=40, deep_accuracy: int=50, # ???
            ):
        
        sig_name = deepcopy(sig_name)
        sig_val = deepcopy(sig_val)
        # sig_name: array (n_sig x n_sample), the name of signature
        # sig_val: array (n_sig x n_sample), the value of signature, used for ranking the sig_name

        sig_val = genesig.process_signature(sig_val, center=center, add_noise=add_noise)
        res = sigtest.sig_enrich(
                sig_name, sig_val, library, sig_sep=sig_sep,
                method=method, n_perm=n_perm,
                seed = self.seed, processes=self.processes,
                verbose=verbose, min_size=min_size, max_size=max_size,
                accuracy=accuracy, deep_accuracy=deep_accuracy, # ???
        )
        return res