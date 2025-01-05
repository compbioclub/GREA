import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy

from src import enrich
from src import genesig
from src import sigtest

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
            prob_method='perm', sig_sep=',',
            n_perm=1000,
            add_noise=False, center=True,
            verbose: bool=False,
            min_size: int=5, max_size: int=4000,
            accuracy: int=40, deep_accuracy: int=50, # ???
            cal_method: str='ES',):
        #2dim    
        if sig_name.ndim == 1:
            sig_name = sig_name.reshape(-1, 1)
        if sig_val.ndim == 1:
            sig_val = sig_val.reshape(-1, 1)

        # Ensure that sig_name and sig_val have the same number of rows
        if sig_name.shape[0] != sig_val.shape[0]:
            raise ValueError("sig_name and sig_val must be same number of rows.")


        #IF the column of sig_name is 1 and the column of sig_val is greater than 1, repeat sig_name to match the number of columns of sig_val
        if sig_name.shape[1] == 1 and sig_val.shape[1] > 1:
            sig_name = np.repeat(sig_name, sig_val.shape[1], axis=1)
        sig_name = deepcopy(sig_name)
        sig_val = deepcopy(sig_val)
        # sig_name: array (n_sig x n_sample), the name of signature
        # sig_val: array (n_sig x n_sample), the value of signature, used for ranking the sig_name

        sig_val = genesig.process_signature(sig_val, center=center, add_noise=add_noise)
        res = sigtest.sig_enrich(
                sig_name, sig_val, library, sig_sep=sig_sep,
                method=method, n_perm=n_perm, prob_method=prob_method,
                seed = self.seed, processes=self.processes,
                verbose=verbose, min_size=min_size, max_size=max_size,
                accuracy=accuracy, deep_accuracy=deep_accuracy, # ???
        )
        return res