import numpy as np
import random
from tqdm import tqdm

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
    
    def fit(self, signature, library, 
            add_noise=False, center=True,
            verbose: bool=False,
            min_size: int=5, max_size: int=4000,
            accuracy: int=40, deep_accuracy: int=50, # ???
            ):
        sig_hash = hash(signature.to_string()) # ???
        signature, abs_signature = genesig.process_signature(signature, center=center, add_noise=add_noise)
        
        sig2i = {}
        i2sig = {}
        for i, s in enumerate(signature.index):
            sig2i[s] = i
            i2sig[i] = s    

        res = sigtest.sig_enrich(
            signature, abs_signature, sig2i, i2sig, library, 
            seed = self.seed, processes=self.processes,
            verbose=verbose, min_size=min_size, max_size=max_size,
            accuracy=accuracy, deep_accuracy=deep_accuracy, # ???
            )
        return res
