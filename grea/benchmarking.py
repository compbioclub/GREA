import blitzgsea as blitz
import time
import pandas as pd
import numpy as np
from grea.grea import GREA
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import multiprocessing
import time
from tqdm import tqdm 
import gseapy
import os

def chopped_gsea(rnk, gene_sets, processes, permutation_num=100, max_lib_size=100, outdir='test/prerank_report_kegg', format='png', seed=1):
    library_keys = list(gene_sets.keys())
    chunks = [library_keys[i:i+max_lib_size] for i in range(0, len(library_keys), max_lib_size)]
    results = []
    rnk.index = rnk.index.astype(str)
    for chunk in chunks:
        tlib = {}
        for k in chunk:
            tlib[k] = gene_sets[k]
        pre_res = gseapy.prerank(rnk=rnk, gene_sets=tlib, processes=processes, permutation_num=permutation_num, outdir=outdir, format=format, seed=seed)
        results.append(pre_res.res2d)
    return pd.concat(results)

def run_method(method, signature, library, i, j, ii):
    signature.columns = ["i","v"]
    sig_name = signature.values[:, 0][:, np.newaxis]
    sig_val = signature.values[:, 1][:, np.newaxis]
    if method == 'blitz':
        res1 = blitz.gsea(signature, library, permutations=ii, processes=1, seed=j*i, signature_cache=False)
        res1["Method"] = method 
        return method, res1
    
    elif method == 'gseapy':
        sig = signature.sort_values('v', ascending=False)
        sig = sig[~sig.index.duplicated(keep='first')]
        res2 = chopped_gsea(sig, library, processes=1, permutation_num=ii, seed=i*j+j, max_lib_size=25)
        res2 = res2.set_index('Term')
        res2["Method"] = method 
        return method, res2
    
    elif method == 'grea_es':
        obj = GREA(processes=1, perm_n=ii, seed=j*i, symmetric=True, verbose=False)
        res3_es = obj.fit(sig_name, sig_val, library)
        res3_es = res3_es.set_index('Term')
        res3_es["Method"] = method 
        return method, res3_es
    
    elif method == 'grea_esd':
        obj = GREA(processes=1, perm_n=ii, seed=j*i, symmetric=True, verbose=False)
        res3_esd = obj.fit(sig_name, sig_val, library, cal_method='ESD')
        res3_esd = res3_esd.set_index('Term')
        res3_esd["Method"] = method
        return method, res3_esd
    
    elif method == 'grea_auc':
        obj = GREA(processes=1, perm_n=ii, seed=j*i, symmetric=True, verbose=False)
        res3_auc = obj.fit(sig_name, sig_val, library, method='RC')
        res3_auc = res3_auc.set_index('Term')
        res3_auc["Method"] = method
        return method, res3_auc



def benchmark_parallel(signature, library, methods=None, n=11, base_perm=250,output_dir='result'):
    if methods is None:
        methods = ['blitz', 'gseapy', 'grea_es', 'grea_esd', 'grea_auc']
    
    for i in range(1, n):

        print(f"Outer loop iteration: {i}")
        method_results = {method: [] for method in methods}
        # 
        tasks = []
        for j in range(1, n):
            print(f"  Inner loop iteration: {j}")
            for method in methods:
                tasks.append((method, signature, library, i, j, i * base_perm))

        with multiprocessing.Pool(processes=len(methods)) as pool:
            results_list = pool.starmap(run_method, tasks)
        
        for method, res in results_list:
            method_results[method].append(res)

        for method in methods:
                combined_results = pd.concat(method_results[method])
                combined_results.to_csv(os.path.join(output_dir, f"{method}_{i}.csv"))
    
        
    return None