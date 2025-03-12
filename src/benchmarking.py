import blitzgsea as blitz
import time
import pandas as pd
import numpy as np
from src.grea import GREA
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import multiprocessing
import time
from tqdm import tqdm 
import gseapy

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
    results = []
    times = []
    signature.columns = ["i","v"]
    sig_name = signature.values[:, 0][:, np.newaxis]
    print(sig_name.shape)
    sig_val = signature.values[:, 1][:, np.newaxis]
    print(sig_val.shape)
    if method == 'blitz':

        st = time.time()
        res1 = blitz.gsea(signature, library, permutations=ii, processes=1, seed=j*i, signature_cache=False)
        times.append(time.time() - st)
        results = {'df': res1, 'pval': 'pval'}
    
    if method == 'gseapy':
        sig = signature.sort_values('v', ascending=False)
        sig = sig[~sig.index.duplicated(keep='first')]
        st = time.time()
        res2 = chopped_gsea(sig, library, 1, permutation_num=ii, seed=i*j+j, max_lib_size=25)
        res2 = res2.set_index('Term')
        times.append(time.time() - st)
        results = {'df': res2, 'pval': 'NOM p-val'}
    
    if method == 'grea_es':
        st = time.time()
        obj = GREA(processes=1, perm_n=ii, seed=j*i, symmetric=True, verbose=False)
        res3_es = obj.fit(sig_name, sig_val, library)
        res3_es = res3_es.set_index('Term')
        times.append(time.time() - st)
        results = {'df': res3_es, 'pval': 'es_pval'}
    
    if method == 'grea_esd':
        st = time.time()
        obj = GREA(processes=1, perm_n=ii, seed=j*i, symmetric=True, verbose=False)
        res3_esd = obj.fit(sig_name, sig_val, library, cal_method='ESD')
        res3_esd = res3_esd.set_index('Term')
        times.append(time.time() - st)
        results = {'df': res3_esd, 'pval': 'esd_pval'}
    
    if method == 'grea_auc':
        st = time.time()
        obj = GREA(processes=1, perm_n=ii, seed=j*i, symmetric=True, verbose=False)
        res3_auc = obj.fit(sig_name, sig_val, library, method='RC')
        res3_auc = res3_auc.set_index('Term')
        times.append(time.time() - st)
        results = {'df': res3_auc, 'pval': 'AUC_pval'}

    return method, results, times


def benchmark_parallel(signature, library, methods=None, n=11, base_perm=250,):
    if methods is None:
        methods = ['blitz', 'gseapy', 'grea_es', 'grea_esd', 'grea_auc']
    
    results = {'pval': {method: [] for method in methods}, 'times': {method: [] for method in methods}}

    for i in range(1, n):
        pvals = {method: [] for method in methods}
        times = {method: [] for method in methods}
        print(f"Outer loop time: {i}")
        
        # 
        with multiprocessing.Pool(processes=len(methods)) as pool:
            tasks = [(method, signature, library, i, j, i * base_perm) for j in range(1, n) for method in methods]
            results_list = pool.starmap(run_method, tasks)
        
        method_results = {}
        for method, res_info, time_list in results_list:
            method_results[method] = res_info
            times[method].append(time_list)

        print(method_results)
        
        # Find pathways that overlap across all methods
        if method_results:
            all_sets = [set(res_info['df'].index) for res_info in method_results.values()]
            overlap = list(set.intersection(*all_sets)) if all_sets else []
            
            # Extract p-values for overlapping pathways
            for method, res_info in method_results.items():
                if overlap:
                    pval = res_info['pval']
                    p_values = np.array(res_info['df'].loc[overlap, pval]).astype("float")
                    pvals[method].append(p_values)
                else:
                    pvals[method].append(np.array([]))
        
        # Store results for this iteration
        for method in methods:
            results['pval'][method].append(pvals[method])
            results['times'][method].append(times[method])
    
        
    return results