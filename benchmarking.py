import blitzgsea as blitz
import time
import pandas as pd
import numpy as np
from grea import GREA
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import multiprocessing
import time
from tqdm import tqdm 
import gseapy
import os
import scanpy as sc

def preprocess_signature(signature, group =None, FC = True,method = 't-test',key_added = 'ttest_symptom'):

    if FC and group is not None:
        #Adata
        adata = sc.AnnData(X=signature.T)
        adata.obs_names = signature.columns
        adata.var_names = signature.index
        #group
        adata.obs['symptom'] = pd.Categorical([group.get(x, 'Unknown') for x in adata.obs_names])
        #preprocessing
        sc.pp.filter_genes(adata, min_cells=3)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.layers["log1p"] = adata.X.copy()
        #DEG computation
        sc.tl.rank_genes_groups(
            adata, 
            groupby='symptom', 
            groups=['Symptomatic'], 
            reference='Asymptomatic',
            method=method, 
            layer='log1p',
            key_added= key_added  
        )
        #result
        res = sc.get.rank_genes_groups_df(adata, group='Symptomatic', key=key_added)
        signature = res[['names','scores']]
    else:
        return signature

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

def run_method(method, signature, library, group, i, perm,):
    signature = preprocess_signature(signature,group)
    signature.columns = ["i","v"]
    sig_name = signature.values[:, 0][:, np.newaxis]
    sig_val = signature.values[:, 1][:, np.newaxis]
    if method == 'blitz':
        res1 = blitz.gsea(signature, library, permutations=perm, processes=1, seed=perm*i, signature_cache=False)
        res1["Method"] = method 
        return method, res1
    
    elif method == 'gseapy':
        sig = signature.sort_values('v', ascending=False)
        sig = sig[~sig.index.duplicated(keep='first')]
        res2 = chopped_gsea(sig, library, processes=1, permutation_num=perm, seed=perm*i, max_lib_size=25)
        res2 = res2.set_index('Term')
        res2["Method"] = method 
        return method, res2
    
    elif method == 'grea_es':
        obj = GREA(processes=1, perm_n=perm, seed=perm*i, symmetric=True, verbose=False)
        res3_es = obj.fit(sig_name, sig_val, library)
        res3_es = res3_es.set_index('Term')
        res3_es["Method"] = method 
        return method, res3_es
    
    elif method == 'grea_esd':
        obj = GREA(processes=1, perm_n=perm, seed=perm*i, symmetric=True, verbose=False)
        res3_esd = obj.fit(sig_name, sig_val, library, cal_method='ESD')
        res3_esd = res3_esd.set_index('Term')
        res3_esd["Method"] = method
        return method, res3_esd
    
    elif method == 'grea_auc':
        obj = GREA(processes=1, perm_n=perm, seed=perm*i, symmetric=True, verbose=False)
        res3_auc = obj.fit(sig_name, sig_val, library, method='RC')
        res3_auc = res3_auc.set_index('Term')
        res3_auc["Method"] = method
        return method, res3_auc



def benchmark_parallel(signature, library, group, methods=['blitz', 'gseapy', 'grea_es', 'grea_esd', 'grea_auc'], rep_n=11, perm_list=[250,500,750,1000,1250,1500,1750,2000,2250,2500,],output_dir='result',):
    
    for i in range(1, rep_n):

        print(f"Outer loop iteration: {i}")
        method_results = {method: [] for method in methods}
        # 
        tasks = []
        for perm in perm_list:
            print(f"perm = {perm}")
            for method in methods:
                tasks.append((method, signature, library, group, i, perm))

        with multiprocessing.Pool(processes=len(methods)) as pool:
            results_list = pool.starmap(run_method, tasks)
        
        for method, res in results_list:
            method_results[method].append(res)

        for method in methods:
            if method_results[method]: 
                combined_results = pd.concat(method_results[method])
                combined_results.to_csv(os.path.join(output_dir, f"{method}_{i}.csv"))
    
        
    return None