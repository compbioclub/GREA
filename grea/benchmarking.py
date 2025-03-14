import blitzgsea as blitz
import pandas as pd
import numpy as np
from grea import GREA
import multiprocessing
import gseapy
import os
from grea.dataprocess import preprocess_signature



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

def run_method(method, signature, library, i, perm,):
    signature.columns = ["i","v"]
    sig_name = signature.values[:, 0][:, np.newaxis]
    sig_val = signature.values[:, 1][:, np.newaxis]
    if method == 'blitz':
        res1 = blitz.gsea(signature, library, permutations=perm, processes=1, seed=perm*i, signature_cache=False)
        res1["Method"] = method 
        res1['Perm_num'] = perm
        return method, res1
    
    elif method == 'gseapy':
        sig = signature.sort_values('v', ascending=False)
        sig = sig[~sig.index.duplicated(keep='first')]
        res2 = chopped_gsea(sig, library, processes=1, permutation_num=perm, seed=perm*i, max_lib_size=25)
        res2 = res2.set_index('Term')
        res2["Method"] = method 
        res2['Perm_num'] = perm
        return method, res2
    
    elif method == 'grea_KS_signperm_es':
        obj = GREA(processes=1, perm_n=perm, seed=perm*i, symmetric=True, verbose=False)
        res3_es_1 = obj.fit(sig_name, sig_val, library, prob_method='signperm')
        res3_es_1 = res3_es_1.set_index('Term')
        res3_es_1["Method"] = method 
        res3_es_1['Perm_num'] = perm
        return method, res3_es_1
    
    elif method == 'grea_KS_signgamma_es':
        obj = GREA(processes=1, perm_n=perm, seed=perm*i, symmetric=True, verbose=False)
        res3_es_2 = obj.fit(sig_name, sig_val, library, prob_method='signgamma')
        res3_es_2 = res3_es_2.set_index('Term')
        res3_es_2["Method"] = method
        res3_es_2['Perm_num'] = perm
        return method, res3_es_2
    
    elif method == 'grea_KS_signperm_esd':
        obj = GREA(processes=1, perm_n=perm, seed=perm*i, symmetric=True, verbose=False)
        res3_esd_1 = obj.fit(sig_name, sig_val, library, cal_method='ESD', prob_method='signperm')
        res3_esd_1 = res3_esd_1.set_index('Term')
        res3_esd_1["Method"] = method
        res3_esd_1['Perm_num'] = perm
        return method, res3_esd_1
    
    elif method == 'grea_KS_signgamma_esd':
        obj = GREA(processes=1, perm_n=perm, seed=perm*i, symmetric=True, verbose=False)
        res3_esd_2 = obj.fit(sig_name, sig_val, library, cal_method='ESD', prob_method='signgamma')
        res3_esd_2 = res3_esd_2.set_index('Term')
        res3_esd_2["Method"] = method
        res3_esd_2['Perm_num'] = perm
        return method, res3_esd_2
    
    elif method == 'grea_RC_perm_AUC':
        obj = GREA(processes=1, perm_n=perm, seed=perm*i, symmetric=True, verbose=False)
        res3_auc_1 = obj.fit(sig_name, sig_val, library, method='RC')
        res3_auc_1 = res3_auc_1.set_index('Term')
        res3_auc_1["Method"] = method
        res3_auc_1['Perm_num'] = perm
        return method, res3_auc_1
    
    elif method == 'grea_RC_gamma_auc':
        obj = GREA(processes=1, perm_n=perm, seed=perm*i, symmetric=True, verbose=False)
        res3_auc_2 = obj.fit(sig_name, sig_val, library, method='RC',prob_method='gamma')
        res3_auc_2 = res3_auc_2.set_index('Term')
        res3_auc_2["Method"] = method
        res3_auc_2['Perm_num'] = perm
        return method, res3_auc_2
    



def benchmark_parallel(signature, library, methods=['blitz', 'gseapy', 'grea_KS_signperm_es', 'grea_KS_signgamma_es', 'grea_KS_signperm_esd', 'grea_KS_signgamma_esd', 'grea_RC_perm_AUC', 'grea_RC_gamma_auc'], rep_n=11, perm_list=[250,500,750,1000,1250,1500,1750,2000,2250,2500,],output_dir='result',):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(1, rep_n+1):

        print(f"Outer loop iteration: {i+1}")
        method_results = {method: [] for method in methods}
        # 
        tasks = []
        for perm in perm_list:
            print(f"perm = {perm}")
            for method in methods:
                tasks.append((method, signature, library, i, perm))

        with multiprocessing.Pool(processes=len(methods)) as pool:
            results_list = pool.starmap(run_method, tasks)
        
        for method, res in results_list:
            method_results[method].append(res)

        for method in methods:
            if method_results[method]: 
                combined_results = pd.concat(method_results[method])
                combined_results.to_csv(os.path.join(output_dir, f"{method}_{i}.csv"))
    
        
    return None