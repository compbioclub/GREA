from statsmodels.nonparametric.smoothers_lowess import lowess
from collections import Counter
from scipy import interpolate
import pandas as pd
import numpy as np
import random
from scipy.stats import gamma
from scipy.stats import kstest
import multiprocessing
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpmath import mp, mpf
from statsmodels.stats.multitest import multipletests
import seaborn as sns
from multiprocessing import Pool



import grea.enrich as enrich

from mpmath import mp, exp, log
import math
from scipy.stats import norm, gamma

def gammacdf(x, k, theta, dps=100):
    """
    Gamma distribution cumulative distribution function.
    k is the shape parameter
    theta is the scale parameter (reciprocal of the rate parameter)
    Unlike scipy, a location parameter is not included.
    """
    mp.dps = dps
    with mp.extradps(mp.dps):
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        return mp.gammainc(k, 0, x/theta, regularized=True)

def invcdf(p, mu=0, sigma=1):
    orig_p = p
    if p > 0.5:
        p = 1- p
    p = float(p)
    if math.isnan(p):
        p = 1
    p = min(max(p, 0), 1)
    n = norm.isf(p)
    if orig_p > 0.5:
        return -n
    return n

def estimate_signgamma_paras(es, symmetric):
    pos = es[es > 0]
    neg = es[es < 0]

    if (len(neg) < 250 or len(pos) < 250) and not symmetric:
        symmetric = True
    
    if symmetric:
        aes = np.abs(es)[es != 0]
        aes = aes[~np.isnan(aes) & ~np.isinf(aes)]
        fit_alpha, fit_beta, ks = estimate_gamma_paras(aes)
        alpha_pos = fit_alpha
        beta_pos = fit_beta
        ks_pos = ks
        alpha_neg = fit_alpha
        beta_neg = fit_beta
        ks_neg = ks
    else:
        alpha_pos, beta_pos, ks_pos = estimate_gamma_paras(pos)
        alpha_neg, beta_neg, ks_neg = estimate_gamma_paras(-np.array(neg))

    pos_ratio = len(pos)/(len(pos)+len(neg))
    return alpha_pos, beta_pos, ks_pos, alpha_neg, beta_neg, ks_neg, pos_ratio

def estimate_gamma_paras(nulls):
    nulls = np.array(nulls)

    nulls = nulls[np.isfinite(nulls)] 
    nulls = nulls[nulls > 0] 

    if len(nulls) < 2:
        print("Warning: Not enough valid data to fit gamma distribution. Using default parameters.")
        return np.nan, np.nan, np.nan  # 直接返回默认值，而不是抛出异常
    try:
        fit_alpha, fit_loc, fit_beta = gamma.fit(nulls, floc=0)
        if not (np.isfinite(fit_alpha) and np.isfinite(fit_beta)):
            raise print(
                "Gamma distribution fitting failed: invalid parameters. "
                "Please consider using permutation method instead."
            )
        ks = kstest(nulls, 'gamma', args=(fit_alpha, fit_loc, fit_beta))[1]
        return fit_alpha, fit_beta, ks
    except Exception as e:
        print(f"Warning: Failed to fit gamma distribution: {str(e)}. Using default parameters.")
        print(f"Data summary - mean: {np.mean(nulls):.3f}, "
              f"var: {np.var(nulls):.3f}, "
              f"range: [{np.min(nulls):.3f}, {np.max(nulls):.3f}], "
              f"size: {len(nulls)}")
        return np.nan,np.nan,np.nan



def pred_signgamma_prob(obs, nulls, symmetric=True, accuracy=40, deep_accuracy=50):
    probs = []
    for i in range(obs.shape[0]):
        nes, prob = pred_signgamma_prob_aux(obs[i], nulls[:, i], symmetric=symmetric, accuracy=accuracy, deep_accuracy=deep_accuracy)
        probs.append(prob)
    return np.array(probs)

def pred_signgamma_prob_aux(obs, nulls, symmetric=True, accuracy=40, deep_accuracy=50):
    
    # invalid_indices = np.where(np.isnan(nulls) | np.isinf(nulls))
    # print("Indices of invalid (NaN or Inf) values:", invalid_indices)
    # nan_indices = np.where(np.isnan(nulls))
    # print("Indices of NaN values:", nan_indices)
    
    # inf_indices = np.where(np.isinf(nulls))
    # print("Indices of Infinite values:", inf_indices)
    
    alpha_pos, beta_pos, ks_pos, alpha_neg, beta_neg, ks_neg, pos_ratio = estimate_signgamma_paras(nulls, symmetric)

    if np.isnan(alpha_pos) or np.isnan(beta_pos) or np.isnan(ks_pos):
        return np.nan, np.nan

    mp.dps = accuracy
    mp.prec = accuracy

    
    if obs > 0:
        prob = gamma.cdf(obs, float(alpha_pos), scale=float(beta_pos))
        if prob > 0.999999999 or prob < 0.00000000001:
            mp.dps = deep_accuracy
            mp.prec = deep_accuracy
            prob = gammacdf(obs, float(alpha_pos), float(beta_pos), dps=deep_accuracy)
        prob_two_tailed = np.min([0.5,(1-np.min([prob*pos_ratio+1-pos_ratio,1]))])
        nes = invcdf(1-np.min([1,prob_two_tailed]))
        pval = 2*prob_two_tailed
    else:
        prob = gamma.cdf(-obs, float(alpha_neg), scale=float(beta_neg))
        if prob > 0.999999999 or prob < 0.00000000001:
            mp.dps = deep_accuracy
            mp.prec = deep_accuracy
            prob = gammacdf(-obs, float(alpha_neg), float(beta_neg), dps=deep_accuracy)
        prob_two_tailed = np.min([0.5,(1-np.min([(((prob)-(prob*pos_ratio))+pos_ratio),1]))])
        if prob_two_tailed == 0.5:
            prob_two_tailed = prob_two_tailed-prob

        nes = invcdf(np.min([1,prob_two_tailed])) 
        pval = 2*prob_two_tailed
            
    mp.dps = accuracy
    mp.prec = accuracy
    return nes, pval

def pred_gamma_prob(obs, nulls, accuracy=40, deep_accuracy=50):

    probs = []
    for i in range(obs.shape[0]):
        current_nulls = nulls[:, i]
        print(f"\nFor i={i}:")
        print(f"Current nulls shape: {current_nulls.shape}")
        print(f"Number of valid values: {np.sum((current_nulls > 0) & np.isfinite(current_nulls))}")
        prob = pred_gamma_prob_aux(obs[i], nulls[:, i], accuracy=accuracy, deep_accuracy=deep_accuracy)
        probs.append(prob)
    return np.array(probs)

def pred_gamma_prob_aux(obs, nulls, accuracy=40, deep_accuracy=50):
    alpha_pos, beta_pos, ks_pos = estimate_gamma_paras(nulls)

    if np.isnan(alpha_pos) or np.isnan(beta_pos) or np.isnan(ks_pos):
        return np.nan

    mp.dps = accuracy
    mp.prec = accuracy
 
    prob = gamma.cdf(obs, float(alpha_pos), scale=float(beta_pos))

    if not isinstance(prob, (int, float, np.number)):
        print("gamma.cdf returned invalid type:", type(prob), prob)
        return np.nan


    if prob > 0.999999999 or prob < 0.00000000001:
        mp.dps = deep_accuracy
        mp.prec = deep_accuracy
        prob = gammacdf(obs, float(alpha_pos), float(beta_pos), dps=deep_accuracy)

        if not isinstance(prob, (int, float, np.number)):
            print("gamma.cdf returned invalid type:", type(prob), prob)
            return np.nan
        
    return 1-prob


def process_library_key(lib_key, library, sig_name, sig_val, sig_sep, method, prob_method, cal_method, n_perm, save_permutation, verbose):
    
    n_sample = sig_val.shape[1]
    
    lib_sigs = library[lib_key]
    n_lib_sig = len(lib_sigs)

    overlap_ratios, n_hits = enrich.get_overlap(sig_name, lib_sigs, sig_sep)
    obs_rs, sort_indices = enrich.get_running_sum(sig_val, overlap_ratios, method=method)

    if n_hits.shape[0] == 1:  
        n_hits = np.tile(np.array(n_hits), n_sample)

    res = pd.DataFrame({
        "Sample": range(n_sample),
        "Term": [lib_key] * n_sample,
        "n_gene": [n_lib_sig] * n_sample,
        "n_hit": n_hits,
    })

    i2sig = np.take_along_axis(sig_name, sort_indices, axis=0)
    i2overlap_ratios = np.take_along_axis(overlap_ratios, sort_indices, axis=0)
    sorted_abs = np.abs(np.take_along_axis(sig_val, sort_indices, axis=0))

    if method == 'KS' and cal_method == 'ES':
        KS_res = sig_enrich_KS(obs_rs, sorted_abs, i2sig, i2overlap_ratios, sort_indices,
                               prob_method=prob_method, cal_method=cal_method,
                               n_perm=n_perm, save_permutation=save_permutation)
        res = pd.concat([res, KS_res], axis=1)
        res = res.sort_values("es_pval", key=abs, ascending=True)

    elif method == 'KS' and cal_method == 'ESD':
        KS_res = sig_enrich_KS(obs_rs, sorted_abs, i2sig, i2overlap_ratios, sort_indices,
                               prob_method=prob_method, cal_method=cal_method,
                               n_perm=n_perm, save_permutation=save_permutation)
        res = pd.concat([res, KS_res], axis=1)
        res = res.sort_values("esd_pval", key=abs, ascending=True)

    if method == 'RC':
        RC_res = sig_enrich_RC(obs_rs, sorted_abs, i2overlap_ratios, sort_indices,
                               prob_method=prob_method, n_perm=n_perm, save_permutation=save_permutation)
        res = pd.concat([res, RC_res], axis=1)
        res = res.sort_values("AUC_pval", key=abs, ascending=True)

    if method == 'PLAGE':
        obs_plage = enrich.get_plage(sig_val)
        null_plage = enrich.get_plage_null(sig_val, overlap_ratios, n_perm=n_perm)

    return res

def sig_enrich(sig_name, sig_val, library, 
               sig_sep=',', method='KS', n_perm=1000, prob_method='perm',
               seed: int=1, processes: int=4,
               verbose: bool=False, plot: bool=False,
               min_size: int=5, max_size: int=4000,
               accuracy: int=40, deep_accuracy: int=50, 
               cal_method: str='ES', # ???
               save_permutation: bool=False, 
               ):
    
    args_list = [(lib_key, library, sig_name, sig_val, sig_sep, method, prob_method, cal_method, n_perm, save_permutation, verbose) 
             for lib_key in library.keys()]
    
    if processes == 1:
        df_list = []
        for args in args_list:
            df_list.append(process_library_key(*args))
    else:
        df_list = []
        with Pool(processes=processes) as pool:
            results_list = (pool.starmap(process_library_key, args_list))

        df_list.extend(results_list)
        
    if not verbose:
        np.seterr(divide = 'ignore')
    
    df = pd.concat(df_list)
    if method == 'RC':
        df = df.sort_values("AUC_pval", key=abs, ascending=True)
    elif method == 'KS':
        if cal_method == 'ES':
            df = df.sort_values("es_pval", key=abs, ascending=True)
        elif cal_method == 'ESD':
            df = df.sort_values("esd_pval", key=abs, ascending=True)

    return df


def pred_perm_prob(obs, nulls): 
    """
    Computes probabilities for each observed value by comparing with its corresponding column in the null distribution.

    Parameters:
    obs (np.ndarray): Observed values, shape (n,).
    nulls (np.ndarray): Null distribution, shape (m, n).
    
    Returns:
    np.ndarray: Probabilities for each observed value, shape (n,).
    """
    return (obs <= nulls).sum(axis=0)/nulls.shape[0]         


def pred_signperm_prob(obs, nulls):
    """
    Computes the probability of the observed value based on the sign and the corresponding column in the null distribution.

    Parameters:
    obs (np.ndarray): Observed values, shape (n,).
    nulls (np.ndarray): Null distribution, shape (m, n).
    
    Returns:
    np.ndarray: Probabilities for each observed value, shape (n,).
    """
    obs = np.asarray(obs)
    nulls = np.asarray(nulls)

    pos_mask = obs > 0
    neg_mask = obs < 0

    probs = np.zeros_like(obs, dtype=float)
    if np.any(pos_mask):
        pos_obs = obs[pos_mask]
        pos_nulls = nulls[:, pos_mask]
        probs[pos_mask] = (
            np.sum(pos_obs <= pos_nulls, axis=0) / nulls.shape[0]
        )
    if np.any(neg_mask):
        neg_obs = obs[neg_mask]
        neg_nulls = nulls[:, neg_mask]
        probs[neg_mask] = (
            np.sum(neg_obs >= neg_nulls, axis=0) / nulls.shape[0]
        )
    return probs

def pred_prob(obs, nulls, prob_method):
    if prob_method == 'perm':
        return pred_perm_prob(obs, nulls)
    if prob_method == 'signperm':
        return pred_signperm_prob(obs, nulls)
    if prob_method == 'signgamma':
        return pred_signgamma_prob(obs, nulls)
    if prob_method == 'gamma':
        return pred_gamma_prob(obs, nulls)

def sig_enrich_KS(obs_rs, sorted_abs, i2sig, i2overlap_ratios,sort_indices,prob_method='signgamma', cal_method='ES', n_perm=1000,save_permutation=False):
    n_sample = obs_rs.shape[1]
    es, esd, peak = enrich.get_ES_ESD(obs_rs)
    le_genes = []
    null_es, null_esd, null_peak = enrich.get_ES_ESD_null(sorted_abs,i2overlap_ratios,sort_indices, n_perm=n_perm,save_permutation=save_permutation)
    es_pval = pred_prob(es, null_es, prob_method=prob_method)
    esd_pval = pred_prob(esd, null_esd, prob_method=prob_method)
    nes, nesd = [0]*n_sample, [0]*n_sample
    
    le_genes_list = enrich.get_leading_edge(i2sig, i2overlap_ratios, es, peak)

    if prob_method == 'signperm':
        es_null_mean = np.mean(null_es, axis=0)
        es_null_mean[es_null_mean == 0] = np.finfo(float).eps
        esd_null_mean = np.mean(null_esd, axis=0)
        esd_null_mean[esd_null_mean == 0] = np.finfo(float).eps
        nes = es/es_null_mean
        nesd = esd/esd_null_mean
    elif prob_method == 'signgamma':
        # alpha_pos, beta_pos, ks_pos, alpha_neg, beta_neg, ks_neg, pos_ratio = estimate_signgamma_paras(null_es, symmetric=True)
        for i in range(n_sample):
            nes[i], _ = pred_signgamma_prob_aux(es[i], null_es[:, i], symmetric=True, accuracy=40, deep_accuracy=50)
            nesd[i], _ = pred_signgamma_prob_aux(esd[i], null_esd[:, i], symmetric=True, accuracy=40, deep_accuracy=50)

    le_ns = [len(le) for le in le_genes_list]
    le_gene_str = [','.join(le) for le in le_genes_list]


    if len(es_pval) > 1:  # may need to apply, need to rewrite
       es_fdr_values = multipletests(es_pval, method="fdr_bh")[1]
       es_sidak_values = multipletests(es_pval, method="sidak")[1]
       esd_fdr_values = multipletests(esd_pval, method="fdr_bh")[1]
       esd_sidak_values = multipletests(esd_pval, method="sidak")[1]
    else:
       es_fdr_values = es_pval
       es_sidak_values = es_pval
       esd_fdr_values = esd_pval
       esd_sidak_values = esd_pval
    if cal_method == 'ES':
        res = pd.DataFrame({
            "es": es,
            "nes": nes,
            "es_pval": es_pval,
            "fdr": es_fdr_values,
            "sidak": es_sidak_values,
            "leading_edge_n":  le_ns,
            "leading_edge": le_gene_str,
            "prob_method": prob_method,
        })
        
    elif cal_method == 'ESD':
        # Return results based on 'ESD'
        res = pd.DataFrame({
            "esd": esd,
            "nesd": nesd,
            "esd_pval": esd_pval,
            "fdr": esd_fdr_values,
            "sidak": esd_sidak_values,
            "leading_edge_n": le_ns,
            "leading_edge": le_gene_str,
            "prob_method": prob_method,
        })
    else:
        raise ValueError("Invalid cal_method. Choose either 'ES' or 'ESD'.")
    
    res = res.dropna(subset=["es"] if cal_method == "ES" else ["esd"])
    return  res

def sig_enrich_RC(obs_rs, sorted_abs, i2overlap_ratios,sort_indices, prob_method='perm',n_perm=1000,save_permutation=False):

    n_sample = obs_rs.shape[1]

    auc = enrich.get_AUC(obs_rs)
    null_auc = enrich.get_AUC_null(sorted_abs,i2overlap_ratios,sort_indices, n_perm=n_perm,save_permutation=save_permutation)
    auc_pval = pred_prob(auc, null_auc, prob_method=prob_method)

    nauc =  [0]*n_sample

    if prob_method == 'perm':
        auc_null_mean = np.mean(null_auc, axis=0)
        auc_null_mean[auc_null_mean == 0] = np.finfo(float).eps
        nauc = auc/auc_null_mean
    elif prob_method == 'gamma':
        for i in range(n_sample):
            prob = pred_gamma_prob_aux(auc[i], null_auc[:, i], accuracy=40, deep_accuracy=50)
            if np.isnan(prob):
                nauc[i] = np.nan
            else:
                nauc[i] = invcdf(prob)

    if len(auc_pval) > 1:  # may need to apply, need to rewrite
       auc_fdr_values = multipletests(auc_pval, method="fdr_bh")[1]
       auc_sidak_values = multipletests(auc_pval, method="sidak")[1]
    else:
       auc_fdr_values = auc_pval
       auc_sidak_values = auc_pval

    res = pd.DataFrame({
        "AUC": auc, 
        "NAUC": nauc,
        "AUC_pval": auc_pval, 
        'prob_method': prob_method,
        "fdr": auc_fdr_values,
        "sidak": auc_sidak_values
    })
    res = res.dropna(subset=["AUC"])
    return res
