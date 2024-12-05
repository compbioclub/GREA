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


import enrich


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
    fit_alpha, fit_loc, fit_beta = gamma.fit(nulls, floc=0)
    ks = kstest(nulls, 'gamma', args=(fit_alpha, fit_loc, fit_beta))[1]
    return fit_alpha, fit_alpha, ks

def pred_signgamma_prob(obs, nulls, symmetric=True, accuracy=40, deep_accuracy=50):
    probs = []
    for i in range(obs.shape[0]):
        nes, prob = pred_signgamma_prob_aux(obs[i], nulls[:, i], symmetric=symmetric, accuracy=accuracy, deep_accuracy=deep_accuracy)
        probs.append(prob)
    return np.array(probs)

def pred_signgamma_prob_aux(obs, nulls, symmetric=True, accuracy=40, deep_accuracy=50):
    alpha_pos, beta_pos, ks_pos, alpha_neg, beta_neg, ks_neg, pos_ratio = estimate_signgamma_paras(nulls, symmetric)

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
        prob = gamma.cdf(-obs, float(alpha_pos), scale=float(beta_pos))
        if prob > 0.999999999 or prob < 0.00000000001:
            mp.dps = deep_accuracy
            mp.prec = deep_accuracy
            prob = gammacdf(-obs, float(alpha_pos), float(beta_pos), dps=deep_accuracy)
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
        prob = pred_gamma_prob_aux(obs[i], nulls[:, i], accuracy=accuracy, deep_accuracy=deep_accuracy)
        probs.append(prob)
    return np.array(probs)

def pred_gamma_prob_aux(obs, nulls, accuracy=40, deep_accuracy=50):
    alpha_pos, beta_pos, ks_pos = estimate_gamma_paras(nulls)

    mp.dps = accuracy
    mp.prec = accuracy

    prob = gamma.cdf(obs, float(alpha_pos), scale=float(beta_pos))
    if prob > 0.999999999 or prob < 0.00000000001:
        mp.dps = deep_accuracy
        mp.prec = deep_accuracy
        prob = gammacdf(obs, float(alpha_pos), float(beta_pos), dps=deep_accuracy)

    mp.dps = accuracy
    mp.prec = accuracy
    return prob

def sig_enrich(sig_name, sig_val, library, 
               sig_sep=',', method='KS', n_perm=1000, prob_method='perm',
               seed: int=1, processes: int=4,
               verbose: bool=False, plot: bool=False,
               min_size: int=5, max_size: int=4000,
               accuracy: int=40, deep_accuracy: int=50, # ???
               ):

    n_sample = sig_val.shape[1]

    df_list = []
    for lib_key in tqdm(list(library.keys()), desc="Enrichment ", disable=not verbose):
        
        lib_sigs = library[lib_key]
        n_lib_sig = len(lib_sigs)

        overlap_ratios, n_hits = enrich.get_overlap(sig_name, lib_sigs, sig_sep)
        obs_rs, null_rs = enrich.get_running_sum(sig_val, overlap_ratios, method=method, n_perm=n_perm)
        if n_hits.shape[0] == 1: # for single sig_name
            n_hits = np.tile(np.array(n_hits), n_sample)
        res = pd.DataFrame({
            "Sample": range(n_sample),
            "Term": [lib_key]*n_sample,
            'n_gene': [n_lib_sig]*n_sample, 
            'n_hit': n_hits, 
        }) 
        
        if method == 'KS':
            KS_res = sig_enrich_KS(obs_rs, null_rs, prob_method=prob_method)
            res = pd.concat([res, KS_res], axis=1)
        if method == 'RC':
            RC_res = sig_enrich_RC(obs_rs, null_rs, prob_method=prob_method)
            res = pd.concat([res, RC_res], axis=1)      
        df_list.append(res)

    if not verbose:
        np.seterr(divide = 'ignore')
    
    df = pd.concat(df_list)
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

def sig_enrich_KS(obs_rs, null_rs, prob_method='signgamma'):
    n_sample = obs_rs.shape[1]
    es, esd, peak = enrich.get_ES_ESD(obs_rs)
    le_genes = []
    null_es, null_esd, null_peak = enrich.get_ES_ESD_null(null_rs)
    es_pval = pred_prob(es, null_es, prob_method=prob_method)
    esd_pval = pred_prob(esd, null_esd, prob_method=prob_method)
    nes, nesd = [0]*n_sample, [0]*n_sample

    le_ns = len(le_genes)
    le_gene_str = ','.join(le_genes)

    #if len(es_pval) > 1:  # may need to apply, need to rewrite
    #    es_fdr_values = multipletests(es_pval, method="fdr_bh")[1]
    #    es_sidak_values = multipletests(es_pval, method="sidak")[1]
    #    esd_fdr_values = multipletests(esd_pval, method="fdr_bh")[1]
    #    esd_sidak_values = multipletests(esd_pval, method="sidak")[1]
    #else:
    #    es_fdr_values = es_pval
    #    es_sidak_values = es_pval
    #    esd_fdr_values = esd_pval
    #    esd_sidak_values = esd_pval

    res = pd.DataFrame({
        "es": es, "nes": nes, "es_pval": es_pval, 
        #"es_sidak": es_sidak_values, "es_fdr": es_fdr_values,
        "esd": esd, "nesd": nesd, "esd_pval": esd_pval, 
        #"esd_sidak": esd_sidak_values, "esd_fdr": esd_fdr_values,
        "leading_edge_n": [le_ns]*n_sample, 
        "leading_edge": [le_gene_str]*n_sample,
        'prob_method': prob_method,
    })
    return res

def sig_enrich_RC(obs_rs, null_rs, prob_method='perm'):
    auc = enrich.get_AUC(obs_rs)
    null_auc = enrich.get_AUC_null(null_rs)
    auc_pval = pred_prob(auc, null_auc, prob_method=prob_method)

    #if len(auc_pval) > 1:  # may need to refine
    #    AUC_fdr_values = multipletests(auc_pval, method="fdr_bh")[1]
    #    AUC_sidak_values = multipletests(auc_pval, method="sidak")[1]
    #else:
    #    AUC_fdr_values = auc_pval
    #    AUC_sidak_values = auc_pval

    res = pd.DataFrame({
        "AUC": auc, "AUC_pval": auc_pval, 
        #"AUC_sidak": AUC_sidak_values, "AUC_fdr": AUC_fdr_values,
        'prob_method': prob_method,
    })
    return res