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

from src import enrich


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

def get_peak_size(signature, abs_signature, sig2i, size, perm_n, seed): # ???
    es = []
    random.seed(seed)
    for _ in range(perm_n):
        rgenes = random.sample(list(signature.index), size)
        es.append(enrichment_score(abs_signature, sig2i, rgenes)[1])
    return es



def loess_interpolation(x, y, frac=0.5): # ???
    yl = np.array(y)
    #xout, yout, wout = loess_1d(x, yl, frac=frac)
    #return interpolate.interp1d(xout, yout)
    yout = lowess(y, x, frac=frac)[:, 1]
    return interpolate.interp1d(x, yout)


def estimate_anchor(abs_signature, hit_n, perm_n, symmetric, seed):
    es, esd = enrich.get_ESs_null(abs_signature, hit_n, perm_n, seed)
    es_gamma_paras = estimate_anchor_aux(es, symmetric)
    esd_gamma_paras = estimate_anchor_aux(esd, symmetric)
    return es_gamma_paras, esd_gamma_paras

def estimate_anchor_aux(es, symmetric):
    
    pos = es[es > 0]
    neg = es[es < 0]

    if (len(neg) < 250 or len(pos) < 250) and not symmetric:
        symmetric = True
    
    if symmetric:
        aes = np.abs(es)[es != 0]
        fit_alpha, fit_loc, fit_beta = gamma.fit(aes, floc=0)
        ks_pos = kstest(aes, 'gamma', args=(fit_alpha, fit_loc, fit_beta))[1]
        ks_neg = kstest(aes, 'gamma', args=(fit_alpha, fit_loc, fit_beta))[1]

        alpha_pos = fit_alpha
        beta_pos = fit_beta
        
        alpha_neg = fit_alpha
        beta_neg = fit_beta
    else:
        fit_alpha, fit_loc, fit_beta = gamma.fit(pos, floc=0)
        ks_pos = kstest(pos, 'gamma', args=(fit_alpha, fit_loc, fit_beta))[1]
        alpha_pos = fit_alpha
        beta_pos = fit_beta
        
        fit_alpha, fit_loc, fit_beta = gamma.fit(-np.array(neg), floc=0)
        ks_neg = kstest(-np.array(neg), 'gamma', args=(fit_alpha, fit_loc, fit_beta))[1]
        alpha_neg = fit_alpha
        beta_neg = fit_beta

    pos_ratio = len(pos)/(len(pos)+len(neg))
    return alpha_pos, beta_pos, ks_pos, alpha_neg, beta_neg, ks_neg, pos_ratio


def generate_hit_ns(signature, library, 
                         max_size=4000, 
                         calibration_size: int=20):
    # number of genes in each gene set
    ll = [len(library[l]) for l in library]
    nn = np.percentile(ll, q=np.linspace(2, 100, calibration_size))
    hit_ns = sorted(list(set(np.append([1,4,6,max_size, 
                                             np.min([max_size, np.max(ll)]), 
                                             np.min([max_size, int(signature.shape[0]/2)]), 
                                             np.min([max_size, signature.shape[0]-1])], 
                                             nn).astype("int"))))
    hit_ns = [int(x) for x in hit_ns if x < signature.shape[0]]
    return hit_ns

def fit_gammas(signature, abs_signature, library, 
               perm_n: int=1000, 
               max_size: int=4000, 
               symmetric: bool=False, 
               calibration_size: int=20, 
               plot: bool=True, 
               processes=4, 
               verbose=False, 
               progress=False, 
               seed: int=0):

    hit_ns = generate_hit_ns(signature, library, max_size=max_size, calibration_size=calibration_size)
    process_generator = (estimate_anchor(abs_signature, hit_n, perm_n, symmetric, seed+hit_n) for hit_n in hit_ns)
    results = list(tqdm(process_generator, desc="Calibration", total=len(hit_ns), disable=not progress))

    es_results = []
    esd_results = []
    for res in results:
        es_results.append(res[0])
        esd_results.append(res[1])

    es_gamma_paras = fit_gammas_aux(es_results, hit_ns, key='ES', verbose=verbose, plot=plot)
    esd_gamma_paras = fit_gammas_aux(esd_results, hit_ns, key='ESD', verbose=verbose, plot=plot)
    return es_gamma_paras, esd_gamma_paras

def fit_gammas_aux(results, hit_ns, key='ES', verbose=False, plot=False):
    alpha_pos = []
    beta_pos = []
    ks_pos = []
    alpha_neg = []
    beta_neg = []
    ks_neg = []
    pos_ratio = []

    for res in results:
        f_alpha_pos, f_beta_pos, f_ks_pos, f_alpha_neg, f_beta_neg, f_ks_neg, f_pos_ratio = res
        alpha_pos.append(f_alpha_pos)
        beta_pos.append(f_beta_pos)
        ks_pos.append(f_ks_pos)
        alpha_neg.append(f_alpha_neg)
        beta_neg.append(f_beta_neg)
        ks_neg.append(f_ks_neg)
        pos_ratio.append(f_pos_ratio)

    if np.max(pos_ratio) > 1.5 and verbose:
        print(key, 'Significant unbalance between positive and negative enrichment scores detected. Signature values are not centered close to 0.')


    hit_ns = np.array(hit_ns, dtype=float)
    
    f_alpha_pos = loess_interpolation(hit_ns, alpha_pos)
    f_beta_pos = loess_interpolation(hit_ns, beta_pos, frac=0.2)
    
    f_alpha_neg = loess_interpolation(hit_ns, alpha_neg)
    f_beta_neg = loess_interpolation(hit_ns, beta_neg, frac=0.2)

    # fix issue with numeric instability
    pos_ratio = pos_ratio - np.abs(0.0001*np.random.randn(len(pos_ratio)))
    f_pos_ratio = loess_interpolation(hit_ns, pos_ratio)
    
    if plot:
        xx = np.linspace(min(hit_ns), max(hit_ns), 1000)
        
        
        yy = f_alpha_pos(xx)
        plt.plot(xx, yy, '--', lw=3)
        plt.plot(hit_ns, alpha_pos, 'ko')
        plt.title(key + ' alpha_pos')
        plt.show()

        yy = f_alpha_neg(xx)
        plt.plot(xx, yy, '--', lw=3, c="orange")
        plt.plot(hit_ns, alpha_neg, 'o', c="coral")
        plt.title(key + ' alpha_neg')
        plt.show()

        yy = f_beta_pos(xx)
        plt.plot(xx, yy, '--', lw=3)
        plt.plot(hit_ns, beta_pos, 'ko')
        plt.title(key + ' beta_neg')
        plt.show()

        yy = f_beta_neg(xx)
        plt.plot(xx, yy, '--', lw=3, c="orange")
        plt.plot(hit_ns, beta_neg, 'o', c="coral")
        plt.title(key + ' beta_neg')
        plt.show()

        yy = f_pos_ratio(xx)
        plt.figure(2)
        plt.plot(xx, yy, lw=3)
        plt.plot(hit_ns, pos_ratio, 'o', c="black")
        plt.title(key + ' pos_ratio')
        plt.show()
    
    ks_pos, ks_neg = np.mean(ks_pos), np.mean(ks_neg)
    if (ks_pos < 0.05 or ks_neg < 0.05) and verbose:
        print(key, 'Kolmogorov-Smirnov test failed. Gamma approximation deviates from permutation samples.\n'+"KS p-value (pos): "+str(ks_pos)+"\nKS p-value (neg): "+str(ks_neg))
    return f_alpha_pos, f_beta_pos, f_pos_ratio, ks_pos, ks_neg

def pred_gamma_prob(es, hit_n, Gamma_paras,
                    accuracy: int=40, deep_accuracy: int=50,
                    ):
    f_alpha_pos, f_beta_pos, f_pos_ratio, ks_pos, ks_neg = Gamma_paras
    pos_alpha = f_alpha_pos(hit_n)
    pos_beta = f_beta_pos(hit_n)
    pos_ratio = max(0, min(1.0, f_pos_ratio(hit_n)))

    mp.dps = accuracy
    mp.prec = accuracy

    if es > 0:
        prob = gamma.cdf(es, float(pos_alpha), scale=float(pos_beta))
        if prob > 0.999999999 or prob < 0.00000000001:
            mp.dps = deep_accuracy
            mp.prec = deep_accuracy
            prob = gammacdf(es, float(pos_alpha), float(pos_beta), dps=deep_accuracy)
        prob_two_tailed = np.min([0.5,(1-np.min([prob*pos_ratio+1-pos_ratio,1]))])
        nes = invcdf(1-np.min([1,prob_two_tailed]))
        pval = 2*prob_two_tailed
    else:
        prob = gamma.cdf(-es, float(pos_alpha), scale=float(pos_beta))
        if prob > 0.999999999 or prob < 0.00000000001:
            mp.dps = deep_accuracy
            mp.prec = deep_accuracy
            prob = gammacdf(-es, float(pos_alpha), float(pos_beta), dps=deep_accuracy)
        prob_two_tailed = np.min([0.5,(1-np.min([(((prob)-(prob*pos_ratio))+pos_ratio),1]))])
        if prob_two_tailed == 0.5:
            prob_two_tailed = prob_two_tailed-prob

        nes = invcdf(np.min([1,prob_two_tailed])) 
        pval = 2*prob_two_tailed
            
    mp.dps = accuracy
    mp.prec = accuracy
    return nes, pval

def sig_enrich(signature, abs_signature, sig2i, i2sig, library, 
               seed: int=1, processes: int=4,
               verbose: bool=False, plot: bool=False,
               min_size: int=5, max_size: int=4000,
               accuracy: int=40, deep_accuracy: int=50, # ???
               ):

    es_gamma_paras, esd_gamma_paras = fit_gammas(signature, abs_signature, library, 
                                seed=seed, plot=plot, processes=processes)
    sig_genes = set(signature.index)

    keys, hit_ns, gene_ns = [], [], []
    es_list, esd_list = [], []
    es_pval_list, esd_pval_list = [], []
    nes_list, nesd_list = [], []
    le_ns, le_gene_list = [], []
   
    for k in tqdm(list(library.keys()), desc="Enrichment ", disable=not verbose):
        genes = library[k]
        hit_genes = [x for x in genes if x in sig_genes]
        hit_n = len(hit_genes)
        gene_n = len(genes)
        if hit_n >= min_size and hit_n <= max_size:
            keys.append(k)
            hit_ns.append(hit_n)
            gene_ns.append(gene_n)
            rs, es, esd, peak, le_genes = enrich.get_SE(abs_signature, sig2i, i2sig, genes)
            nes, es_pval = pred_gamma_prob(es, hit_n, es_gamma_paras, accuracy, deep_accuracy)
            nesd, esd_pval = pred_gamma_prob(esd, hit_n, esd_gamma_paras, accuracy, deep_accuracy)

            es_list.append(float(es))
            esd_list.append(float(esd))
            nes_list.append(-float(nes))
            nesd_list.append(-float(nesd))
            es_pval_list.append(float(es_pval))
            esd_pval_list.append(float(esd_pval))
            le_ns.append(len(le_genes))
            le_gene_list.append(','.join(le_genes))

    if not verbose:
        np.seterr(divide = 'ignore')
    
    if len(es_pval_list) > 1:
        es_fdr_values = multipletests(es_pval_list, method="fdr_bh")[1]
        es_sidak_values = multipletests(es_pval_list, method="sidak")[1]
        esd_fdr_values = multipletests(esd_pval_list, method="fdr_bh")[1]
        esd_sidak_values = multipletests(esd_pval_list, method="sidak")[1]
    else:
        es_fdr_values = es_pval_list
        es_sidak_values = es_pval_list
        esd_fdr_values = esd_pval_list
        esd_sidak_values = esd_pval_list


    res = pd.DataFrame([
        keys, 
        np.array(es_list), np.array(nes_list), np.array(es_pval_list), np.array(es_sidak_values), np.array(es_fdr_values),
        np.array(esd_list), np.array(nesd_list), np.array(esd_pval_list), np.array(esd_sidak_values), np.array(esd_fdr_values), 
        np.array(gene_ns), np.array(hit_ns), np.array(le_ns), np.array(le_gene_list)
        ]).T
    stats = ["es", "nes", "es_pval", "es_sidak", "es_fdr",
             "esd", "nesd", "esd_pval", "esd_sidak", "esd_fdr"]
    res.columns = ["Term"] + stats + \
        ["gene_n", "hit_n", "leading_edge_n", "leading_edge"]
    res["Term"] = res['Term'].astype("str")
    for col in stats:
        res[col] = res[col].astype("float")
    res["gene_n"] = res['gene_n'].astype("int")
    res["hit_n"] = res['hit_n'].astype("int")
    res["leading_edge_n"] = res['leading_edge_n'].astype("int")
    res["leading_edge"] = res['leading_edge'].astype("str")
    res = res.set_index("Term")

    return res.sort_values("es_pval", key=abs, ascending=True)