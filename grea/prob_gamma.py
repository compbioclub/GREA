import numpy as np
from scipy.stats import gamma
from scipy.stats import kstest
from mpmath import mp
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
        return np.nan, np.nan, np.nan  # return default nan value
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
    n_term, n_obs = obs.shape
    probs = np.zeros((n_term, n_obs))
    for t in range(n_term):
        for i in range(n_obs):
            nes, prob = pred_signgamma_prob_aux(obs[i], nulls[:, i], symmetric=symmetric, accuracy=accuracy, deep_accuracy=deep_accuracy)
            probs[t, i] = prob 
    return probs

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
    # obs: n_term, n_obs
    # nulls: n_perm, n_term, n_obs
    # return
    # probs: n_term, n_obs
    n_term, n_obs = obs.shape
    probs = np.zeros((n_term, n_obs))
    for t in range(n_term):
        for i in range(n_obs):
            current_nulls = nulls[:, t, i]
            # print(f"\nFor i={i}:")
            # print(f"Current nulls shape: {current_nulls.shape}")
            # print(f"Number of valid values: {np.sum((current_nulls > 0) & np.isfinite(current_nulls))}")
            prob = pred_gamma_prob_aux(obs[t, i], nulls[:, t, i], accuracy=accuracy, deep_accuracy=deep_accuracy)
            probs[t, i] = prob
    return probs

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





