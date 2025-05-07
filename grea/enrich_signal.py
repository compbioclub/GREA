import os
import numpy as np


def get_leading_edge(obj):
    """
    Get leading-edge sig names for each (n_term, n_obs) enrichment signal.

    Args:
        obj.sorted_sig_names: np.ndarray of shape [n_term, n_sig, n_obs] — names of sorted signatures [interactions or genes].
        obj.sorted_or: np.ndarray of shape [n_term, n_sig, n_obs] — overlap ratios.
        obj.ES: np.ndarray of shape [n_term, n_obs] — enrichment scores.
        obj.peak: np.ndarray of shape [n_term, n_obs] — index of ES peaks.
        obj.sig_names: [n_sig, n_obs]
        obj.sort_indices: [n_sig, n_obs]
    Returns:
        obj.lead_sigs: List[List[List[str]]] of shape [n_term][n_obs][#sigs] — names of leading-edge sigs.
    """
    n_term, _, n_obs = obj.sorted_or.shape
    obj.sorted_sig_names = np.take_along_axis(
        np.broadcast_to(obj.sig_names[None, :, :], (n_term, *obj.sig_names.shape)),
        obj.sort_indices[None, :, :],
        axis=1
    )
    # nested list comprehensions, which are faster than for loops in Python
    obj.lead_sigs = [
        [
            [
                str(obj.sorted_sig_names[t, j, i]) 
                for j in (np.where(obj.sorted_or[t, :obj.peak[t, i]+1, i] > 0)[0]
                          if obj.ES[t, i] > 0 else
                          np.where(obj.sorted_or[t, obj.peak[t, i]:, i] > 0)[0] + obj.peak[t, i])
            ]
            for i in range(n_obs)
        ]
        for t in range(n_term)
    ]


def get_overlap(obj):
    """    
    Args:
        obj.sig_names: np.ndarray of shape [n_sig, n_obs], strings of sig_name separated by sig_sep
        obj.term_genes_list: List[List[str]], length = n_term, each is a set of genes
        obj.sig_sep: separator used in sig_name
        
    Returns:
        obj.overlap_ratios: np.ndarray of shape [n_term, n_sig, n_obs]
        obj.n_hits: np.ndarray of shape [n_term, n_obs] # number of sigs hit in the obs
    """
    n_sig, n_obs = obj.sig_names.shape
    n_term = len(obj.term_genes_list)

    obj.overlap_ratios = np.zeros((n_term, n_sig, n_obs), dtype=np.float32)
    obj.n_hits = np.zeros((n_term, n_obs), dtype=np.int32)

    overlap_ratio_dict = {}
    for t, lib_genes in enumerate(obj.term_genes_list):
        lib_genes = set(lib_genes)
        for j in range(n_obs):
            for i in range(n_sig):
                sig_name = obj.sig_names[i, j]
                if (t, sig_name) in overlap_ratio_dict:
                    ratio = overlap_ratio_dict[(t, sig_name)]
                else:
                    genes_in_cell = set(sig_name.split(obj.sig_sep))
                    n_overlap = len(genes_in_cell & lib_genes)
                    
                    if len(genes_in_cell) > 0:
                        ratio = n_overlap / len(genes_in_cell)
                        overlap_ratio_dict[(t, sig_name)] = ratio
                if ratio > 0:
                    obj.overlap_ratios[t, i, j] = ratio
                    obj.n_hits[t, j] += 1


def get_sorted(obj):
    """
    Args:
        obj.sig_vals: np.ndarray [n_sig, n_obs]
        obj.sig_names: np.ndarray [n_sig, n_obs]
    Returns:
        obj.sort_indices: np.ndarray [n_sig, n_obs]
        obj.sorted_sig_vals: np.ndarray [n_sig, n_obs]
        obj.sorted_abs_vals: np.ndarray [n_sig, n_obs]
    """
    obj.sort_indices = np.argsort(obj.sig_vals, axis=0)[::-1, :]  
    obj.sorted_sig_vals = np.take_along_axis(obj.sig_vals, obj.sort_indices, axis=0)  
    obj.sorted_abs_vals = np.abs(obj.sorted_sig_vals)


def get_running_sum(obj):
    """
    Fully vectorized version of running sum.
    Input:
        obj.sorted_abs_vals: [n_sig, n_obs]
        obj.overlap_ratios: [n_term, n_sig, n_obs]
        obj.sort_indices: [n_sig, n_obs]
    Output:
        obj.ks_rs: [n_term, n_sig, n_obs]
        obj.rc_rs: [n_term, n_sig, n_obs]
        obj.sorted_or: [n_term, n_sig, n_obs]
    """
    obj.ks_rs, obj.rc_rs, obj.sorted_or = _get_running_sum(obj.sorted_abs_vals, obj.overlap_ratios, obj.sort_indices)


def _get_running_sum(sorted_abs_vals, overlap_ratios, sort_indices):
    """
    Fully vectorized version of running sum.
    Input:
        sorted_abs_vals: [n_sig, n_obs]
        overlap_ratios: [n_term, n_sig, n_obs]
        sort_indices: [n_sig, n_obs]
    Output:
        ks_rs: [n_term, n_sig, n_obs]
        rc_rs: [n_term, n_sig, n_obs]
        sorted_or: [n_term, n_sig, n_obs]
    """
    _, n_sig, _ = overlap_ratios.shape

    # Hit/miss indicators
    hit_indicator = (overlap_ratios > 0).astype(int) # [n_term, n_sig, n_obs]
    miss_indicator = 1 - hit_indicator # [n_term, n_sig, n_obs]
    number_hit = hit_indicator.sum(axis=1)  # [n_term, n_obs]
    number_miss = n_sig - number_hit       # [n_term, n_obs]
    
    # Prepare: sort overlap_ratios for each term using sort_indices
    sorted_or = np.take_along_axis(overlap_ratios, sort_indices[None, :, :], axis=1)  # [n_term, n_sig, n_obs]

    # Normalize hit
    sum_hit_scores = np.sum(sorted_or * sorted_abs_vals[None, :, :], axis=1)  # [n_term, n_obs]
    sum_hit_scores[sum_hit_scores == 0] = np.finfo(float).eps
    norm_hit = 1.0 / sum_hit_scores  # [n_term, n_obs]

    # KS
    norm_miss = np.zeros_like(number_miss, dtype=float)
    nonzero_mask = number_miss > 0
    norm_miss[nonzero_mask] = 1.0 / number_miss[nonzero_mask]
    # Sorted miss indicators
    sorted_miss = np.take_along_axis(miss_indicator, sort_indices[None, :, :], axis=1)
    score = sorted_or * sorted_abs_vals[None, :, :] * norm_hit[:, None, :] - sorted_miss * norm_miss[:, None, :]
    ks_rs = np.cumsum(score, axis=1)  # cumsum over sigs
    # RC
    score = sorted_or * sorted_abs_vals[None, :, :] * norm_hit[:, None, :]
    rc_rs = np.cumsum(score, axis=1)  # cumsum over sigs
    return ks_rs, rc_rs, sorted_or

def get_running_sum_null(obj):
    """
    Batch version of null running sum.
    Randomly permutes overlap_ratios along sig axis.
    
    Args:
        obj.sorted_abs: [n_sig, n_obs]
        obj.overlap_ratios: [n_term, n_sig, n_obs]
        obj.sort_indices: [n_sig, n_obs]    
    Returns:
        ks_rs: [n_term, n_sig, n_obs]
        rc_rs: [n_term, n_sig, n_obs]
    """
    n_term, n_sig, n_obs =  obj.overlap_ratios.shape
    random_indices = np.argsort(np.random.rand(n_term, n_sig, n_obs), axis=1)  # [n_term, n_sig, n_obs]
    # use fancy indexing do batch-wise shuffle
    term_idx = np.arange(n_term)[:, None, None]
    obs_idx = np.arange(n_obs)[None, None, :]
    shuffled_or =  obj.overlap_ratios[term_idx, random_indices, obs_idx]  # [n_term, n_sig, n_obs]
    ks_rs, rc_rs, _ = _get_running_sum(obj.sorted_abs_vals, shuffled_or, obj.sort_indices)
    return ks_rs, rc_rs



def get_metrics_null(obj):
    """
    Compute null ES ESD AUC via permutations for shape [n_term, n_obs].
    Args:
        obj.sorted_abs_vals: np.ndarray of shape [n_term, n_sig, n_obs]
        obj.sorted_or: np.ndarray of shape [n_term, n_sig, n_obs]
        obj.sort_indices: np.ndarray of shape [n_term, n_obs] — sort order index per term/obs
        obj.n_perm: int — number of permutations
        obj.save_permutation: bool — whether to save running sum for each permutation
    Returns:
        obj.null_ES: np.ndarray of shape [n_perm, n_term, n_obs]
        obj.null_ESD: np.ndarray of shape [n_perm, n_term, n_obs]
        obj.null_peak: np.ndarray of shape [n_perm, n_term, n_obs]
        obj.null_AUC: np.ndarray of shape [n_perm, n_term, n_obs]
    """
    n_term, _, n_obs = obj.sorted_or.shape

    obj.null_ES = np.zeros((obj.n_perm, n_term, n_obs))
    obj.null_ESD = np.zeros((obj.n_perm, n_term, n_obs))
    obj.null_peak = np.zeros((obj.n_perm, n_term, n_obs))
    obj.null_AUC = np.zeros((obj.n_perm, n_term, n_obs))

    for i in range(obj.n_perm):
        ks_rs, rc_rs = get_running_sum_null(obj)
        if obj.save_permutation:
            if not os.path.exists("permutation_test"):
                os.makedirs("permutation_test")       
            np.save(os.path.join("permutation_test", f"permutation_{i}_ks_running_sum.npy"), ks_rs)
            np.save(os.path.join("permutation_test", f"permutation_{i}_rc_running_sum.npy"), rc_rs)
            print(f"Saved permutation {i} running sum to permutation_test/permutation_{i}_ks_running_sum.npy")
            print(f"Saved permutation {i} running sum to permutation_test/permutation_{i}_rc_running_sum.npy")

        ES, ESD, peak = _get_ES_ESD(ks_rs)
        obj.null_ES[i, :] = ES
        obj.null_ESD[i, :] = ESD
        obj.null_peak[i, :] = peak
        AUC = _get_AUC(rc_rs)  # AUC per [n_term, n_obs]
        obj.null_AUC[i, :] = AUC
    

def get_ES_ESD(obj):
    """
    Find the maximum absolute value (enrichment score, ES) and its index (peak) for each (n_term, n_obs)
    Find the enrichment score difference (ESD) for each (n_term, n_obs)
    Args:
        obj.ks_rs: [n_term, n_sig, n_obs]
    Returns:
        obj.ES: [n_term, n_obs] with the enrichment scores.
        obj.ESD: [n_term, n_obs] with the enrichment score differences.
        obj.peak: [n_term, n_obs] with the peak indices.
    """
    obj.ES, obj.ESD, obj.peak = _get_ES_ESD(obj.ks_rs)


def _get_ES_ESD(ks_rs):
    """
    Find the maximum absolute value (enrichment score, ES) and its index (peak) for each (n_term, n_obs)
    Find the enrichment score difference (ESD) for each (n_term, n_obs)
    Args:
        ks_rs: [n_term, n_sig, n_obs]
    Returns:
        ES: [n_term, n_obs] with the enrichment scores.
        ESD: [n_term, n_obs] with the enrichment score differences.
        peak: [n_term, n_obs] with the peak indices.
    """
    # Get indices of peak (maximum absolute running sum) per (n_term, n_obs)
    peak = np.argmax(np.abs(ks_rs), axis=1)
    # Gather ES values at peak indices
    ES = np.take_along_axis(ks_rs, peak[:, np.newaxis, :], axis=1).squeeze(axis=1)
    # Maximum positive value (per term, per obs)
    max_positive = np.max(np.where(ks_rs > 0, ks_rs, 0), axis=1)
    # Maximum negative value (per term, per obs)
    max_negative = np.min(np.where(ks_rs < 0, ks_rs, 0), axis=1)
    # Enrichment Score Difference (ESD)
    ESD = max_positive + max_negative
    return ES, ESD, peak


def get_AUC(obj):
    """
    Calculate AUC for each (n_term, n_obs) based on running sum matrix.
    Args:
        obj.rc_rs: np.ndarray of shape [n_term, n_sig, n_obs]
    Returns:
        obj.AUC: np.ndarray of shape [n_term, n_obs]
    """
    obj.AUC = _get_AUC(obj.rc_rs)


def _get_AUC(rc_rs):
    """
    Calculate AUC for each (n_term, n_obs) based on running sum matrix.
    Args:
        rs_matrix: np.ndarray of shape [n_term, n_sig, n_obs]
    Returns:
        AUC: np.ndarray of shape [n_term, n_obs]
    """
    _, n_sig, _ = rc_rs.shape
    rc_rs = rc_rs.astype(float)
    # Mask NaNs
    valid_mask = ~np.isnan(rc_rs)
    # Sum over n_sig dimension, ignoring NaNs
    sum_clean = np.nansum(rc_rs, axis=1)  # shape [n_term, n_obs]
    # Check if each (n_term, n_obs) slice has any valid values
    any_valid = np.any(valid_mask, axis=1)  # shape [n_term, n_obs]
    # Compute AUC with adjustment
    AUC = np.where(any_valid, np.maximum(0, (sum_clean - 0.5) / n_sig), np.nan)
    return AUC





    