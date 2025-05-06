import os
import numpy as np

def get_leading_edge(sorted_signame, sorted_or, ES, peak):
    lead_sigs = []
    for i in range(len(ES)):  
        if ES[i] > 0:
            local_indices = np.where(sorted_or[:peak[i]+1, i] > 0)[0]
            le_indices = local_indices 
        else:
            local_indices = np.where(sorted_or[peak[i]:, i] > 0)[0] 
            le_indices = local_indices + peak[i]

        le_genes = [str(sorted_signame[j, i]) for j in le_indices]  
        lead_sigs.append(le_genes)
        
    return [list(le) for le in lead_sigs]  

def get_leading_edge_batch(sorted_signame, sorted_or, ES, peak):
    """
    Get leading-edge sig names for each (n_term, n_obs) enrichment signal.

    Args:
        sorted_signame: np.ndarray of shape [n_term, n_sig, n_obs] — names of sorted signatures [interactions or genes].
        sorted_or: np.ndarray of shape [n_term, n_sig, n_obs] — overlap ratios.
        ES: np.ndarray of shape [n_term, n_obs] — enrichment scores.
        peak: np.ndarray of shape [n_term, n_obs] — index of ES peaks.

    Returns:
        lead_sigs: List[List[List[str]]] of shape [n_term][n_obs][#sigs] — names of leading-edge sigs.
    """
    n_term, _, n_obs = sorted_or.shape
    # nested list comprehensions, which are faster than for loops in Python
    lead_sigs = [
        [
            [
                str(sorted_signame[t, j, i]) 
                for j in (np.where(sorted_or[t, :peak[t, i]+1, i] > 0)[0]
                          if ES[t, i] > 0 else
                          np.where(sorted_or[t, peak[t, i]:, i] > 0)[0] + peak[t, i])
            ]
            for i in range(n_obs)
        ]
        for t in range(n_term)
    ]

    return lead_sigs  # shape: [n_term][n_obs][#genes]


def get_overlap(sig_name, lib_genes, sig_sep):
    n_obs = sig_name.shape[1]
    overlap_ratios = np.zeros(sig_name.shape)
    n_hits = np.zeros(n_obs)
    for j in range(n_obs):
        for i in range(sig_name.shape[0]):
            sig_names = set(sig_name[i, j].split(sig_sep))
            # if i < 10 and j == 0:
            #     print(f"Row {i}, Col {j}, original: {sig_name[i, j]}, split into: {sig_names}")
            n = len(set(lib_genes).intersection(sig_names))
            overlap_ratios[i, j] = n/len(sig_names)
            if n > 0:
                n_hits[j] += 1
    return overlap_ratios, n_hits


def get_overlap_batch(sig_name_matrix, lib_genes_list, sig_sep=','):
    """
    Batch version of get_overlap.
    
    Args:
        sig_name_matrix: np.ndarray of shape [n_sig, n_obs], strings of sig_name separated by sig_sep
        lib_genes_list: List[List[str]], length = n_term, each is a set of genes
        sig_sep: separator used in sig_name
        
    Returns:
        overlap_ratio_matrix: np.ndarray of shape [n_term, n_sig, n_obs]
        n_hit_matrix: np.ndarray of shape [n_term, n_obs] # number of gene hits in the obs
    """
    n_sig, n_obs = sig_name_matrix.shape
    n_term = len(lib_genes_list)

    overlap_ratio_matrix = np.zeros((n_term, n_sig, n_obs), dtype=np.float32)
    n_hit_matrix = np.zeros((n_term, n_obs), dtype=np.int32)

    overlap_ratio_dict = {}
    for t, lib_genes in enumerate(lib_genes_list):
        lib_genes = set(lib_genes)
        for j in range(n_obs):
            for i in range(n_sig):
                sig_name = sig_name_matrix[i, j]
                if (t, sig_name) in overlap_ratio_dict:
                    ratio = overlap_ratio_dict[(t, sig_name)]
                else:
                    genes_in_cell = set(sig_name.split(sig_sep))
                    n_overlap = len(genes_in_cell & lib_genes)
                    
                    if len(genes_in_cell) > 0:
                        ratio = n_overlap / len(genes_in_cell)
                        overlap_ratio_dict[(t, sig_name)] = ratio
                if ratio > 0:
                    overlap_ratio_matrix[t, i, j] = ratio
                    n_hit_matrix[t, j] += 1
    return overlap_ratio_matrix, n_hit_matrix


def get_running_sum(sig_val, overlap_ratios, metric):
    sort_indices = np.argsort(sig_val, axis=0)[::-1, :]
    sorted_sig = np.take_along_axis(sig_val, sort_indices, axis=0)
    sorted_abs = np.abs(sorted_sig)   
    obs_rs = get_running_sum_aux(sorted_abs, overlap_ratios, sort_indices, metric=metric)

    return obs_rs, sort_indices


def get_running_sum_batch(sig_val_matrix, overlap_ratio_matrix, metric):
    """
    Batch version: supports multiple terms.

    Args:
        sig_val_matrix: np.ndarray [n_sig, n_obs]
        overlap_ratio_matrix: np.ndarray [n_term, n_sig, n_obs]
        metric: 'KS-ES', 'KS-ESD' or 'RC-AUC'

    Returns:
        rs_matrix: np.ndarray [n_term, n_sig, n_obs]
        sort_indices: np.ndarray [n_sig, n_obs]
    """
    sort_indices = np.argsort(sig_val_matrix, axis=0)[::-1, :]  # shape [n_sig, n_obs]
    sorted_sig = np.take_along_axis(sig_val_matrix, sort_indices, axis=0)  # [n_sig, n_obs]
    sorted_abs = np.abs(sorted_sig)

    rs_matrix, sorted_or = get_running_sum_aux_batch(sorted_abs, overlap_ratio_matrix, sort_indices, metric=metric)

    return rs_matrix, sorted_abs, sorted_or, sort_indices


def get_running_sum_null(sorted_abs, overlap_ratios, sort_indices, metric='KS'):
    n_sig, n_obs = overlap_ratios.shape
    indices = np.argsort(np.random.rand(n_sig, n_obs), axis=0)
    shuffled = overlap_ratios[indices, np.arange(n_obs)]    
    return get_running_sum_aux(sorted_abs, shuffled, sort_indices, metric=metric)


def get_running_sum_null_batch(sorted_abs, overlap_ratios, sort_indices, metric='KS'):
    """
    Batch version of null running sum.
    Randomly permutes overlap_ratios along sig axis.
    
    Args:
        sorted_abs: [n_sig, n_obs]
        overlap_ratios: [n_term, n_sig, n_obs]
        sort_indices: [n_sig, n_obs]
        metric: 'KS' or 'RC'
    
    Returns:
        null_rs: [n_term, n_sig, n_obs]
    """
    n_term, n_sig, n_obs = overlap_ratios.shape
    random_indices = np.argsort(np.random.rand(n_term, n_sig, n_obs), axis=1)  # [n_term, n_sig, n_obs]
    # use fancy indexing do batch-wise shuffle
    term_idx = np.arange(n_term)[:, None, None]
    obs_idx = np.arange(n_obs)[None, None, :]
    shuffled_or = overlap_ratios[term_idx, random_indices, obs_idx]  # [n_term, n_sig, n_obs]
    running_sum, sorted_or = get_running_sum_aux_batch(sorted_abs, shuffled_or, sort_indices, metric=metric)
    return running_sum

def get_running_sum_aux(sorted_abs, overlap_ratios, sort_indices, metric='KS-ESD'):
    # sig_val, n_sig x n_obs
    # overlap_ratios, n_sig x n_obs
    n_sig = overlap_ratios.shape[0]
    hit_indicator = (overlap_ratios > 0).astype(int)
    miss_indicator = 1 - hit_indicator

    number_hit = hit_indicator.sum(axis=0)
    number_miss = n_sig - number_hit
    #print(signature.shape, overlap_ratios.shape)
    sorted_or = np.take_along_axis(overlap_ratios, sort_indices, axis=0)
    sum_hit_scores = np.sum(sorted_abs * sorted_or, axis=0)
    sum_hit_scores[sum_hit_scores == 0] = np.finfo(float).eps
    norm_hit = 1.0 / sum_hit_scores.astype(float)
    if metric.startswith('KS'):
        norm_miss = np.zeros_like(number_miss, dtype=float)
        nonzero_mask = number_miss > 0  
        norm_miss[nonzero_mask] = 1.0 / number_miss[nonzero_mask]
        sorted_miss = np.take_along_axis(miss_indicator, sort_indices, axis=0)
        score = sorted_or * sorted_abs * norm_hit[np.newaxis, :] - sorted_miss * norm_miss
    else:  # RC - recovery curve
        score = sorted_or * sorted_abs * norm_hit[np.newaxis, :]
    running_sum = np.cumsum(score, axis=0)
    return running_sum


def get_running_sum_aux_batch(sorted_abs, overlap_ratio_matrix, sort_indices,
                              metric='KS-ESD'):
    """
    Fully vectorized version of running sum.
    Args:
        sorted_abs: [n_sig, n_obs]
        overlap_ratio_matrix: [n_term, n_sig, n_obs]
        sort_indices: [n_sig, n_obs]
        metric: 'KS-ES', 'KS-ESD' or 'RC-AUC'
    Returns:
        rs_matrix: [n_term, n_sig, n_obs]
        sorted_or: [n_term, n_sig, n_obs]
    """
    _, n_sig, _ = overlap_ratio_matrix.shape

    # Hit/miss indicators
    hit_indicator = (overlap_ratio_matrix > 0).astype(int) # [n_term, n_sig, n_obs]
    miss_indicator = 1 - hit_indicator # [n_term, n_sig, n_obs]
    number_hit = hit_indicator.sum(axis=1)  # [n_term, n_obs]
    number_miss = n_sig - number_hit       # [n_term, n_obs]
    
    # Prepare: sort overlap_ratios for each term using sort_indices
    sorted_or = np.take_along_axis(overlap_ratio_matrix, sort_indices[None, :, :], axis=1)  # [n_term, n_sig, n_obs]
    # Normalize hit
    sum_hit_scores = np.sum(sorted_or * sorted_abs[None, :, :], axis=1)  # [n_term, n_obs]
    sum_hit_scores[sum_hit_scores == 0] = np.finfo(float).eps
    norm_hit = 1.0 / sum_hit_scores  # [n_term, n_obs]

    if metric.startswith('KS'):
        norm_miss = np.zeros_like(number_miss, dtype=float)
        nonzero_mask = number_miss > 0
        norm_miss[nonzero_mask] = 1.0 / number_miss[nonzero_mask]
        # Sorted miss indicators
        sorted_miss = np.take_along_axis(miss_indicator, sort_indices[None, :, :], axis=1)
        score = sorted_or * sorted_abs[None, :, :] * norm_hit[:, None, :] - sorted_miss * norm_miss[:, None, :]
    else:  # RC
        score = sorted_or * sorted_abs[None, :, :] * norm_hit[:, None, :]

    running_sum = np.cumsum(score, axis=1)  # cumsum over sigs
    return running_sum, sorted_or



def get_AUC(obs_rs):
    # running_sum  n_sig x n_obs
    n_sig, n_obs = obs_rs.shape
    obs_rs = obs_rs.astype(float)
    
    valid_mask = ~np.isnan(obs_rs)
    
    sum_clean = np.nansum(obs_rs, axis=0)  # 对每列求和，忽略 NaN
    any_valid = np.any(valid_mask, axis=0)  # 检查每列是否至少有一个非 NaN 值

    AUCs = np.where(any_valid, np.maximum(0, (sum_clean - 0.5) / n_sig), np.nan)
            
    return AUCs

def get_AUC_batch(rs_matrix):
    """
    Calculate AUC for each (n_term, n_obs) based on running sum matrix.
    Args:
        rs_matrix: np.ndarray of shape [n_term, n_sig, n_obs]
    Returns:
        AUCs: np.ndarray of shape [n_term, n_obs]
    """
    _, n_sig, _ = rs_matrix.shape
    rs_matrix = rs_matrix.astype(float)
    # Mask NaNs
    valid_mask = ~np.isnan(rs_matrix)
    # Sum over n_sig dimension, ignoring NaNs
    sum_clean = np.nansum(rs_matrix, axis=1)  # shape [n_term, n_obs]
    # Check if each (n_term, n_obs) slice has any valid values
    any_valid = np.any(valid_mask, axis=1)  # shape [n_term, n_obs]
    # Compute AUC with adjustment
    AUCs = np.where(any_valid, np.maximum(0, (sum_clean - 0.5) / n_sig), np.nan)
    return AUCs


def get_AUC_null(sorted_abs, sorted_or, sort_indices, n_perm=1000,save_permutation=False):
    n_sig, n_obs = sorted_abs.shape

    null_AUC = np.zeros((n_perm, n_obs))

    for i in range(n_perm):
        rs = get_running_sum_null(sorted_abs, sorted_or, sort_indices, metric="RC-AUC")
        if save_permutation:
            print(f"Saved permutation {i} running sum to permutation_test/permutation_{i}_running_sum.npy")
            if not os.path.exists("permutation_test"):
                os.makedirs("permutation_test")       
            np.save(os.path.join("permutation_test", f"permutation_{i}_running_sum.npy"), rs)
        null_AUC[i, :] = get_AUC(rs)

    return null_AUC

def get_AUC_null_batch(sorted_abs, sorted_or, sort_indices, n_perm=1000, save_permutation=False):
    """
    Compute null AUCs via permutations for shape [n_term, n_obs].
    Args:
        sorted_abs: np.ndarray of shape [n_sig, n_obs]
        sorted_or: np.ndarray of shape [n_term, n_sig, n_obs]
        sort_indices: np.ndarray of shape [n_term, n_obs] — sort order index per term/obs
        n_perm: int — number of permutations
        save_permutation: bool — whether to save running sum for each permutation
    Returns:
        null_AUC: np.ndarray of shape [n_perm, n_term, n_obs]
    """
    n_term, n_sig, n_obs = sorted_or.shape
    null_AUC = np.zeros((n_perm, n_term, n_obs))

    for i in range(n_perm):
        rs = get_running_sum_null_batch(sorted_abs, sorted_or, sort_indices, metric="RC-AUC")  
        # should return [n_term, n_sig, n_obs]

        if save_permutation:
            print(f"Saved permutation {i} running sum to permutation_test/permutation_{i}_running_sum.npy")
            os.makedirs("permutation_test", exist_ok=True)
            np.save(os.path.join("permutation_test", f"permutation_{i}_running_sum.npy"), rs)

        null_AUC[i] = get_AUC_batch(rs)  # AUC per [n_term, n_obs]

    return null_AUC


def get_ES_ESD(obs_rs):
    # Find the maximum absolute value (enrichment score, ES) and its index (peak) for each column
    peak = np.argmax(np.abs(obs_rs), axis=0)
    ES = obs_rs[peak, np.arange(obs_rs.shape[1])]
    # Find the maximum positive value for each column
    max_positive = np.max(np.where(obs_rs > 0, obs_rs, 0), axis=0)
    # Find the maximum negative value for each column
    max_negative = np.min(np.where(obs_rs < 0, obs_rs, 0), axis=0)
    # Calculate the enrichment score difference (ESD) for each column
    ESD = max_positive + max_negative
    return ES, ESD, peak

    
def get_ES_ESD_batch(rs_matrix):
    """
    Find the maximum absolute value (enrichment score, ES) and its index (peak) for each (n_term, n_obs)
    Find the enrichment score difference (ESD) for each (n_term, n_obs)
    Args:
        rs_matrix: [n_term, n_sig, n_obs]
    Returns:
        ES: [n_term, n_obs] with the enrichment scores.
        ESD: [n_term, n_obs] with the enrichment score differences.
        peak: [n_term, n_obs] with the peak indices.
    """
    # Get indices of peak (maximum absolute running sum) per (n_term, n_obs)
    peak = np.argmax(np.abs(rs_matrix), axis=1)
    # Gather ES values at peak indices
    ES = np.take_along_axis(rs_matrix, peak[:, np.newaxis, :], axis=1).squeeze(axis=1)
    # Maximum positive value (per term, per obs)
    max_positive = np.max(np.where(rs_matrix > 0, rs_matrix, 0), axis=1)
    # Maximum negative value (per term, per obs)
    max_negative = np.min(np.where(rs_matrix < 0, rs_matrix, 0), axis=1)
    # Enrichment Score Difference (ESD)
    ESD = max_positive + max_negative
    return ES, ESD, peak


    null_ES, null_ESD, _ = enrich.get_ES_ESD_null_batch(
        sorted_abs, sorted_or, sort_indices,
        n_perm=n_perm, save_permutation=save_permutation
    )  # shape: [n_perm, n_term, n_obs]


def get_ES_ESD_null_batch(sorted_abs, sorted_or, sort_indices,
                          n_perm=1000, save_permutation=False):
    """
    Compute null ES ESD via permutations for shape [n_term, n_obs].
    Args:
        sorted_abs: np.ndarray of shape [n_term, n_sig, n_obs]
        sorted_or: np.ndarray of shape [n_term, n_sig, n_obs]
        sort_indices: np.ndarray of shape [n_term, n_obs] — sort order index per term/obs
        n_perm: int — number of permutations
        save_permutation: bool — whether to save running sum for each permutation
    Returns:
        null_ES: np.ndarray of shape [n_perm, n_term, n_obs]
        null_ESD: np.ndarray of shape [n_perm, n_term, n_obs]
        null_peak: np.ndarray of shape [n_perm, n_term, n_obs]
    """
    n_term, _, n_obs = sorted_or.shape

    null_ES = np.zeros((n_perm, n_term, n_obs))
    null_ESD = np.zeros((n_perm, n_term, n_obs))
    null_peak = np.zeros((n_perm, n_term, n_obs))
    
    for i in range(n_perm):
        rs = get_running_sum_null_batch(sorted_abs, sorted_or, sort_indices, metric="KS")
        if save_permutation:
            print(f"Saved permutation {i} running sum to permutation_test/permutation_{i}_running_sum.npy")
            if not os.path.exists("permutation_test"):
                os.makedirs("permutation_test")       
            np.save(os.path.join("permutation_test", f"permutation_{i}_running_sum.npy"), rs)
        ES, ESD, peak = get_ES_ESD_batch(rs)
        null_ES[i, :] = ES
        null_ESD[i, :] = ESD
        null_peak[i, :] = peak

    return null_ES, null_ESD, null_peak


def get_ES_ESD_null(sorted_abs, sorted_or, sort_indices,
                    n_perm=1000, save_permutation=False):

    n_sig, n_obs = sorted_abs.shape

    null_ES = np.zeros((n_perm, n_obs))
    null_ESD = np.zeros((n_perm, n_obs))
    null_peak = np.zeros((n_perm, n_obs))
    
    for i in range(n_perm):
        rs = get_running_sum_null(sorted_abs, sorted_or, sort_indices, metric="KS")
        if save_permutation:
            print(f"Saved permutation {i} running sum to permutation_test/permutation_{i}_running_sum.npy")
            if not os.path.exists("permutation_test"):
                os.makedirs("permutation_test")       
            np.save(os.path.join("permutation_test", f"permutation_{i}_running_sum.npy"), rs)
        es, esd, peak = get_ES_ESD(rs)
        null_ES[i, :] = es
        null_ESD[i, :] = esd
        null_peak[i, :] = peak

    return null_ES, null_ESD, null_peak






    