import os
import numpy as np
import random
from scipy.linalg import svd
from scipy.spatial.distance import pdist, squareform

# def get_leading_edge(i2sig, hit_indicator, ES, peak):
#     if ES > 0:
#         le_genes = [i2sig[i] for i, x in enumerate(hit_indicator[:peak+1]) if x > 0]
#     else:
#         le_genes = [i2sig[i] for i, x in enumerate(hit_indicator[-peak:]) if x > 0]
#     return le_genes


def get_leading_edge(i2sig, hit_indicator, ES, peak):
    leading_edge_genes = []
    for i in range(len(ES)):  
        if ES[i] > 0:
            le_indices = np.where(hit_indicator[:peak[i]+1, i] > 0)[0] 
        else:
            le_indices = np.where(hit_indicator[peak[i]:, i] > 0)[0] 

        le_genes = [str(i2sig[j, i]) for j in le_indices]  
        leading_edge_genes.append(le_genes)
        
    return [list(le) for le in leading_edge_genes]  


def get_overlap(sig_name, lib_sigs, sig_sep):
    n_sample = sig_name.shape[1]
    overlap_ratios = np.zeros(sig_name.shape)
    n_hits = np.zeros(n_sample)
    for j in range(n_sample):
        for i in range(sig_name.shape[0]):
            sig_names = set(sig_name[i, j].split(sig_sep))
            # if i < 10 and j == 0:
            #     print(f"Row {i}, Col {j}, original: {sig_name[i, j]}, split into: {sig_names}")
            n = len(set(lib_sigs).intersection(sig_names))
            overlap_ratios[i, j] = n/len(sig_names)
            if n > 0:
                n_hits[j] += 1
    return overlap_ratios, n_hits


def get_running_sum(sig_val, overlap_ratios, method='KS'):
    sort_indices = np.argsort(sig_val, axis=0)[::-1, :]
    sorted_sig = np.take_along_axis(sig_val, sort_indices, axis=0)
    sorted_abs = np.abs(sorted_sig)   
    obs_rs = get_running_sum_aux(sorted_abs, overlap_ratios, sort_indices, method=method)

    # null_rs = np.zeros((n_perm, *sig_val.shape))  
    # # n_perm x n_sig x n_sample 
    # for i in range(n_perm):
    #     rs = get_running_sum_null(sorted_abs, overlap_ratios, sort_indices, method=method)
    #     null_rs[i, :, :] = rs
    return obs_rs, sort_indices


def get_running_sum_null(sorted_abs, overlap_ratios, sort_indices, method='KS'):
    n_sig, n_sample = overlap_ratios.shape
    indices = np.argsort(np.random.rand(n_sig, n_sample), axis=0)
    shuffled = overlap_ratios[indices, np.arange(n_sample)]    
    return get_running_sum_aux(sorted_abs, shuffled, sort_indices, method=method)


def get_running_sum_aux(sorted_abs, overlap_ratios, sort_indices, method='KS'):
    # sig_val, n_sig x n_sample
    # overlap_ratios, n_sig x n_sample
    
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

    if method == 'KS':
        norm_miss = np.zeros_like(number_miss, dtype=float)
        nonzero_mask = number_miss > 0  
        norm_miss[nonzero_mask] = 1.0 / number_miss[nonzero_mask]

        
        sorted_miss = np.take_along_axis(miss_indicator, sort_indices, axis=0)
        score = sorted_or * sorted_abs * norm_hit[np.newaxis, :] - sorted_miss * norm_miss
    else:  # RC - recovery curve
        score = sorted_or * sorted_abs * norm_hit[np.newaxis, :]

    running_sum = np.cumsum(score, axis=0)
    return running_sum


def get_AUC(obs_rs):
    # running_sum  n_sig x n_sample
    n_sig, n_sample = obs_rs.shape
    obs_rs = obs_rs.astype(float)
    
    valid_mask = ~np.isnan(obs_rs)
    
    sum_clean = np.nansum(obs_rs, axis=0)  # 对每列求和，忽略 NaN
    any_valid = np.any(valid_mask, axis=0)  # 检查每列是否至少有一个非 NaN 值

    AUCs = np.where(any_valid, np.maximum(0, (sum_clean - 0.5) / n_sig), np.nan)
            
    return AUCs

# def get_AUC_null(null_rs):
#     """
#     Compute the area under the curve (AUC) for a 3D array (null_rs) with size (n_perm, n_sig, n_sample).

#     Parameters:
#         null_rs (np.ndarray): Array of size (n_perm, n_sig, n_sample).

#     Returns:
#         AUCs (np.ndarray): Array of size (n_perm, n_sample) with the computed AUCs.
#     """
#     null_rs = null_rs.copy()
#     n_perm, n_sig, n_sample = null_rs.shape   
#     AUCs = np.zeros((n_perm, n_sample))
#     for i in range(n_sample):    
#         sample_data = null_rs[:, :, i]  # shape: (n_perm, n_sig)
#         valid_mask = ~np.isnan(sample_data).any(axis=1)
#         if valid_mask.any():
#             AUCs[valid_mask, i] = np.sum(sample_data[valid_mask] * (1.0 / n_sig), axis=1)
#         else:
#             AUCs[:, i] = np.nan
#     return AUCs

def get_AUC_null(sorted_abs, overlap_ratios,sort_indices, n_perm=1000,save_permutation=False):
    n_sig, n_sample = sorted_abs.shape

    AUCs = np.zeros((n_perm, n_sample))

    for i in range(n_perm):
        rs = get_running_sum_null(sorted_abs, overlap_ratios, sort_indices, method="RC")
        if save_permutation:
            print(f"Saved permutation {i} running sum to permutation_test/permutation_{i}_running_sum.npy")
            if not os.path.exists("permutation_test"):
                os.makedirs("permutation_test")       
            np.save(os.path.join("permutation_test", f"permutation_{i}_running_sum.npy"), rs)
        AUCs[i, :] = get_AUC(rs)

    return AUCs

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
    # # If there's only one sample, return scalars instead of arrays
    # if ES.size == 1:
    #     ES = ES[0]
    #     ESD = ESD[0]
    #     peak = peak[0]
    return ES, ESD, peak


# def get_ES_ESD_null(null_rs):
#     """
#     Compute ES, ESD, and peak for a 3D array (null_rs) with size (n_perm, n_sig, n_sample).

#     Parameters:
#         null_rs (np.ndarray): Array of size (n_perm, n_sig, n_sample).

#     Returns:
#         ES (np.ndarray): Array of size (n_perm, n_sample) with the enrichment scores.
#         ESD (np.ndarray): Array of size (n_perm, n_sample) with the enrichment score differences.
#         peak (np.ndarray): Array of size (n_perm, n_sample) with the peak indices.
#     """
#     # Find the maximum absolute value (ES) and its index (peak) for each n_perm and n_sample
#     peak = np.argmax(np.abs(null_rs), axis=1)
#     ES = np.take_along_axis(null_rs, peak[:, np.newaxis, :], axis=1).squeeze(axis=1)
#     # Find the maximum positive value for each n_perm and n_sample
#     max_positive = np.max(np.where(null_rs > 0, null_rs, 0), axis=1)
#     # Find the maximum negative value for each n_perm and n_sample
#     max_negative = np.min(np.where(null_rs < 0, null_rs, 0), axis=1)
#     # Calculate the enrichment score difference (ESD) for each n_perm and n_sample
#     ESD = max_positive + max_negative
#     return ES, ESD, peak

def get_ES_ESD_null(sorted_abs, overlap_ratios,sort_indices, n_perm=1000,save_permutation=False):

    n_sig, n_sample = sorted_abs.shape

    ES = np.zeros((n_perm, n_sample))
    ESD = np.zeros((n_perm, n_sample))
    peek = np.zeros((n_perm, n_sample))
    
    for i in range(n_perm):
        rs = get_running_sum_null(sorted_abs, overlap_ratios, sort_indices, method="KS")
        if save_permutation:
            print(f"Saved permutation {i} running sum to permutation_test/permutation_{i}_running_sum.npy")
            if not os.path.exists("permutation_test"):
                os.makedirs("permutation_test")       
            np.save(os.path.join("permutation_test", f"permutation_{i}_running_sum.npy"), rs)
        es, esd, peak = get_ES_ESD(rs)
        ES[i, :] = es
        ESD[i, :] = esd
        peek[i, :] = peak


    return ES, ESD, peak


def get_plage(sig_val):
    """
    Perform PLAGE (Pathway Level Analysis of Gene Expression) analysis.
    """
    sort_indices = np.argsort(sig_val, axis=0)[::-1, :]
    sorted_sig = np.take_along_axis(sig_val, sort_indices, axis=0)
    sorted_abs = np.abs(sorted_sig) 
    U, s, Vt = svd(sorted_abs, full_matrices=False)
    plage = Vt[0, :]
    return plage

def get_plage_null(sig_val, n_perm=1000):
    """
    Generate null distribution for PLAGE activity scores through permutation.
    """
    n_sig, n_sample = sig_val.shape
    plage_null = np.zeros((n_perm, n_sample))
    
    for i in range(n_perm):  
        perm_inx = np.array([np.random.permutation(n_sig) for _ in range(n_sample)])
        perm_sig_val = np.array([sig_val[idx, j] for j, idx in enumerate(perm_inx.T)]).T
        plage_null[i, :] = get_plage(perm_sig_val)
    return plage_null




def get_z_score(sig_val):
    zscore = []
    sort_indices = np.argsort(sig_val, axis=0)[::-1, :]
    sorted_sig = np.take_along_axis(sig_val, sort_indices, axis=0)
    sorted_abs = np.abs(sorted_sig) 
    zscore.append(np.sum(sig_val[sorted_abs, :], axis=0) / np.sqrt(len(sorted_abs)))
    return zscore   

def get_z_score_null(sig_val, n_perm=1000):

    n_sig, n_sample = sig_val.shape
    z_score_null = np.zeros((n_perm, n_sample))
    for i in range(n_perm):
        perm_inx = np.array([np.random.permutation(n_sig) for _ in range(n_sample)])
        perm_sig_val = np.array([sig_val[idx, j] for j, idx in enumerate(perm_inx.T)]).T
        z_score_null[i, :] = get_z_score(perm_sig_val)     
    
    return z_score_null


def get_vision(sig_val, overlap_ratios, lib_sigs, sig_sep: str = ','):

    """
    Calculate VISION style signature scores using formula:
    sj = (Σ(Gpos) egj - Σ(Gneg) egj) / (|Gpos| + |Gneg|)
    """
    pos_genes = []
    neg_genes = []
    for i in range(sig_val.shape[0]):
        sig_names = set(sig_val[i, 0].split(sig_sep))
        if len(set(lib_sigs).intersection(sig_names)) > 0:
            pos_genes.append(i)
        else:
            neg_genes.append(i)
    vision = (np.sum(sig_val[pos_genes], axis=0) - np.sum(sig_val[neg_genes], axis=0)) / (len(pos_genes) + len(neg_genes))
    
    return vision



def get_vision_null(sig_val, n_perm=1000):

    """
    Generate null distribution for VISION.
    """
    n_sig, n_sample = sig_val.shape
    vision_null = np.zeros((n_perm, n_sample))
    for i in range(n_perm):
        perm_inx = np.array([np.random.permutation(n_sig) for _ in range(n_sample)])
        perm_sig_val = np.array([sig_val[idx, j] for j, idx in enumerate(perm_inx.T)]).T
        vision_null[i, :] = get_vision(perm_sig_val)

    return vision_null





    