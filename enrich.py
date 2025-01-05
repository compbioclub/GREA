import numpy as np
import random

def get_leading_edge(i2sig, hit_indicator, ES, peak):
    if ES > 0:
        le_genes = [i2sig[i] for i, x in enumerate(hit_indicator[:peak+1]) if x > 0]
    else:
        le_genes = [i2sig[i] for i, x in enumerate(hit_indicator[-peak:]) if x > 0]
    return le_genes

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


def get_running_sum(sig_val, overlap_ratios, method='KS', n_perm=1000):
    sort_indices = np.argsort(sig_val, axis=0)[::-1, :]
    sorted_sig = np.take_along_axis(sig_val, sort_indices, axis=0)
    sorted_abs = np.abs(sorted_sig)   
    obs_rs = get_running_sum_aux(sorted_abs, overlap_ratios, sort_indices, method=method)

    null_rs = np.zeros((n_perm, *sig_val.shape))  
    # n_perm x n_sig x n_sample 
    for i in range(n_perm):
        rs = get_running_sum_null(sorted_abs, overlap_ratios, sort_indices, method=method)
        null_rs[i, :, :] = rs
    return obs_rs, null_rs


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
        norm_miss = 1.0 / number_miss
        sorted_miss = np.take_along_axis(miss_indicator, sort_indices, axis=0)
        score = sorted_or * sorted_abs * norm_hit[np.newaxis, :] - sorted_miss * norm_miss
    else:  # RC - recovery curve
        score = sorted_or * sorted_abs * norm_hit[np.newaxis, :]

    running_sum = np.cumsum(score, axis=0)
    return running_sum


    # if n_sig == 1:
    #     sorted_or = np.take_along_axis(overlap_ratios[:, np.newaxis], sort_indices, axis=0)
    # else:
    #     sorted_or = np.take_along_axis(overlap_ratios, sort_indices, axis=0)

    # sum_hit_scores = np.sum(sorted_abs * sorted_or, axis=0)
    # norm_hit = 1.0/sum_hit_scores.astype(float)

    # if method == 'KS':
    #     norm_miss = 1.0/number_miss
    #     if n_sig == 1:
    #         sorted_miss = np.take_along_axis(miss_indicator[:, np.newaxis], sort_indices, axis=0)
    #     else:
    #         sorted_miss = np.take_along_axis(miss_indicator, sort_indices, axis=0)

    #     score = sorted_or * sorted_abs * norm_hit[np.newaxis, :] - sorted_miss * norm_miss
    # else: # RC - recovery curve
    #     score = sorted_or * sorted_abs * norm_hit[np.newaxis, :]
    # running_sum = np.cumsum(score, axis=0)
    # running_sum  n_sig x n_sample

    # return running_sum

def get_AUC(obs_rs):
    # running_sum  n_sig x n_sample
    n_sig, n_sample = obs_rs.shape
    AUCs = np.zeros(n_sample)
    
    for i in range(n_sample):
        data = obs_rs[:, i]
        valid_mask = ~np.isnan(data)
        
        if np.any(valid_mask):
            clean_data = data[valid_mask]
            AUCs[i] = (clean_data * (1.0 / n_sig)).sum()
        else:
            AUCs[i] = np.nan
            
    return AUCs

def get_AUC_null(null_rs):
    """
    Compute the area under the curve (AUC) for a 3D array (null_rs) with size (n_perm, n_sig, n_sample).

    Parameters:
        null_rs (np.ndarray): Array of size (n_perm, n_sig, n_sample).

    Returns:
        AUCs (np.ndarray): Array of size (n_perm, n_sample) with the computed AUCs.
    """
    n_perm, n_sig, n_sample = null_rs.shape
    
    # 对每个样本分别计算AUC
    AUCs = np.zeros((n_perm, n_sample))
    
    for i in range(n_sample):
        # 获取当前样本的数据
        sample_data = null_rs[:, :, i]  # shape: (n_perm, n_sig)
        
        # 移除包含NaN的行
        valid_mask = ~np.isnan(sample_data).any(axis=1)
        clean_data = sample_data[valid_mask, :]
        
        if clean_data.shape[0] > 0:  # 确保有有效数据
            # 计算AUC
            AUCs[valid_mask, i] = (clean_data * (1.0 / n_sig)).sum(axis=1)
        else:
            AUCs[:, i] = np.nan
    
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


def get_ES_ESD_null(null_rs):
    """
    Compute ES, ESD, and peak for a 3D array (null_rs) with size (n_perm, n_sig, n_sample).

    Parameters:
        null_rs (np.ndarray): Array of size (n_perm, n_sig, n_sample).

    Returns:
        ES (np.ndarray): Array of size (n_perm, n_sample) with the enrichment scores.
        ESD (np.ndarray): Array of size (n_perm, n_sample) with the enrichment score differences.
        peak (np.ndarray): Array of size (n_perm, n_sample) with the peak indices.
    """
    # Find the maximum absolute value (ES) and its index (peak) for each n_perm and n_sample
    peak = np.argmax(np.abs(null_rs), axis=1)
    ES = np.take_along_axis(null_rs, peak[:, np.newaxis, :], axis=1).squeeze(axis=1)
    # Find the maximum positive value for each n_perm and n_sample
    max_positive = np.max(np.where(null_rs > 0, null_rs, 0), axis=1)
    # Find the maximum negative value for each n_perm and n_sample
    max_negative = np.min(np.where(null_rs < 0, null_rs, 0), axis=1)
    # Calculate the enrichment score difference (ESD) for each n_perm and n_sample
    ESD = max_positive + max_negative
    return ES, ESD, peak