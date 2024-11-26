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
            sig_names = sig_name[i, j].split(sig_sep)
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


    if n_sig == 1:
        sorted_or = np.take_along_axis(overlap_ratios[:, np.newaxis], sort_indices, axis=0)
    else:
        sorted_or = np.take_along_axis(overlap_ratios, sort_indices, axis=0)

    sum_hit_scores = np.sum(sorted_abs * sorted_or, axis=0)
    norm_hit = 1.0/sum_hit_scores.astype(float)

    if method == 'KS':
        norm_miss = 1.0/number_miss
        if n_sig == 1:
            sorted_miss = np.take_along_axis(miss_indicator[:, np.newaxis], sort_indices, axis=0)
        else:
            sorted_miss = np.take_along_axis(miss_indicator, sort_indices, axis=0)

        score = sorted_or * sorted_abs * norm_hit[np.newaxis, :] - sorted_miss * norm_miss
    else: # RC - recovery curve
        score = sorted_or * sorted_abs * norm_hit[np.newaxis, :]
    running_sum = np.cumsum(score, axis=0)
    # running_sum  n_sig x n_sample

    return running_sum

def get_SE(running_sum):
    ES, ESD, peak = get_ES_ESD(running_sum)
    le_genes = [] # get_leading_edge(i2sig, hit_indicator, ES, peak)

    return ES, ESD, peak, le_genes

def get_AUC(obs_rs):
    # running_sum  n_sig x n_sample
    n_sig = obs_rs.shape[0]
    AUCs = (obs_rs * 1/n_sig).sum(axis=0)
    return AUCs

def get_AUC_null(null_rs):
    """
    Compute the area under the curve (AUC) for a 3D array (null_rs) with size (n_perm, n_sig, n_sample).

    Parameters:
        null_rs (np.ndarray): Array of size (n_perm, n_sig, n_sample).

    Returns:
        AUCs (np.ndarray): Array of size (n_perm, n_sample) with the computed AUCs.
    """
    n_sig = null_rs.shape[1]  # Number of signals
    AUCs = (null_rs * (1 / n_sig)).sum(axis=1)  # Sum along the n_sig axis
    return AUCs

def get_SE_old(signature, i2sig, overlap_ratios):
    # abs_signature, n_sig x n_sample
    # sig2i: signature name to index
    # gene_set: target genes in a particular gene set
    #print(gene_set)
    #

    hit_indicator = (overlap_ratios > 0).astype(int)
    miss_indicator = 1 - hit_indicator
    number_hit_is = hit_indicator.sum()
    number_miss = len(signature) - number_hit_is
    #print(signature.shape, overlap_ratios.shape)

    sort_indices = np.argsort(signature, axis=0)[::-1, :]

    sorted_sig = np.take_along_axis(signature.values, sort_indices, axis=0)
    sorted_abs = np.abs(sorted_sig)
    sorted_or = np.take_along_axis(overlap_ratios[:, np.newaxis], sort_indices, axis=0)
    sum_hit_scores = np.sum(sorted_abs * sorted_or, axis=0)
    norm_hit = 1.0/sum_hit_scores.astype(float)
    norm_miss = float(1.0/number_miss)
    sorted_miss = np.take_along_axis(miss_indicator[:, np.newaxis], sort_indices, axis=0)
    #print(sorted_miss.shape)
    #print(sorted_or.shape, sorted_abs.shape, norm_hit[np.newaxis, :].shape)
    #print((sorted_or * sorted_abs * norm_hit[np.newaxis, :]).max())
    #print( (sorted_miss * norm_miss).shape)
    score = sorted_or * sorted_abs * norm_hit[np.newaxis, :] - sorted_miss * norm_miss
    running_sum = np.cumsum(score, axis=0)
    #print('--', running_sum)
    # running_sum  n_gene x n_sample

    ES, ESD, peak = get_ES_ESD(running_sum)
    le_genes = [] # get_leading_edge(i2sig, hit_indicator, ES, peak)

    return running_sum, ES, ESD, peak, le_genes

def get_ESs_null(abs_signature, number_hits, perm_n, seed): # ???
    random.seed(seed)
    es = []
    esd = []
    hit_indicator = np.zeros(len(abs_signature))
    hit_indicator[0:number_hits] = 1
    for i in range(perm_n):
        running_sum, ES, ESD, peak = get_ES_null(abs_signature, hit_indicator)
        es.append(ES)
        esd.append(ESD)
    return np.array(es), np.array(esd)


def get_ES_null(abs_signature, hit_indicator):
    np.random.shuffle(hit_indicator)
    hit_is = np.where(hit_indicator == 1)[0]
    number_hit_is = len(hit_is)
    number_miss = len(abs_signature) - number_hit_is
    sum_hit_scores = np.sum(abs_signature[hit_is])
    norm_hit = 1.0/sum_hit_scores
    norm_miss = 1.0/number_miss
    running_sum = np.cumsum(hit_indicator * abs_signature * norm_hit - (1 - hit_indicator) * norm_miss)
    ES, ESD, peak = get_ES_ESD(running_sum)
    return running_sum, ES, ESD, peak


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


def get_ES_ESD_(running_sum):
    # Find the maximum value (enrichment score, ES) and its index ES_peak
    peak = np.argmax(np.abs(running_sum))
    ES = running_sum[peak]
    # Find the maximum positive value and its index
    max_positive = running_sum[running_sum > 0].max(initial=0)  # Max positive value (or 0 if none)
    # Find the maximum negative value and its index
    max_negative = running_sum[running_sum < 0].min(initial=0)  # Max negative (or 0 if none)
    # Calculate the enrichment score difference (ES) and its index ESD_peak
    ESD = max_positive + max_negative

    return ES, ESD, peak