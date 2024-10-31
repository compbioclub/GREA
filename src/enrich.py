import numpy as np
import random

def get_leading_edge(i2sig, hit_is, ES, peak):
    hit_is = np.array(hit_is)
    if ES > 0:
        le_genes = [i2sig[i] for i in hit_is[hit_is <= peak]]
    else:
        le_genes = [i2sig[i] for i in hit_is[hit_is >= peak]]
    return le_genes


def get_SE(abs_signature, sig2i, i2sig, gene_list):
    # abs_signature
    # sig2i: signature name to index
    # gene_list: list of genes in a particular gene set
    hit_is = [sig2i[x] for x in gene_list if x in sig2i.keys()]
    hit_indicator = np.zeros(len(abs_signature))
    hit_indicator[hit_is] = 1
    miss_indicator = 1 - hit_indicator
    number_hit_is = len(hit_is)
    number_miss = len(abs_signature) - number_hit_is
    sum_hit_scores = np.sum(abs_signature[hit_is])
    norm_hit = float(1.0/sum_hit_scores)
    norm_miss = float(1.0/number_miss)
    running_sum = np.cumsum(hit_indicator * abs_signature * norm_hit - miss_indicator * norm_miss)

    ES, ESD, peak = get_ES_ESD(running_sum)

    le_genes = get_leading_edge(i2sig, hit_is, ES, peak)

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


def get_ES_ESD(running_sum):
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