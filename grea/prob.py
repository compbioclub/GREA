import numpy as np
from grea import prob_gamma

def get_prob(obj):
    """
    Args:
        obj.ES: [n_term, n_obs] with the enrichment scores.
        obj.ESD: [n_term, n_obs] with the enrichment score differences.
        obj.AUC: np.ndarray of shape [n_term, n_obs]
        obj.null_ES: np.ndarray of shape [n_perm, n_term, n_obs]
        obj.null_ESD: np.ndarray of shape [n_perm, n_term, n_obs]
        obj.null_AUC: np.ndarray of shape [n_perm, n_term, n_obs]
        obj.null_nAUC: np.ndarray of shape [n_perm, n_term, n_obs]
    returns:
        obj.ES_pval: [n_term, n_obs] 
        obj.ESD_pval: [n_term, n_obs] 
        obj.AUC_pval: [n_term, n_obs]   
    """
    obj.ES_pval = _get_prob(obj.ES, obj.null_ES, prob_method='sign'+obj.prob_method)    # [n_term, n_obs]
    obj.ESD_pval = _get_prob(obj.ESD, obj.null_ESD, prob_method='sign'+obj.prob_method) # [n_term, n_obs]
    obj.AUC_pval = _get_prob(obj.AUC, obj.null_AUC, prob_method=obj.prob_method)
    obj.nAUC_pval = _get_prob(obj.nAUC, obj.null_nAUC, prob_method=obj.prob_method)

def _get_prob(obs, nulls, prob_method):
    """
        General batch p-value wrapper.
        Parameters:
            obs: np.ndarray [n_term, n_obs]
            nulls: np.ndarray [n_perm, n_term, n_obs]
            prob_method: str
        Returns:
            pvals: np.ndarray [n_term, n_obs] 
    """
    if prob_method == 'perm':
        return pred_perm_prob(obs, nulls)
    if prob_method == 'signperm':
        return pred_signperm_prob(obs, nulls)
    if prob_method == 'signgamma':
        return prob_gamma.pred_signgamma_prob(obs, nulls)
    if prob_method == 'gamma':
        return prob_gamma.pred_gamma_prob(obs, nulls)
    

def pred_perm_prob(obs, nulls):
    """
    permutation p-value.

    Parameters:
        obs: np.ndarray [n_term, n_obs]
        nulls: np.ndarray [n_perm, n_term, n_obs]

    Returns:
        pvals: np.ndarray [n_term, n_obs]
    """
    min_p = 1.0 / (nulls.shape[0] * 10)
    max_p = 1.0 - min_p

    pvals = (obs[None, :, :] <= nulls).sum(axis=0) / nulls.shape[0]
    return np.clip(pvals, min_p, max_p)


def pred_signperm_prob(obs, nulls):
    """
    signed permutation p-value.

    Parameters:
        obs: np.ndarray [n_term, n_obs]
        nulls: np.ndarray [n_perm, n_term, n_obs]

    Returns:
        pvals: np.ndarray [n_term, n_obs]
    """
    pos_mask = obs > 0
    neg_mask = obs < 0

    pvals = np.zeros_like(obs, dtype=float)

    if np.any(pos_mask):
        pos_obs = obs[pos_mask]  # shape [k]
        pos_nulls = nulls[:, pos_mask]  # shape [n_perm, k]
        pvals[pos_mask] = np.sum(pos_obs <= pos_nulls, axis=0) / nulls.shape[0]

    if np.any(neg_mask):
        neg_obs = obs[neg_mask]
        neg_nulls = nulls[:, neg_mask]
        pvals[neg_mask] = np.sum(neg_obs >= neg_nulls, axis=0) / nulls.shape[0]

    min_p = 1.0 / (nulls.shape[0] * 10)
    max_p = 1.0 - min_p
    return np.clip(pvals, min_p, max_p)