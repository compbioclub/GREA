import os
import numpy as np
import torch

from grea import enrich_signal_np

def _csr2coo3(csr, n_term, n_sig, n_obs):
    coo2 = csr.to_sparse_coo().coalesce()
    rows, cols = coo2.indices()
    vals       = coo2.values()
    t = rows // n_obs
    o = rows %  n_obs
    i = cols
    idx3 = torch.stack([t, i, o], dim=0)
    return torch.sparse_coo_tensor(
        idx3, vals, size=(n_term, n_sig, n_obs)
        ).coalesce()


def get_overlap(obj):
    """ This version obj.sig_names is the same for all obs
    Args:
        obj.sig_names: np.ndarray of shape [n_sig, n_obs], strings of sig_name separated by sig_sep
        obj.sig_sep: separator used in sig_name
        
    Returns:
        obj.term_genes_list: List[List[str]], length = n_term, each is a set of genes
        obj.overlap_ratios: torch.sparse.FloatTensor COO [n_term, n_sig, n_obs]
        obj.n_hits: np.ndarray of shape [n_term, n_obs] # number of sigs hit in the obs
    """
    enrich_signal_np.get_overlap(obj)

    # --- 1) overlap_ratios: shape [n_term, n_sig, n_obs] → sparse COO ---
    # assume obj.overlap_ratios is a NumPy array (or memmap) of dtype float32
    arr = obj.overlap_ratios  # shape = (n_term, n_sig, n_obs)
    # 1.a) find all non-zero coords in NumPy
    coords = np.nonzero(arr)             # a tuple of three 1-D arrays: (i, j, k)
    # 1.b) build the 2D index matrix Torch expects: shape = [ndim, nnz]
    #    stack them into a (3, nnz) array, then LongTensor
    indices = torch.LongTensor(np.vstack(coords))  
    #    indices[:, p] = (i_p, j_p, k_p) for the p-th non-zero
    # 1.c) grab the values at those coords
    values = torch.FloatTensor(arr[coords])  
    # 1.d) build the sparse tensor
    sparse_overlap = torch.sparse_coo_tensor(
        indices, values, size=arr.shape, dtype=torch.float32
    ).coalesce()  # coalesce to sum duplicate indices (optional)
    # Now sparse_overlap is a torch.sparse.FloatTensor


    # --- 2) n_hits: shape [n_term, n_obs] → you can choose COO or CSR ---
    hits = obj.n_hits  # dtype=int32 or int64

    # Option A) COO (same pattern as above):
    '''
    coords2 = np.nonzero(hits)
    indices2 = torch.LongTensor(np.vstack(coords2))
    values2  = torch.LongTensor(hits[coords2])
    sparse_hits = torch.sparse_coo_tensor(
        indices2,
        values2,
        size=hits.shape,
        dtype=torch.int64
    ).coalesce()
    '''
    # Option B) CSR (more efficient for 2D row‐indexed access):
    #   build crow_indices (row‐pointer) and col_indices + values
    row_inds, col_inds = np.nonzero(hits)
    vals = hits[row_inds, col_inds]
    #  compute crow_indices: length = n_term+1
    n_term, n_obs = hits.shape
    # count how many nonzeros per row
    row_counts = np.bincount(row_inds, minlength=n_term)
    crow_indices = np.empty(n_term+1, dtype=np.int64)
    crow_indices[0] = 0
    crow_indices[1:] = np.cumsum(row_counts)
    csr_hits = torch.sparse_csr_tensor(
        torch.from_numpy(crow_indices),
        torch.from_numpy(col_inds.astype(np.int64)),
        torch.from_numpy(vals.astype(np.int64)),
        size=hits.shape,
        dtype=torch.int64
    )

    obj.overlap_ratios = sparse_overlap.to(obj.device)
    obj.n_hits = csr_hits.to(obj.device)

def get_sorted(obj):
    """
    Args:
        obj.sig_vals is a torch.Tensor of shape [n_sig, n_obs] (on CPU or GPU)
    Returns:
        obj.sort_indices      : LongTensor [n_sig, n_obs]
        #obj.sig_rank          : LongTensor [n_sig, n_obs]
        obj.sorted_sig_vals   : Tensor     [n_sig, n_obs]
        obj.sorted_abs_vals   : Tensor     [n_sig, n_obs]
    """
    # 1) sort descending along dim=0
    #    sort_vals: [n_sig, n_obs],  sort_indices: LongTensor [n_sig, n_obs]
    obj.sig_vals = torch.from_numpy(obj.sig_vals).float().to(obj.device)
    
    obj.sorted_sig_vals, obj.sort_indices = obj.sig_vals.sort(dim=0, descending=True)
    obj.sorted_abs_vals  = obj.sorted_sig_vals.abs()

    # 2) build rank‐matrix: for each column j, the element at row i
    #    of the ORIGINAL sig_vals lands at rank position k = sig_rank[i,j]
    #    We invert the permutation `sort_indices`:
    #    scatter the ranks 0…n_sig-1 BACK into the positions given by sort_indices
    #rank_values = torch.arange(n_sig, device=obj.sig_vals.device, dtype=torch.long
    #                          ).unsqueeze(1).expand(n_sig, n_obs)
    #sig_rank = torch.empty_like(sort_indices).scatter_(0, sort_indices, rank_values)


def get_running_sum(obj):
    """
    Fully vectorized version of running sum.
    Input:
        obj.sorted_abs_vals:    Tensor      [n_sig, n_obs]
        obj.overlap_ratios:     COO Tensor  [n_term, n_sig, n_obs]
        obj.sort_indices:       LongTensor  [n_sig, n_obs]
    Output:
        obj.ks_rs: [n_term, n_sig, n_obs]
        obj.rc_rs: [n_term, n_sig, n_obs]
        obj.sorted_or: [n_term, n_sig, n_obs]
    """
    obj.ks_rs, obj.rc_rs, obj.nrc_rs, obj.sorted_or = _get_running_sum(obj.sorted_abs_vals, obj.overlap_ratios, obj.sort_indices)


def _get_running_sum(sorted_abs_vals, overlap_ratios, sort_indices):
    """
    Fully vectorized version of running sum.
    Input:
        sorted_abs_vals:    Tensor      [n_sig, n_obs]
        overlap_ratios:     COO Tensor  [n_term, n_sig, n_obs]
        sort_indices:       LongTensor  [n_sig, n_obs]
    Output:
        ks_rs:      COO Tensor  [n_term, n_sig, n_obs]
        rc_rs:      COO Tensor  [n_term, n_sig, n_obs]
        nrc_rs:     COO Tensor  [n_term, n_sig, n_obs]
        sorted_or:  COO Tensor  [n_term, n_sig, n_obs]
    """
    # dims
    n_term, n_sig, n_obs = overlap_ratios.shape
    R = n_term * n_obs
    C = n_sig
    eps = torch.finfo(sorted_abs_vals.dtype).eps

    # 1) coalesce input
    sp = overlap_ratios.coalesce()
    (T_idx, I_idx, O_idx), OR_vals = sp.indices(), sp.values()
    # 2) build inverse permutation sig_rank so that
    #    sorted_or[t,i_new,o] = overlap_ratios[t,i_old,o]
    #    where i_new = sig_rank[i_old,o]
    arange_sig = torch.arange(n_sig, device=sort_indices.device).unsqueeze(1).expand(n_sig, n_obs)
    sig_rank = torch.empty_like(sort_indices).scatter_(0, sort_indices, arange_sig)

    # 3) re‐index to get sorted_or in COO
    new_i = sig_rank[I_idx, O_idx]
    idx3 = torch.stack([T_idx, new_i, O_idx], dim=0)
    sorted_or_sp = torch.sparse_coo_tensor(
        idx3, OR_vals, size=(n_term, n_sig, n_obs)
    ).coalesce()

    # 4) flatten into 2D‐CSR: rows = t*n_obs + o, cols = i
    so = sorted_or_sp
    (t2, i2, o2), v2 = so.indices(), so.values()
    row2 = t2 * n_obs + o2
    col2 = i2
    # sort by (row, col) so that each row's entries are contiguous
    key = row2.to(torch.int64)*(C+1) + col2.to(torch.int64)
    perm = torch.argsort(key)
    row2 = row2[perm]; col2 = col2[perm]; v2 = v2[perm]

    # build crow
    row_counts = torch.bincount(row2, minlength=R)
    crow = torch.empty(R+1, dtype=torch.long, device=OR_vals.device)
    crow[0] = 0
    crow[1:] = torch.cumsum(row_counts, dim=0)

    # 5) compute number_hit, number_miss, norm_hit, norm_miss
    number_hit_flat = row_counts                       # [R]
    number_miss_flat = C - number_hit_flat
    norm_hit_flat = number_hit_flat.to(torch.float32)
    # we need sum_hit_scores = Σ_i sorted_or * abs
    abs_part   = sorted_abs_vals[col2, (row2 % n_obs)]  # [nnz]
    sumhit_flat = torch.bincount(
        row2, weights=(v2 * abs_part), minlength=R
    )
    # avoid zeros
    sumhit_flat = sumhit_flat + eps
    norm_hit_flat = 1.0 / sumhit_flat                   # [R]
    # miss‐norm
    norm_miss_flat = torch.zeros_like(norm_hit_flat)
    nz = number_miss_flat > 0
    norm_miss_flat[nz] = 1.0 / number_miss_flat[nz].float()

    # 6) form the three value arrays in CSR‐order
    vals_rc  = v2 * abs_part                            # raw cumulative part
    nh_flat   = norm_hit_flat[row2]                     # expand to each nnz
    nm_flat   = norm_miss_flat[row2]
    vals_nrc = vals_rc * nh_flat
    vals_ks  = vals_nrc.clone()                         # ks on hit‐part only

    # 7) row‐wise in-place prefix‐sum over each of vals_rc, vals_nrc, vals_ks
    def csr_prefix(crow, vals):
        out = vals.clone()
        for r in range(R):
            s, e = int(crow[r]), int(crow[r+1])
            if e> s:
                out[s:e] = torch.cumsum(out[s:e], dim=0)
        return out

    vals_rc_cum  = csr_prefix(crow, vals_rc)
    vals_nrc_cum = csr_prefix(crow, vals_nrc)
    vals_ks_cum  = csr_prefix(crow, vals_ks)

    # 8) pack back into sparse_csr_tensors of shape [R, C]
    rc_csr  = torch.sparse_csr_tensor(crow, col2, vals_rc_cum,  size=(R, C))
    nrc_csr = torch.sparse_csr_tensor(crow, col2, vals_nrc_cum, size=(R, C))
    ks_csr  = torch.sparse_csr_tensor(crow, col2, vals_ks_cum,  size=(R, C))

    # 9) convert each 2D‐CSR → 3D‐COO [n_term, n_sig, n_obs]

    rc_rs  = _csr2coo3(rc_csr, n_term, n_sig, n_obs)
    nrc_rs = _csr2coo3(nrc_csr, n_term, n_sig, n_obs)
    ks_rs  = _csr2coo3(ks_csr, n_term, n_sig, n_obs)

    return ks_rs, rc_rs, nrc_rs, sorted_or_sp

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
        ks_rs:      COO Tensor  [n_term, n_sig, n_obs]
    Returns:
        ES:     Tensor  [n_term, n_obs] with the enrichment scores at the peak
        ESD:    Tensor  [n_term, n_obs] with the enrichment score differences
        peak:   LongTensor   [n_term, n_obs] with the peak indices along dim=1
    """
    # Convert to dense for axis-wise reductions
    # (if n_sig is very large this may be memory-heavy,
    #  but PyTorch does not yet support argmax/gather on sparse dims)
    ks = ks_rs.coalesce().to_dense()           # [T, S, O]
    # T = n_term, S = n_sig, O = n_obs

    # 1) Find peak = argmax(abs(ks), dim=1) → [T, O]
    abs_ks = ks.abs()
    peak = abs_ks.argmax(dim=1)                # LongTensor [T, O]
    # 2) Gather ES = ks[t, peak[t,o], o]
    #    peak.unsqueeze(1): [T,1,O] so we gather along dim=1 → result [T,1,O]
    ES = ks.gather(1, peak.unsqueeze(1)).squeeze(1)  # [T, O]
    # 3) Compute max_positive = max(ks where ks>0 else 0, dim=1)
    zeros = torch.zeros_like(ks)
    positive = torch.where(ks > 0, ks, zeros)
    max_positive, _ = positive.max(dim=1)     # [T, O]
    # 4) Compute max_negative = min(ks where ks<0 else 0, dim=1)
    negative = torch.where(ks < 0, ks, zeros)
    max_negative, _ = negative.min(dim=1)     # [T, O]
    # 5) Enrichment score difference
    ESD = max_positive + max_negative        # [T, O]
    return ES, ESD, peak


def get_AUC(obj):
    """
    Calculate AUC for each (n_term, n_obs) based on running sum matrix.
    Args:
        obj.rc_rs:  COO Tensor [n_term, n_sig, n_obs]
        obj.nrc_rs: COO Tensor [n_term, n_sig, n_obs]
    Returns:
        obj.RC_AUC:     COO Tensor [n_term, n_obs]
        obj.RC_nAUC:    COO Tensor [n_term, n_obs]
        obj.nRC_AUC:    COO Tensor [n_term, n_obs]
    """
    obj.RC_AUC = _get_AUC(obj.rc_rs)
    # 2) normalized AUC: divide by n_hits where non‐zero
    #    cast n_hits to float for safe division
    nh = obj.n_hits.to(obj.RC_AUC.dtype).to_dense()
    obj.RC_nAUC = torch.where(
        nh != 0, obj.RC_AUC / nh,
        torch.zeros_like(obj.RC_AUC)) 
    obj.nRC_AUC = _get_AUC(obj.nrc_rs)

def _get_AUC(rc_rs):
    """
    Calculate AUC for each (n_term, n_obs) based on running sum matrix.
    Args:
        rc_rs:  COO Tensor [n_term, n_sig, n_obs]
    Returns:
        AUC:    COO Tensor [n_term, n_obs]
    """
    # 1) unpack dims
    n_term, n_sig, n_obs = rc_rs.shape
    # 2) densify (we need axis‐wise reductions)
    rs = rc_rs.coalesce().to_dense()               # [T, S, O]
    # 3) mask NaNs → treat them as zero in the sum
    valid = ~torch.isnan(rs)
    rs_clean = torch.where(valid, rs, torch.zeros_like(rs))
    # 4) sum over the signature axis
    sum_clean = rs_clean.sum(dim=1)                # [T, O]
    # 5) check if any valid per (term,obs)
    any_valid = valid.any(dim=1)                   # [T, O]
    # 6) compute AUC = max(0, (sum_clean - 0.5) / n_sig) or NaN if no valid
    raw = (sum_clean - 0.5) / float(n_sig)
    auc = torch.where(any_valid,
                      torch.clamp(raw, min=0.0),
                      torch.full_like(raw, float('nan')))
    return auc

