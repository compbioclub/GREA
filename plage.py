import numpy as np
from scipy.linalg import svd
from scipy import stats

def plage_analysis(sig_name, sig_val, library, sig_sep=','):
    """
    Perform PLAGE (Pathway Level Analysis of Gene Expression) analysis.
    
    Parameters:
    -----------
    sig_name : np.ndarray
        Gene names matrix, shape (n_genes, n_samples)
    sig_val : np.ndarray
        Expression value matrix, shape (n_genes, n_samples)
    library : dict
        Dictionary of pathway/gene set definitions
    sig_sep : str, optional
        Separator for multiple gene names in sig_name
        
    Returns:
    --------
    pd.DataFrame
        Results containing pathway activity scores and statistics
    """
    from collections import defaultdict
    import pandas as pd
    
    n_sample = sig_val.shape[1]
    results = []
    
    for pathway_name, pathway_genes in library.items():
        # Get overlap between pathway genes and expression matrix genes
        gene_indices = []
        for i, names in enumerate(sig_name[:,0]):  # Using first column for gene names
            gene_set = set(names.split(sig_sep))
            if any(gene in pathway_genes for gene in gene_set):
                gene_indices.append(i)
                
        if len(gene_indices) < 3:  # Skip if too few genes
            continue
            
        # Extract expression values for pathway genes
        pathway_expression = sig_val[gene_indices, :]
        
        # Standardize genes (center and scale)
        pathway_expression = stats.zscore(pathway_expression, axis=1)
        
        try:
            # Perform SVD
            U, s, Vt = svd(pathway_expression, full_matrices=False)
            
            # Get pathway activity score (first right singular vector)
            activity_score = Vt[0, :]
            
            # Standardize activity score
            activity_score = stats.zscore(activity_score)
            
            # Calculate statistics
            t_stat, p_val = stats.ttest_1samp(activity_score, 0.0)
            
            # Store results for this pathway
            for i in range(n_sample):
                results.append({
                    'Pathway': pathway_name,
                    'Sample': i,
                    'Activity_Score': activity_score[i],
                    'P_Value': p_val,
                    'T_Statistic': t_stat,
                    'n_genes': len(gene_indices)
                })
                
        except Exception as e:
            print(f"Warning: Failed to process pathway {pathway_name}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add multiple testing correction
    if len(results_df) > 0:
        unique_pathways = results_df['Pathway'].unique()
        p_values = [results_df[results_df['Pathway'] == p]['P_Value'].iloc[0] 
                   for p in unique_pathways]
        
        # Calculate FDR
        from statsmodels.stats.multitest import multipletests
        fdr_values = multipletests(p_values, method='fdr_bh')[1]
        
        # Add FDR to results
        fdr_dict = dict(zip(unique_pathways, fdr_values))
        results_df['FDR'] = results_df['Pathway'].map(fdr_dict)
    
    return results_df

def combine_grea_plage(sig_name, sig_val, library, 
                      sig_sep=',', method='KS', 
                      n_perm=1000, prob_method='perm',
                      include_plage=True):
    """
    Combine GREA and PLAGE analyses.
    
    Parameters:
    -----------
    sig_name : np.ndarray
        Gene names matrix
    sig_val : np.ndarray
        Expression value matrix
    library : dict
        Gene set library
    sig_sep : str, optional
        Separator for multiple gene names
    method : str, optional
        Method for GREA ('KS' or 'RC')
    n_perm : int, optional
        Number of permutations for GREA
    prob_method : str, optional
        Probability calculation method
    include_plage : bool, optional
        Whether to include PLAGE analysis
        
    Returns:
    --------
    tuple
        (grea_results, plage_results)
    """
    # Perform GREA analysis
    grea_results = sig_enrich(sig_name, sig_val, library, 
                            sig_sep=sig_sep, method=method,
                            n_perm=n_perm, prob_method=prob_method)
    
    # Perform PLAGE analysis if requested
    plage_results = None
    if include_plage:
        plage_results = plage_analysis(sig_name, sig_val, library, sig_sep=sig_sep)
    
    return grea_results, plage_results