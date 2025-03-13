import pandas as pd
import scanpy as sc

def preprocess_signature(signature, group = None, FC = True, stat = 't-test',key_added = 'ttest_symptom'):

    if not FC and group is None:
        return signature
    if FC and group is not None:
        group.rename(columns={group.columns[0]: 'SampleID', group.columns[1]: 'Label'}, inplace=True)
        samples = list(signature.columns)
        group= group[group['SampleID'].isin(samples)]
        group = dict(zip(group['SampleID'], group['Label']))
        #Adata
        adata = sc.AnnData(X=signature.T)
        adata.obs_names = signature.columns
        adata.var_names = signature.index
        #group
        adata.obs['group'] = pd.Categorical([group.get(x) for x in adata.obs_names])
        #preprocessing
        sc.pp.filter_genes(adata, min_cells=3)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.layers["log1p"] = adata.X.copy()
        #DEG computation
        sc.tl.rank_genes_groups(
            adata, 
            groupby='group', 
            method=stat, 
            layer='log1p',
            key_added= key_added,
            pairwise=True  
        )
        unique_groups = adata.obs['group'].unique()
        all_results = []
        #result
        for group in unique_groups:
            res = sc.get.rank_genes_groups_df(adata, group=group, key=key_added)
            res = res[['names', 'scores']]  # Keep only 'names' and 'scores' columns
            res['group'] = group  # Add the group column to track results from each group comparison
            all_results.append(res)

        # Concatenate all group results into a single DataFrame
        signature = pd.concat(all_results, ignore_index=True)
        return signature
    else:
        return signature