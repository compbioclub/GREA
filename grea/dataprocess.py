import pandas as pd
import scanpy as sc
import os
import re

def preprocess_signature(signature, group = None, FC = True, stat = 'wilcoxon',key_added = 'wilcoxon_symptom',output_dir='Signature_divided',base_name = 'Flu'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        grouped_signature = {group: signature[signature['group'] == group] for group in signature['group'].unique()}
        for group, df in grouped_signature.items():
            df = df.drop(columns=['group'])
            safe_group = re.sub(r'[\/:*?"<>|]', '_', group)
            output_file = f"{output_dir}/{base_name}_{safe_group}.csv"
            df.to_csv(output_file, index=False)
        return grouped_signature
    else:

        return signature
    

def HVGs(exp_matrix, n_top_genes=5000, flavor = 'seurat_v3',output_dir = 'Processed_Data', base_name = 'FLu'):
 
     # Create output directory if it doesn't exist
     if not os.path.exists(output_dir):
         os.makedirs(output_dir)
     # Process expression matrix with Scanpy
     adata = sc.AnnData(exp_matrix.T)
     adata.obs_names = exp_matrix.columns
     adata.var_names = exp_matrix.index
     sc.pp.normalize_total(adata, target_sum=1e4)
     sc.pp.log1p(adata)
     adata.layers["log1p"] = adata.X.copy()
     # Identify highly variable genes
     sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor= flavor)
     adata = adata[:,adata.var['highly_variable']]
     exp_matrix_5000 = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)
     exp_matrix_5000.to_csv(os.path.join(output_dir, f"{base_name}_HVGs.csv"))
     gene_list = exp_matrix_5000.index
     gene_list.to_series().to_csv(os.path.join(output_dir, f"{base_name}_genelist.csv"), index=False)
     return exp_matrix_5000, gene_list
 
 
def entropy2gene(entropy,output_dir = 'Processed_Data', base_name = 'FLu'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Calculate entropy-based
    entropy.set_index(['node1','node2'], inplace=True)
    node1 = entropy.groupby('node1').mean()
    node2 = entropy.groupby('node2').mean()
    inter_node = pd.concat([node1,node2],axis=0)
    inter_node.rename_axis('Gene', inplace=True)
    single_genes = inter_node.groupby('Gene').mean()
    single_genes.to_csv(os.path.join(output_dir,f"{base_name}_single_genes.csv"))
    print(f"Done: {entropy}")
    return single_genes
    

def entropy_HVGs(entropy, gene_list, output_dir = 'Processed_Data', base_name = 'FLu'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Process interaction data if available
    entropy.set_index(['node1','node2'], inplace=True)
    entropy_5000 = entropy[entropy.index.get_level_values(0).isin(gene_list) & entropy.index.get_level_values(1).isin(gene_list)]
    entropy_5000.to_csv(os.path.join(output_dir,f"{base_name}_entropy_HVGs.csv"))
    print(f"Done: {entropy}")
    return entropy_5000