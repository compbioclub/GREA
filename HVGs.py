import scanpy as sc
import numpy as np
import pandas as pd
import os


#Input Data must have the index
#Exapmple:
# exp_matrix = pd.read_csv('data/FLu/FLu_exp_matrix.csv', index_col=0)
# enropy = pd.read_csv('data/FLu/FLu_entropy.csv', index_col=[0,1])

def HVGs(exp_matrix, n_top_genes=5000, flavor = 'seurat_v3',output_dir = 'Processed_Data', base_name = 'FLu'):

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Process expression matrix with Scanpy
    print(exp_matrix.columns)
    adata = sc.AnnData(exp_matrix.T)
    adata.obs_names = exp_matrix.columns
    adata.var_names = exp_matrix.index
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p"] = adata.X.copy()
    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor= flavor)
    adata = adata[:,adata.var['highly_variable']]
    print(adata.shape)
    exp_matrix_5000 = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)
    exp_matrix_5000.to_csv(os.path.join(output_dir, f"{base_name}_HVGs.csv"))
    gene_list = exp_matrix_5000.index
    gene_list.to_series().to_csv(os.path.join(output_dir, f"{base_name}_genelist.csv"), index=False)
    return exp_matrix_5000, gene_list


def entropy2gene(entropy,output_dir = 'Processed Data', base_name = 'FLu'):
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
    

def entropy_HVGs(entropy, gene_list, output_dir = 'Processed Data', base_name = 'FLu'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Process interaction data if available
    entropy_5000 = entropy[entropy.index.get_level_values(0).isin(gene_list) & entropy.index.get_level_values(1).isin(gene_list)]
    entropy_5000.to_csv(os.path.join(output_dir,f"{base_name}_entropy_HVGs.csv"))
    print(f"Done: {entropy}")
    return entropy_5000