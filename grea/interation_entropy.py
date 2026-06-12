import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix
from pyseat.SEAT import SEAT
import pickle
import os
import pickle
from itertools import permutations
import pymannkendall as mk
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
#import sys
#sys.setrecursionlimit(10000)

pd.options.mode.copy_on_write = True
import numpy as np

from grea.grea import pheno_prerank_enrich

class FREA():

    def __init__(self, adata, bg_net=None, bg_net_score_cutoff=850,
                 n_threads=1,
                 n_hvg=1000,n_pcs=30,
                 dataset='test',
                 out_dir='./out'
                 ):

        adata = adata.copy()
        self.preprocess_adata(adata, n_hvg=n_hvg, n_pcs=n_pcs)
        adata = adata[:, adata.var['highly_variable']]
        adata.obs['i'] = range(adata.shape[0])
        adata.var['i'] = range(adata.shape[1])
        self.adata = adata
        self.n_threads = n_threads
        self.n_hvg = n_hvg
        self.dataset = dataset
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.out_dir = out_dir

    def preprocess_adata(self, adata, n_hvg=5000, random_state=0, n_pcs=30, n_neighbors=10):
        #sc.pp.scale(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='cell_ranger')
        sc.tl.pca(adata)
        sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
        sc.tl.umap(adata, random_state=random_state)

    def build_hierarchy(self, groupby=None, n_neighbors=10, n_top=1000,
                        strategy='top_down',
                        layer='log1p'):

        if groupby is None:
            groups = 'all'
        else:
            groups = self.adata.obs[groupby].unique()

        knn_m = csr_matrix((self.n_hvg, self.n_hvg), dtype=float)

        self.seat_dict = {}
        for group in groups:
            if group == 'all':
                X = self.adata.layers[layer].T
            else:
                X = self.adata[self.adata.obs[groupby] == group].layers[layer].T
            seat = SEAT(affinity="gaussian_kernel", #"gaussian_kernel",precomputed
                        sparsification="knn_neighbors", #"knn_neighbors",affinity
                        objective="SE",
                        n_neighbors=n_neighbors,
                        strategy=strategy,
                        verbose=False)                
            seat.fit_predict(X)   
            knn_m += self.get_top_n_matrix(seat.aff_m, n_top)
            self.seat_dict[group] = seat
    
        self.adata.varm['bg_net'] = knn_m

    def get_top_n_matrix(self, aff_m, n_top = 1000, *args, **kwds):
        # returns only the upper part of the array, automatically setting all diagonal and lower elements to 0
        aff_m = np.triu(aff_m, k=1)  
        flat_indices = np.argpartition(aff_m.ravel(), -n_top)[-n_top:]
        row_indices, col_indices = np.unravel_index(flat_indices, aff_m.shape)
        data = np.ones(len(row_indices), dtype=np.int8)
        matrix = coo_matrix((data, (row_indices, col_indices)), shape=aff_m.shape)
        matrix = matrix.tocsr()      
        return matrix  

    def init_idata(self, groupby=None, batch_size=1000, n_jobs=-1):
        # n_obs, n_iteration
        adata = self.adata
        n_obs = adata.shape[0]
        bg_net = adata.varm['bg_net']
        row, col = bg_net.nonzero()
        n_iteration = bg_net.count_nonzero()
        idata = ad.AnnData(np.zeros((n_obs, n_iteration)))
        idata.obs = adata.obs.copy()
        idata.var_names = adata.var_names[row].astype(str) + '_' + adata.var_names[col].astype(str)
        idata.var['var1'] = adata.var_names[row]
        idata.var['var2'] = adata.var_names[col]
        idata.var['var1_i'] = row
        idata.var['var2_i'] = col
        idata.var['i'] = range(n_iteration)
        self.idata = idata
        print(f'Init idata (obs, interation) with shape {idata.shape}.')

        A = self.adata.X
        bg_mask = self.adata.varm['bg_net'].todense()
        # ij is A[i,j], ik is A[i,k], jk is mask[j,k]
        # Returns B[i,j,k] already masked
        bg_mask_B = np.einsum('ij,ik,jk->ijk', A, A, bg_mask)
        idata.X = bg_mask_B[:, row, col]
        idata.layers[f'prod'] = idata.X

        # entropy
        if groupby is None:
            groups = 'all'
        else:
            groups = adata.obs[groupby].unique()
            group_df = adata.obs[groupby]

        vol_dict = {}
        entropy_m = np.zeros(idata.shape)
        for group in groups:        
            if group == 'all':
                group_mask = np.ones(n_obs)
            else:
                group_mask = (adata.obs['symptom'] == group).astype(int).to_numpy()
            tree = self.seat_dict[group].se_tree
            lca_list = []
            for u, v in zip(idata.var['var1_i'],idata.var['var2_i']):
                for n_id in list(tree.node_list.keys())[::-1]:
                    node = tree.node_list[n_id]
                    if u in node.vs and v in node.vs:
                        lca_list.append(n_id)
                        break
            self.idata.var[f'{group}_lca_id'] = lca_list        
            # get the gene volumne
            for n_id in set(idata.var['var1_i'].to_list()) | set(idata.var['var2_i'].to_list()):
                # bg is upper tria
                vol_dict[(group, n_id)] = bg_mask_B[:, n_id, :].sum(axis=1) + bg_mask_B[:, :, n_id].sum(axis=1)
            # get the node volumne
            for n_id in tree.node_list.keys():
                node = tree.node_list[n_id]
                idx = node.vs
                vol_dict[(group, n_id)] = bg_mask_B[:, idx, :].sum(axis=(1,2)) + bg_mask_B[:, :, idx].sum(axis=(1,2))
            # calculate entropy            
            eps = 0.000001
            res = []
            for u, v, lca_id in zip(idata.var['var1_i'],idata.var['var2_i'],idata.var[f'{group}_lca_id']):
                u_vol = vol_dict[(group, u)] 
                v_vol = vol_dict[(group, v)] 
                lca_vol = vol_dict[(group, lca_id)] 
                p_u_v = u_vol/lca_vol
                p_v_u = v_vol/lca_vol
                entropy = -(p_u_v*np.log(p_u_v+eps) + p_v_u*np.log(p_v_u+eps))
                res.append(entropy*group_mask)
            entropy_m += np.array(res).T
        self.vol_dict = vol_dict
        idata.layers[f'entropy'] = entropy_m
        idata.layers[f'entropy_prod'] = entropy_m * idata.layers[f'prod']

    def test_DER(self, groupby, target_group=None, 
                 test_method="wilcoxon", method='prod'):
        mytarget_group = target_group
        groups = self.idata.obs[groupby].unique()
        self.groups = groups
        self.groupby = groupby
        idata = self.idata

        df_list = []
        mean_df = pd.DataFrame(index=idata.var_names)
        for ref_group in groups:
            group_X = idata[idata.obs[groupby] == ref_group].layers[f'{method}']
            mean_df[ref_group] = np.nansum(group_X, axis=0)
        for ref_group in groups:
            sc.tl.rank_genes_groups(idata, layer=f'{method}',
                                        groupby=groupby, reference=ref_group,
                                        method=test_method)
            for target_group in groups:
                if mytarget_group is not None and mytarget_group != target_group:
                    continue
                if ref_group == target_group:
                    continue

                df = sc.get.rank_genes_groups_df(idata, group=target_group)
                df['ref_group'] = ref_group
                df['target_group'] = target_group
                df['method'] = method
                df['ref_group_mean'] = df['names'].apply(lambda x: mean_df[ref_group].to_dict()[x])
                df['target_group_mean'] = df['names'].apply(lambda x: mean_df[target_group].to_dict()[x])
                idata.uns[f'{method}_rank_genes_groups_{ref_group}_{target_group}'] = idata.uns['rank_genes_groups']
                df_list.append(df)

        df = pd.concat(df_list)
        idata.uns[f'{method}_rank_genes_groups_df'] = df
        return df

    def get_DER(self, target_group=None, n_top_relations=None, method='prod', 
                p_adjust=True, p_cutoff=0.05, fc_cutoff=1, sortby='scores',
                ):
        mytarget_group = target_group
        idata = self.idata
        df_list = []

        for ref_group, target_group in permutations(self.groups, 2):
            if mytarget_group is not None and mytarget_group != target_group:
                continue
            df = sc.get.rank_genes_groups_df(idata, group=target_group,
                                             key=f'{method}_rank_genes_groups_{ref_group}_{target_group}')
            df['target_group'] = target_group
            df['ref_group'] = ref_group
            df['method'] = method
            if p_adjust:
                p_method = 'pvals_adj'
            else:
                p_method = 'pvals'

            df['DER'] = (df[p_method] < p_cutoff) & (df['logfoldchanges'] > fc_cutoff)
            df_list.append(df)

        df = pd.concat(df_list)
        tmp = df[df['DER']][['names', 'target_group', 'method']]
        count_df = tmp.value_counts()
        count_df = count_df[count_df == (len(self.groups) - 1)]
        count_df = count_df.reset_index()
        for target_group in self.groups:
            if mytarget_group is not None and mytarget_group != target_group:
                continue
            if target_group not in count_df['target_group'].unique():
                relations = []
                tmp_df = pd.DataFrame()
            else:
                tmp = count_df[(count_df['target_group'] == target_group) & (count_df['method'] == method)]
                relations = tmp.names.tolist()
                tmp_df = df[(df['target_group'] == target_group) & df['names'].isin(relations)]
                if sortby in ['logfoldchanges', 'scores']:
                    tmp_df = tmp_df.sort_values(by=['DER', sortby], ascending=False)
                else:
                    tmp_df = tmp_df.sort_values(by=['DER', sortby], ascending=[False, True])
                relations = tmp_df.names.drop_duplicates().tolist()
                if n_top_relations is not None:
                    relations = relations[: n_top_relations]
            idata.uns[f'{method}_{self.groupby}_{target_group}_DER'] = relations
            idata.uns[f'{method}_{self.groupby}_{target_group}_DER_df'] = tmp_df
            fn = f'{self.out_dir}/{self.dataset}_{method}_{self.groupby}_{target_group}_DER.csv'
            tmp_df.to_csv(fn, index=False)
            print(f'[Output] The differential expressed relation (DER) {len(relations)} statistics are saved to:\n{fn}')
        return

    def pheno_prerank_enrich(self, ref_group, target_group, 
        method = 'prod', sortby='scores', 
        libraries=['KEGG_2021_Human',
                   'GO_Molecular_Function_2023', 'GO_Cellular_Component_2023', 'GO_Biological_Process_2023',
                   'MSigDB_Hallmark_2020'],                                     
        min_size=5, max_size=1000, n_perm=1000, seed=0, prob_method='perm'):

        df = self.idata.uns[f'{method}_rank_genes_groups_df']

        if ref_group is not None:
            df = df[df['ref_group'] == ref_group]
        if target_group is not None:
            df = df[df['target_group'] == target_group]
        df['target vs. ref'] = df['target_group'] + ' vs. ' + df['ref_group']
        df_list = []
        for pheno in df['target vs. ref'].unique():
            tmp = df[df['target vs. ref'] == pheno]
            tmp = tmp[['names',sortby]]
            tmp.index = tmp['names']
            del tmp['names']
            tmp.columns = [pheno]
            df_list.append(tmp)
        pheno_df = pd.concat(df_list, axis=1)
        sig_sep = '_'
        ens_obj = pheno_prerank_enrich(
            pheno_df, libraries, n_perm=n_perm, prob_method=prob_method, sig_sep=sig_sep,
            min_size=min_size, max_size=max_size, seed=seed)
        return ens_obj  

