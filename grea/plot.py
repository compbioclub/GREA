import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.stats import gamma
from matplotlib.patches import Rectangle
import enrich as enrich
import sigtest as sigtest
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy import stats


def running_sum(sig_name, sig_val, geneset, library, result=None, compact=False, center=True, interactive_plot=False, plot_type="ES", method="KS", sig_sep=',', cmap='YlGnBu'):
    """
    Plot the running sum for a given geneset and signature.

    Parameters:
    signature (array-like): The gene expression signature to analyze.
    geneset (str): The name of the gene set for a gene set in the library.
    library (array-like): The gene set library to use for enrichment analysis.
    result (array-like, optional): A precomputed enrichment result. Default is None.
    compact (bool, optional): If True, return a compact representation of the running sum plot for better readability in small plots. Default is False.
    center (bool, optional): Center signature values. This is generally a good idea.
    
    Returns:
    figure: The running sum plot for the given geneset and signature.
    """
    result = result.copy()
    result = result.set_index(['Term'])
    if not interactive_plot:
        plt.ioff()

    if isinstance(cmap, str):
        cm = plt.cm.get_cmap(cmap)
    else:
        cm = cmap 
    
    norm = mpl.colors.Normalize(vmin=0, vmax=1)


    sig_name = np.asarray(sig_name)
    sig_val = np.asarray(sig_val)

    if sig_val.ndim == 1:
        sig_val = sig_val.reshape(-1, 1)

    if sig_val.ndim != 2:
        raise ValueError("signature must be a 2D array.")
    
    n_genes, n_samples = sig_val.shape

    if sig_name.ndim == 1:
        sig_name = sig_name.reshape(-1, 1)
    
    if sig_name.shape[0] != n_genes:
        raise ValueError("Number of genes in sig_name must match the number of genes in sig_val.")
    
    if sig_name.shape[1] == 1:
        sig_name = np.repeat(sig_name, n_samples, axis=1)
    elif sig_name.shape[1] != n_samples:
        raise ValueError("If sig_name has multiple columns, the number of columns must match the number of samples in sig_val.")
    
    figures = [] 

    for i in range(n_samples):
        
        sig_name_i = sig_name[:, i]
        sig_val_i = sig_val[:, i].astype(float)
        if center:
            sig_val_i = sig_val_i - np.mean(sig_val_i)

        sig_val[:, i] = sig_val_i

        lib_sigs = set(library.get(geneset, []))
        
        if len(lib_sigs) == 0:
            raise ValueError(f"The geneset '{geneset}' is not found in the provided library.")
        
        overlap_ratios, hit_n = enrich.get_overlap(sig_name_i[:, np.newaxis],lib_sigs, sig_sep)
        if overlap_ratios.ndim == 1:
            overlap_ratios = overlap_ratios[:, np.newaxis]

        sort_indices = np.argsort(sig_val_i)[::-1]
        if sort_indices.ndim == 1:
            sort_indices = sort_indices[:, np.newaxis]

        sorted_sig = sig_val_i[sort_indices]
        sorted_abs = np.abs(sorted_sig)

        obs_rs = enrich.get_running_sum_aux(sorted_abs, overlap_ratios, sort_indices, method=method) 
        es,esd, *_ = enrich.get_ES_ESD(obs_rs)
        AUC = enrich.get_AUC(obs_rs)
        obs_rs = list(obs_rs[:,0])

        fig = plt.figure(figsize=(7,5),facecolor='white',edgecolor='black')
        
        if compact:
            gs = fig.add_gridspec(5, 11, wspace=0, hspace=0)
            ax1 = fig.add_subplot(gs[0:4, 0:11], facecolor='white')
        else:
            gs = fig.add_gridspec(12, 11, wspace=0, hspace=0)
            ax1 = fig.add_subplot(gs[0:7, 0:11],facecolor='white')

        for spine in ax1.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

        if compact:
            ax1.plot(obs_rs, color=(0,1,0), lw=5)
            ax1.tick_params(labelsize=24)
        else:
            ax1.plot(obs_rs, color=(0,1,0), lw=3)
            ax1.tick_params(labelsize=16)
        plt.xlim([0, len(obs_rs)])
        
        nn = np.argmax(np.abs(obs_rs))  
        ax1.vlines(x=nn, ymin=np.min(obs_rs), ymax=np.max(obs_rs),linestyle = ':', color="red")
        
        # if plot_type == "ES":
        #     if result is not None and geneset in result.index:
        #         try:
                    
        #             if isinstance(result.index, pd.MultiIndex):
        #                 nes = result.xs((geneset, i), level=('Term', 'Sample'))['nes']
        #             else:
        #                 nes = result.loc[geneset, 'nes']
        #                 if isinstance(nes, pd.Series):
        #                     nes = nes.iloc[i] if i < len(nes) else nes.iloc[0]
                    
        #             if isinstance(nes, pd.Series):
        #                 nes = nes.iloc[0]
                    
        #             label_text = f"NES={nes:.3f}"
        #             va = 'bottom' if max(obs_rs) > abs(min(obs_rs)) else 'top'
        #             text_y = max(obs_rs) if va == 'bottom' else min(obs_rs)
        #         except Exception as e:
                    
        #             es_value = float(es[0]) if isinstance(es, np.ndarray) else float(es)
        #             label_text = f"ES={es_value:.3f}"
        #             va = 'bottom' if max(obs_rs) > abs(min(obs_rs)) else 'top'
        #             text_y = max(obs_rs) if va == 'bottom' else min(obs_rs)
        #     else:
        #         es_value = float(es[0]) if isinstance(es, np.ndarray) else float(es)
        #         label_text = f"ES={es_value:.3f}"
        #         va = 'bottom' if max(obs_rs) > abs(min(obs_rs)) else 'top'
        #         text_y = max(obs_rs) if va == 'bottom' else min(obs_rs)



        # elif plot_type == "ESD":
        #     if result is not None and geneset in result.index:
        #         try:
                    
        #             if isinstance(result.index, pd.MultiIndex):
        #                 nesd = result.xs((geneset, i), level=('Term', 'Sample'))['nesd']
        #             else:
        #                 nesd = result.loc[geneset, 'nesd']
        #                 if isinstance(nesd, pd.Series):
        #                     nesd = nesd.iloc[i] if i < len(nesd) else nesd.iloc[0]
                    
        #             if isinstance(nesd, pd.Series):
        #                 nesd = nesd.iloc[0]
                    
        #             label_text = f"NESD={nesd:.3f}"
        #             va = 'bottom' if max(obs_rs) > abs(min(obs_rs)) else 'top'
        #             text_y = max(obs_rs) if va == 'bottom' else min(obs_rs)
        #         except Exception as e:
                    
        #             esd_value = float(esd[0]) if isinstance(esd, np.ndarray) else float(esd)
        #             label_text = f"ESD={esd_value:.3f}"
        #             va = 'bottom' if max(obs_rs) > abs(min(obs_rs)) else 'top'
        #             text_y = max(obs_rs) if va == 'bottom' else min(obs_rs)
        #     else:
                
        #         esd_value = float(esd[0]) if isinstance(esd, np.ndarray) else float(esd)
        #         label_text = f"ESD={esd_value:.3f}"
        #         va = 'bottom' if max(obs_rs) > abs(min(obs_rs)) else 'top'
        #         text_y = max(obs_rs) if va == 'bottom' else min(obs_rs)


        # elif plot_type == "AUC":
        #     if result is not None and geneset in result.index and "AUC" in result.columns:
        #         try:
        #             if isinstance(result.index, pd.MultiIndex):
        #                 auc = result.xs((geneset, i), level=('Term', 'Sample'))['AUC']
        #             else:
        #                 auc = result.loc[geneset, 'AUC']
        #                 if isinstance(auc, pd.Series):
        #                     auc = auc.iloc[i] if i < len(auc) else auc.iloc[0]
                    
        #             if isinstance(auc, pd.Series):
        #                 auc = auc.iloc[0]
                    
        #             label_text = f"AUC={auc:.3f}"
        #             va = 'bottom' if max(obs_rs) > abs(min(obs_rs)) else 'top'
        #             text_y = max(obs_rs) if va == 'bottom' else min(obs_rs)
        #         except Exception as e:
        #             auc_value = float(AUC[0]) if isinstance(auc, np.ndarray) else float(AUC)
        #             label_text = f"AUC={auc_value:.3f}"
        #             va = 'bottom' if max(obs_rs) > abs(min(obs_rs)) else 'top'
        #             text_y = max(obs_rs) if va == 'bottom' else min(obs_rs)
        #     else:
        #         auc_value = float(AUC[0]) if isinstance(AUC, np.ndarray) else float(AUC)
        #         label_text = f"AUC={auc_value:.3f}"
        #         va = 'bottom' if max(obs_rs) > abs(min(obs_rs)) else 'top'
        #         text_y = max(obs_rs) if va == 'bottom' else min(obs_rs)
        # else:
        #     raise ValueError("Invalid plot_type. Choose from 'ES', 'ESD', or 'AUC'.")

        # fontsize_text = 25 if compact else 20
        
        # # text label
        # ax1.text(nn, text_y, label_text, 
        #  size=fontsize_text, 
        #  bbox={'facecolor':'white', 'alpha':0.8, 'edgecolor':'none', 'pad':1}, 
        #  ha='center',  
        #  va=va,        
        #  zorder=100) 
        
        ax1.grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax1.set(xticks=[])
        plt.title(f"{geneset} - {plot_type}(Celltype{i+1})", fontsize=18)
        plt.ylabel("Enrichment Score (ES)" if plot_type == "ES" else 
                "Enrichment Score Derivative (ESD)" if plot_type == "ESD" else 
                "Area Under Curve (AUC)", fontsize=24 if compact else 16)
        

        if compact:
            ax2 = fig.add_subplot(gs[4:5, 0:11],facecolor='white')
            for spine in ax2.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
            hit_positions = np.where(overlap_ratios[:, 0] > 0)[0]
            position_map = {old: new for new, old in enumerate(sort_indices.flatten())}
            sorted_hit_positions = np.array([position_map[pos] for pos in hit_positions])
    
            #print(hit_positions)
            for pos in sorted_hit_positions:
                original_pos = hit_positions[np.where(sorted_hit_positions == pos)[0][0]]
                ratio_val = overlap_ratios[pos, 0]
                #print(pos, ratio_val)
                color = cm(norm(ratio_val))
                ax2.vlines(x=pos, ymin=0, ymax=1, color=color, lw=1.5)
            ax2.set_xlim([0, len(obs_rs)])
            ax2.set_ylim([0, 1])
            ax2.set(yticks=[])
            ax2.set(xticks=[])
            ax2.set_xlabel("Rank", fontsize=24)
            

            rank_vec = sig_val_i.flatten()
            pos_indices = np.where(rank_vec > 0)[0]
            neg_indices = np.where(rank_vec <= 0)[0]
            
            if len(pos_indices) > 0:
                posv = np.percentile(pos_indices, np.linspace(0, 100, 10))
                for j in range(9):
                    ax2.add_patch(Rectangle((posv[j], 0), 
                                            posv[j+1]-posv[j], 
                                            0.5, 
                                            linewidth=0, 
                                            facecolor='red', 
                                            alpha=0.6*(1-j*0.1)))
            if len(neg_indices) > 0:
                negv = np.percentile(neg_indices, np.linspace(0, 100, 10))
                for j in range(9):
                    ax2.add_patch(Rectangle((negv[j], 0), 
                                            negv[j+1]-negv[j], 
                                            0.5, 
                                            linewidth=0, 
                                            facecolor='blue', 
                                            alpha=0.6*(0.1+j*0.1)))
                    
                  
        else:
            ax2 = fig.add_subplot(gs[7:8, 0:11],facecolor='white')
            for spine in ax2.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
            hit_positions = np.where(overlap_ratios[:, 0] > 0)[0]

            position_map = {old: new for new, old in enumerate(sort_indices.flatten())}
            sorted_hit_positions = np.array([position_map[pos] for pos in hit_positions])

            #print(hit_positions)
            for pos in sorted_hit_positions:
                original_pos = hit_positions[np.where(sorted_hit_positions == pos)[0][0]]
                ratio_val = overlap_ratios[original_pos, 0]
                #print(pos, ratio_val)
                color = cm(norm(ratio_val))
                ax2.vlines(x=pos, ymin=-1, ymax=1, color=color, lw=0.5)
            ax2.set_xlim([0, len(obs_rs)])
            ax2.set_ylim([-1, 1])
            ax2.set(yticks=[])
            ax2.set(xticks=[])


            ax3 = fig.add_subplot(gs[8:12, 0:11],facecolor='white')
            for spine in ax3.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
            rank_vec = sig_val_i[sort_indices.flatten()]
            step = max(1, len(rank_vec) // 100)
            x = np.arange(0, len(rank_vec), step).astype(int)
            x = np.append(x, len(rank_vec)-1)
            y = rank_vec[x].astype(float)

            

            if not np.all(np.isfinite(y)):
                raise ValueError("y is NaN or Inf, Please check your input data.")
            
            ax3.fill_between(x, y, color="lightgrey")
            ax3.plot(x, y, color=(0.2,0.2,0.2), lw=1)
            ax3.hlines(y=0, xmin=0, xmax=len(rank_vec), color="black", zorder=100, lw=0.6)
            ax3.set_xlim([0, len(obs_rs)])
            ax3.set_ylim([np.min(rank_vec), np.max(rank_vec)])
            
            signs = np.sign(rank_vec)
            zero_cross_indices = np.where(signs[:-1] != signs[1:])[0]
            if zero_cross_indices.size > 0:
                zero_cross = int(zero_cross_indices[0])
                ax3.vlines(x=zero_cross, ymin=np.min(rank_vec), ymax=np.max(rank_vec), linestyle=':',color = 'black')
                ax3.text(zero_cross, np.max(rank_vec)/3, "Zero crosses at "+str(zero_cross), 
                        bbox={'facecolor':'white','alpha':0.5,'edgecolor':'none','pad':1}, 
                        ha='center', va='center')

            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.xlabel("Rank in Ordered Dataset", fontsize=16)
            plt.ylabel("Ranked list metric", fontsize=16)
            ax3.tick_params(labelsize=16)
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=fig.axes, orientation='vertical', fraction=0.05, pad=0.1)
        cbar.set_label("Overlap Ratio", fontsize=16)
        fig.patch.set_facecolor('white')
        figures.append(fig)

    if n_samples == 1:
        return figures[0]
    else:
        return figures



def top_table(sig_name, sig_val, library, result, n=10, center=True, interactive_plot=False, plot_type="ES", sig_sep=','):
    """
    Plot a table to enrichment results for top N enriched gene sets for a given geneset and signature.

    Parameters:
    signature (array-like): The gene expression signature to analyze.
    library (array-like): The gene set library to use for enrichment analysis.
    result (array-like, optional): A precomputed enrichment result. Default is None.
    n (integer): number of top enriched gene sets to be plotted
    
    Returns:
    figure: The running sum plot for the given geneset and signature.
    """

    if not isinstance(result.index, pd.MultiIndex):
        result = result.set_index(['Term', 'Sample'])

    if not interactive_plot:
        plt.ioff()
    sig_name = sig_name.copy()
    sig_val = sig_val.copy().astype(float)

    if sig_val.shape[1] != sig_name.shape[1]:
        if sig_val.shape[1] > sig_name.shape[1]:
            sig_name = np.repeat(sig_name[:, -1:], sig_val.shape[1], axis=1)
   
    n_samples = sig_val.shape[1]
    figures = []
    for sample_idx in range(n_samples):
        print(f"----Start Sample {sample_idx+1}----")

        col_sig_name = sig_name[:, sample_idx:sample_idx+1]
        col_sig_val = sig_val[:, sample_idx:sample_idx+1]

        df = pd.DataFrame({"sig_name": col_sig_name.flatten(), "sig_val": col_sig_val.flatten()})
        split_genes = df['sig_name'].str.split('_').explode()
        
        df_expanded = pd.DataFrame({
            'sig_name': split_genes,
            'sig_val': df['sig_val'].repeat(df['sig_name'].str.split('_').str.len())
        })

        sig = df_expanded.sort_values(by="sig_val", ascending=False).set_index("sig_name")
        sig = sig[~sig.index.duplicated(keep='first')]


        if center:
            col_sig_val = col_sig_val - np.mean(col_sig_val)    

        sample_result = result.xs(key=sample_idx, level= "Sample")
        #print("Sample result index:", sample_result.index)
        top_n = min(n, len(result))
         
        fig = plt.figure(figsize=(5,0.5*top_n), frameon=False)
        ax = fig.add_subplot(111)
        fig.patch.set_visible(False)
        plt.axis('off')

        ax.vlines(x=[0.2,0.8], ymin=-0.1, ymax=1, color="black")
        ln = np.linspace(-0.1,1,top_n+1)[::-1]
        ax.hlines(y=ln, xmin=0, xmax=1, color="black")

        if plot_type == 'ES':
            ax.text(0.03, 1.03, "NES", fontsize=16)
        elif plot_type == 'ESD':
            ax.text(0.03, 1.03, "NESD", fontsize=16)
        elif plot_type == 'AUC':
            ax.text(0.03, 1.03, "AUC", fontsize=16)
        

        ax.text(0.84, 1.03, "SET", fontsize=16)

        
        for i in range(top_n):
            if plot_type == 'ES':
                value = sample_result.iloc[i]["nes"] 
            elif plot_type == 'ESD':
                value = sample_result.iloc[i]["nesd"]
            elif plot_type == 'AUC':
                value = sample_result.iloc[i]["AUC"]

        
            ax.text(0.03, (ln[i]+ln[i+1])/2, "{:.3f}".format(value), verticalalignment='center')
            ax.text(0.84, (ln[i]+ln[i+1])/2, sample_result.index[i], verticalalignment='center')

            term_name = sample_result.index[i]

            gs = set(library[term_name])
            print(f"Number of genes in gene set: {len(gs)}")

            common_genes = [x for x in sig.index if x in gs]
            print(f"Number of overlapping genes: {len(common_genes)}")
            

            hits = np.array([i for i,x in enumerate(sig.index) if x in gs])
            hits = (hits/len(sig.index))*0.6+0.2
            print(f"Hit positions: {hits}")

            if plot_type == 'ES':    
                if sample_result.iloc[i]["nes"] > 0:
                    ax.vlines(hits, ymax=ln[i], ymin=ln[i+1], color="red", lw=0.5, alpha=0.3)
                else:
                    ax.vlines(hits, ymax=ln[i], ymin=ln[i+1], color="blue", lw=0.5, alpha=0.3)
            elif plot_type == 'ESD':
                if sample_result.iloc[i]["nesd"] > 0:
                    ax.vlines(hits, ymax=ln[i], ymin=ln[i+1], color="red", lw=0.5, alpha=0.3)    
                else:
                    ax.vlines(hits, ymax=ln[i], ymin=ln[i+1], color="blue", lw=0.5, alpha=0.3)
            elif plot_type == 'AUC':
                if sample_result.iloc[i]["AUC"] > 0:
                    ax.vlines(hits, ymax=ln[i], ymin=ln[i+1], color="red", lw=0.5, alpha=0.3)    
                else:
                    ax.vlines(hits, ymax=ln[i], ymin=ln[i+1], color="blue", lw=0.5, alpha=0.3)
        ax.set_title(f"Sample {sample_idx + 1}", fontsize=14)
        fig.patch.set_facecolor('white')
        figures.append(fig)
    if n_samples == 1:
        return figures[0]
    else:
        return figures




        
def plot_box(ax, data, ranks, color, label, min_p_value_line=None, line_label=None):
    # 绘制boxplot
    data = pd.DataFrame(data)  
    box = ax.boxplot(data.T.loc[:, ranks], showfliers=False, patch_artist=True,
                     boxprops=dict(facecolor=color, color=color),
                     capprops=dict(color=color),
                     whiskerprops=dict(color=color),
                     medianprops=dict(color="black"), labels=[""] * len(ranks))
    if min_p_value_line is not None:
        ax.hlines(min_p_value_line, 0, data.shape[0], color="grey", linestyles="--")
        ax.text(data.shape[0], min_p_value_line, line_label, fontsize=14, horizontalalignment='right', verticalalignment='bottom')
    return box        


def plot_gene_heatmap(data, 
                     output_path=None,
                     cluster=True,
                     cluster_method='ward',
                     figsize=(12, 8),
                     cmap="RdBu_r",
                     show_gene_x_labels=True,
                     show_gene_y_labels=True,
                     font_scale=0.8,
                     center=0,
                     gene_list=None,
                     sample_list=None,
                     top_n_genes=50,
                     selection_method='var',
                     sig_sep=','):

    """
    Generate a heatmap for gene expression data.
    
    Parameters:
    -----------
    data : DataFrame or str
        Expression matrix with samples as columns and genes as rows.
    sig_sep : str, optional
        Separator character in gene names to split on (default: ',')
        If specified, will check and process gene names containing this separator
    """
    try:
        # 1. Load and preprocess data
        df = data.copy()
        
        # Check for separator in gene names and warn if found
        if sig_sep:
            genes_with_sep = df.index[df.index.str.contains(sig_sep, regex=False)]
            if len(genes_with_sep) > 0:
                print(f"Warning: Found {len(genes_with_sep)} gene names containing '{sig_sep}'")
                print("Example genes:", genes_with_sep[:5].tolist())
                user_input = input(f"Would you like to split these gene names on '{sig_sep}'? (y/n): ")
                if user_input.lower() == 'y':
                    # Split gene names and keep the first part
                    df.index = df.index.str.split(sig_sep).str[0]
                    print("Gene names have been processed.")
        # Convert all data to numeric, forcing non-numeric to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
    
        
        # 3. Select top genes if no specific gene_list is provided
        if gene_list is None:
            if selection_method == 'var':
                gene_vars = df.var(axis=1)
                # Handle any remaining non-numeric values
                gene_vars = pd.to_numeric(gene_vars, errors='coerce')
                gene_vars = gene_vars.fillna(0)
                selected_genes = gene_vars.sort_values(ascending=False).head(top_n_genes).index.tolist()
            elif selection_method == 'mean':
                gene_means = df.abs().mean(axis=1)
                gene_means = pd.to_numeric(gene_means, errors='coerce')
                gene_means = gene_means.fillna(0)
                selected_genes = gene_means.sort_values(ascending=False).head(top_n_genes).index.tolist()
            elif selection_method == 'mad':
                gene_mads = df.apply(lambda x: np.median(np.abs(x - np.median(x))), axis=1)
                gene_mads = pd.to_numeric(gene_mads, errors='coerce')
                gene_mads = gene_mads.fillna(0)
                selected_genes = gene_mads.sort_values(ascending=False).head(top_n_genes).index.tolist()
            else:
                raise ValueError("Invalid selection_method. Choose 'var', 'mean', or 'mad'")
            
            df = df.loc[selected_genes]
        else:
            df = df.loc[gene_list]


        # 4. Filter samples if specified
        if sample_list is not None:
            df = df[sample_list]

        single_sample = df.shape[1] == 1
        single_gene = df.shape[0] == 1

        if   single_sample or single_gene:
            cluster = False
            
        # 6. Set up plotting parameters
        sns.set(font_scale=font_scale)
        plt.figure(figsize=figsize)
        
        # 7. Generate heatmap
        if cluster:
            # Perform hierarchical clustering
            row_linkage = hierarchy.linkage(row_dist, method=cluster_method)
            col_linkage = hierarchy.linkage(col_dist, method=cluster_method)
            
            # Create clustered heatmap
            g = sns.clustermap(df,
                                row_linkage=row_linkage,
                                col_linkage=col_linkage,
                                cmap=cmap,
                                center=center,
                                xticklabels=show_gene_x_labels,
                                yticklabels=show_gene_y_labels,
                                figsize=figsize)
            g.ax_heatmap.set_xlabel("Samples", fontsize=20)
            g.ax_heatmap.set_ylabel("Genes", fontsize=20)
            g.fig.suptitle("Gene Expression Heatmap", fontsize=24)
        else:
            ax = sns.heatmap(df,
                           cmap=cmap,
                           center=center,
                           xticklabels=show_gene_x_labels,
                           yticklabels=show_gene_y_labels)
            ax.set_xlabel("Samples", fontsize=20)
            ax.set_ylabel("Genes", fontsize=20)
            ax.set_title("Gene Expression Heatmap", fontsize=24)
        # 8. Adjust layout
        plt.tight_layout()
        
        # 9. Save or display the figure
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")