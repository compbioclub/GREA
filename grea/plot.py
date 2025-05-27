import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.stats import gamma
from matplotlib.patches import Rectangle
from sklearn.preprocessing import MaxAbsScaler
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy import stats

from grea import out

def running_sum(obj, metric, term, obs_id,
                interactive_plot=False, 
                hit_cmap='YlGnBu', 
                title=''):
    """
    Plot the running sum for a given geneset and signature.

    Returns:
    figure: The running sum plot for the given geneset and signature.
    """
    if not interactive_plot:
        plt.ioff()

    t = obj.term_names.index(term)
    o = obj.obs_names.index(obs_id)
     
    fig = plt.figure(figsize=(7,5),facecolor='white',edgecolor='black')
    gs = fig.add_gridspec(12, 11, wspace=0, hspace=0)

    curve_ax = fig.add_subplot(gs[0:7, 0:11],facecolor='white')
    _pl_rs_curve(curve_ax, obj, metric, t, o, title=title)
    
    if isinstance(hit_cmap, str):
        cm = plt.cm.get_cmap(hit_cmap)
    else:
        cm = hit_cmap 
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    hit_ax = fig.add_subplot(gs[7:8, 0:11],facecolor='white')
    _plt_rs_hit(hit_ax, obj, t, o, cm, norm)
    
    rank_ax = fig.add_subplot(gs[8:12, 0:11],facecolor='white')
    _pl_rs_rank(rank_ax, obj, o)

    cbar = plt.colorbar(sm, ax=fig.axes, orientation='vertical', fraction=0.05, pad=0.1)
    cbar.set_label("Overlap Ratio", fontsize=12)
    fig.patch.set_facecolor('white')
    return fig

def _pl_rs_curve(ax, obj, metric, t, o, title=''):

    if metric.startswith('KS'):
        rs_matrix = obj.ks_rs
    elif metric == 'RC-AUC':
        rs_matrix = obj.rc_rs
    elif metric == 'RC-nAUC':
        rs_matrix = obj.nrc_rs
    obs_rs = rs_matrix[t, :, o]

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    ax.plot(obs_rs, color=(0,1,0), lw=5)
    ax.tick_params(labelsize=10)
 
    plt.xlim([0, len(obs_rs)])
    nn = np.argmax(np.abs(obs_rs))  

    if metric.startswith("KS"):
        ax.vlines(x=nn, ymin=np.min(obs_rs), ymax=np.max(obs_rs),linestyle = ':', color="red")
        x_pos = len(obs_rs) * 0.2
        y_pos = np.min(obs_rs) + (np.max(obs_rs) -np.min(obs_rs)) * 0.3
    else:
        x_pos = len(obs_rs) * 0.8
        y_pos = np.min(obs_rs) + (np.max(obs_rs) -np.min(obs_rs)) * 0.3
    text = out.print_enrich(obj, metric, t, o)
    ax.text(x_pos, y_pos, text,
                fontsize=9, color='black',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
        
    ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set(xticks=[])
    text = f"{obj.term_names[t]} \n {metric} - {obj.obs_names[o]}"
    if title:
        text += f'\n{title}'
    plt.title(text, fontsize=12)
    plt.ylabel(f"Running Sum ({metric.split('-')[0]})", fontsize=12)


def _plt_rs_hit(ax, obj, t, o, cm, norm):
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    n_sig = obj.sorted_or.shape[1]
    for i in range(n_sig):
        ratio_val = obj.sorted_or[t, i, o]
        if ratio_val == 0:
            continue
        color = cm(norm(ratio_val))
        ax.vlines(x=i, ymin=-1, ymax=1, color=color, lw=0.5)
    ax.set_xlim([0, n_sig])
    ax.set_ylim([-1, 1])
    ax.set(yticks=[])
    ax.set(xticks=[])


def _pl_rs_rank(ax, obj, o):
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    rank_vec = obj.sorted_sig_vals[:, o]            
    step = max(1, len(rank_vec) // 100)
    x = np.arange(0, len(rank_vec), step).astype(int)
    x = np.append(x, len(rank_vec)-1)
    y = rank_vec[x].astype(float)

    scaler = MaxAbsScaler()
    y_norm = scaler.fit_transform(y.reshape(-1,1)).flatten()

    if not np.all(np.isfinite(y)):
        raise ValueError("y is NaN or Inf, Please check your input data.")
            
    ax.fill_between(x, y_norm, color="lightgrey")
    ax.plot(x, y_norm, color=(0.2,0.2,0.2), lw=1)
    ax.hlines(y=0, xmin=0, xmax=len(rank_vec), color="black", zorder=100, lw=0.6)

    ax.set_xlim([0, len(rank_vec)])
    ax.set_ylim([-1.05, 1.05]) 
           
    signs = np.sign(rank_vec)
    zero_cross_indices = np.where(signs[:-1] != signs[1:])[0]
    if zero_cross_indices.size > 0:
        zero_cross = int(zero_cross_indices[0])
        ax.vlines(x=zero_cross, ymin=-1, ymax=1, linestyle=':',color = 'black')
        ax.text(zero_cross,0.3, "Zero crosses at "+str(zero_cross), 
                        bbox={'facecolor':'white','alpha':0.5,'edgecolor':'none','pad':1}, 
                        ha='center', va='center')

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.xlabel("Ordered Signatures", fontsize=12)
    plt.ylabel("Sig Rank Score", fontsize=12)
    ax.tick_params(labelsize=10)


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

        ax.text(0.03, 1.03, "AUC", fontsize=16)
        

        ax.text(0.84, 1.03, "SET", fontsize=16)

        
        for i in range(top_n):
            if plot_type == 'ES':
                value = sample_result.iloc[i]["nes"] 
            elif plot_type == 'ESD':
                value = sample_result.iloc[i]["nesd"]
            elif plot_type == 'AUC':
                value = sample_result.iloc[i]["AUC"]
            elif plot_type == 'nAUC':
                value = sample_result.iloc[i]["nAUC"]
        
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