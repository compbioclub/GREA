import numpy as np
import pandas as pd

from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt

from grea import grea


def profile_bipolarity(rank_df, libraries, 
                       high='high', low='low',
                       n_perm=1000, prefix=''):


    obj = grea.obs_prerank_enrich(rank_df, libraries, center=False, add_noise=False, n_perm=n_perm)
    auc_df = obj.get_enrich_score(metric='nRC-AUC').T

    for col in auc_df.columns:
        if col.endswith('_-'):
            auc_df[col] = 1 - auc_df[col]

    stat_df, perf_df, pred_df = prob_test(
        auc_df, n_perm=n_perm, high=high, low=low, prefix=prefix)

    plot_perm_stat(stat_df, prefix=prefix)

    return auc_df, stat_df, perf_df, pred_df

def plot_perm_stat(stat_df, fontsize=10, figsize=(5, 4), prefix=''):
    df = stat_df.copy()
    df.index = stat_df.index.str.replace('prob_', '')
    fig, ax1 = plt.subplots(figsize=figsize)
    bars = ax1.bar(df.index, df['Rcm'], color='peachpuff', label='Rcm')
    # Add significance annotations for `p_value` and `fdr`
    for i, (pval, fdr) in enumerate(zip(df['p_value'], df['fdr'])):
        if pval < 0.05 or fdr < 0.05:  # Significance threshold
            plt.text(i, 0.2, '*', ha='center', va='bottom', fontsize=fontsize, color='red') 
    # Add labels, title, and legend
    plt.ylim(0, 2)
    plt.axhline(y=1, linestyle='dotted', linewidth=2, color='red')
    plt.xlabel("AUC prob cutoff", fontsize=fontsize)
    plt.ylabel("Rcm value", fontsize=fontsize)

    ax2 = ax1.twinx()
    ax2.plot(df.index, df['p_value'], marker='o', color='green', label='p_value')
    ax2.plot(df.index, df['fdr'], marker='o', color='orange', label='hdr')
    ax2.axhline(y=0.05, linestyle='dotted', linewidth=2, color='red')
    ax2.set_ylabel("p_value / fdr", fontsize=fontsize)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=fontsize)
    if prefix:
        plt.title(f'{prefix} Permutation Statistics')
    else:
        plt.title('Permutation Statistics')
    plt.show()

def prob_test(df, probs=np.arange(100, 0, -10), n_perm=1000,
              high='high', low='low', prefix=''):
                
    pred_df, perf_df = _random_permutation_test_by_probs(
        df, probs=probs, n_perm=n_perm, high=high, low=low)


    p_value = perf_df.apply(lambda x: np.sum(x[1:] >= x[0]) / len(x[1:]), axis=0)
    medianN = perf_df.iloc[1:, :].median(axis=0)
    obsN = perf_df.iloc[0, :]
    maxN = perf_df.iloc[1:, :].max(axis=0)
    minN = perf_df.iloc[1:, :].min(axis=0)
    fdr = multipletests(p_value, method='fdr_bh')[1]
    
    ratio = obsN / maxN
    stat_df = pd.DataFrame({
        'p_value': p_value,
        'fdr': fdr,
        'obsN': obsN,
        'random_medianN': medianN,
        'random_maxN': maxN,
        'random_minN': minN,
        'Rcm': ratio
    })    
    return stat_df, perf_df, pred_df
            

def _random_permutation_test_by_probs(
        combo_df, probs=np.arange(100, 0, -10), n_perm=10, high='high', low='low'):
    
    pred_df, n_selected = _predict_by_probs(combo_df, probs, high=high, low=low)
    perf = [n_selected]
    for i in range(n_perm):
        shuffled_df = combo_df.copy()  
        for i, col in enumerate(shuffled_df.columns):
            np.random.seed(i)
            shuffled_df[col] = np.random.permutation(shuffled_df[col])  # Shuffle the column

        _, n_selected = _predict_by_probs(shuffled_df, probs, high=high, low=low, save_pred=False)              
        perf.append(n_selected)

    perf_df = pd.DataFrame(
        perf, 
        columns=pred_df.columns, 
        index=['obs']+[f'perm_{i}' for i in range(n_perm)])   

    return pred_df, perf_df            

def _predict_by_probs(combo_df, probs=np.arange(100, 0, -10), high='high', low='low',
                      save_pred=True):
    # could be combo_df or series
    n_selected = []
    if save_pred:
        pred_df = pd.DataFrame(index=combo_df.index)
    else:
        pred_df = None
    for prob in probs:
        selected = combo_df[(combo_df >= prob*0.01).all(axis=1)].index.tolist()
        n_selected.append(len(selected))
        if save_pred:
            pred_df[f'prob_{prob}%'] = [high if x in selected else low for x in combo_df.index]
    return pred_df, n_selected   
