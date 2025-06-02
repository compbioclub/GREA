import numpy as np
import pandas as pd

from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt

from grea import grea


def profile_bipolarity(rank_df, libraries, n_perm=1000):


    obj = grea.obs_prerank_enrich(rank_df, libraries, center=False, add_noise=False, n_perm=n_perm)
    df = obj.get_enrich_score(metric='nRC-AUC').T

    for col in df.columns:
        if col.endswith('_-'):
            df[col] = 1 - df[col]

    return df


def prob_test(df, probs=np.arange(100, 0, -10), n_perm=1000):
                
    perf_results = pd.DataFrame()
    pred_results = pd.DataFrame(index=df.index)
    
    for prob in probs:
        print(f'prob {prob}')
        t_perf, common_preds = _random_permutation_test_by_prob(df, prob=prob, n_perm=n_perm)
        perf_results[f'prob_{prob}%'] = t_perf
            
        pred_results[f'prob_{prob}%'] = ['high' if x in common_preds else 'low' for x in df.index]
    
    p_value = perf_results.apply(lambda x: np.sum(x[1:] >= x[0]) / len(x[1:]), axis=0)
    medianN = perf_results.iloc[1:, :].median(axis=0)
    preN = perf_results.iloc[0, :]
    maxN = perf_results.iloc[1:, :].max(axis=0)
    minN = perf_results.iloc[1:, :].min(axis=0)
    fdr = multipletests(p_value, method='fdr_bh')[1]
    
    ratio = preN / maxN
    t_results = pd.DataFrame({
        'p_value': p_value,
        'fdr': fdr,
        'preN': preN,
        'random_medianN': medianN,
        'random_maxN': maxN,
        'random_minN': minN,
        'Rcm': ratio
    })

    t_results.index = t_results.index.str.replace('prob_', '')

    t_results.plot(y='Rcm', kind='bar', ylim=(0, 2), ylabel='Rcm value', color='peachpuff')
    plt.axhline(y=1, linestyle='dotted', linewidth=3, color='red')
    plt.show()    
    return t_results, perf_results, pred_results
            
def _random_permutation_test_by_prob(df, prob=50, n_perm=10, random_seed=0):
    np.random.seed(random_seed)
    t_perf = []
    N = df.shape[0]
    for i in range(n_perm):
        shuffled_df = df.copy()  
        for col in shuffled_df.columns:
            shuffled_df[col] = np.random.permutation(shuffled_df[col])  # Shuffle the column

        random_selected = _predict_by_prob(shuffled_df, prob)              
        t_perf.append(len(random_selected))

    common_preds = _predict_by_prob(df,prob=prob)
        
    t_perf = [len(common_preds)] + t_perf
    
    return t_perf, common_preds            
    
def _predict_by_prob(df, prob):
    preds = []
    for col in df.columns:
        f = df[col]
        preds += df[f > prob*0.01].index.tolist()
    return set(preds)