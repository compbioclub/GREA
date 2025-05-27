
from statsmodels.stats.multitest import multipletests
import pandas as pd

def _get_metric_matrix(obj, metric):
    lead_sigs, pval = None, None
    if metric == 'KS-ES':
        ES = obj.ES
        if obj.get_lead_sigs:
            lead_sigs = obj.lead_sigs
        if obj.get_pval:
            pval = obj.ES_pval
    if metric == 'KS-ESD':
        ES = obj.ESD
        if obj.get_lead_sigs:
            lead_sigs = obj.lead_sigs
        if obj.get_pval:
            pval = obj.ESD_pval
    if metric == 'RC-AUC':
        ES = obj.AUC
        if obj.get_pval:
            pval = obj.AUC_pval
    return ES, pval, lead_sigs

def enrich_long_df(obj, metric):
    ES, pval, lead_sigs_list = _get_metric_matrix(obj, metric)
    n_term, n_obs = ES.shape
    df_list = []
    for o in range(n_obs):
        res_list = []
        for t in range(n_term):
            res = {
                'Term': obj.term_names[t],
                'Obs': obj.obs_names[o],
                metric: ES[t, o],
            }
            if pval is not None:
                res['Prob_method'] = obj.prob_method
                res[f'{metric}_pval'] = pval[t, o]
            if lead_sigs_list:
                lead_sigs = lead_sigs_list[t][o]
                lead_n = len(lead_sigs)
                lead_str = ';'.join(lead_sigs)                
                res['N_lead_sigs'] = lead_n
                res['Lead_sigs'] = lead_str
            res_list.append(res)
        df = pd.DataFrame(res_list)
        if pval is not None:
            df[f'{metric}_fdr'] = multipletests(df[f'{metric}_pval'], method='fdr_bh')[1]  
            df[f'{metric}_sidak'] = multipletests(df[f'{metric}_pval'], method='sidak')[1]
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.dropna(subset=[metric])
    df = df.sort_values(by=metric, ascending=False)
    return df

def enrich_wide_df(obj, metric):
    ES, pval, lead_sigs_list = _get_metric_matrix(obj, metric)
    df = pd.DataFrame(ES, columns=obj.obs_names, index=obj.term_names)
    return df

def print_enrich(obj, metric, t, o, **kwargs):
    ES, pval, lead_sigs = _get_metric_matrix(obj, metric)
    text = f"{metric}: {ES[t, o]:.3f}"
    if pval is not None:
        text += f"\np-val: {pval[t, o]:.3f}"
    if lead_sigs:
        text += f"\n# of leading edge: {len(lead_sigs)}"
    return text