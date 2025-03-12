import pandas as pd
import numpy as np
from grea.benchmarking import benchmark_parallel
import grea.library as library


if __name__ == '__main__':
    library = library.read_gmt("data/Enrichr.KEGG_2021_Human.gmt")
    signature = pd.read_csv("data/ageing_muscle_gtex.tsv")
    


    sub_library = {}
    for i, key in enumerate(library.keys()):
        if i > 5:
            break
        sub_library[key] = library[key]


    res = benchmark_parallel(signature, sub_library,n=4)


    escores_data = {}
    for method in res['pval']:
        values = res['pval'][method]
        if values and all(isinstance(item, list) for item in values):
            flat_values = [item for sublist in values for item in sublist]
            escores_data[method] = flat_values
        else:
            escores_data[method] = values

    escores_df = pd.DataFrame(escores_data)
    escores_df.to_csv("enrichment_pvals.csv", index=False)


    times_data = {}
    for method in res['times']:
        values = res['times'][method]
        if values and all(isinstance(item, list) for item in values):
            flat_values = [item for sublist in values for item in sublist]
            times_data[method] = flat_values
        else:
            times_data[method] = values

    times_df = pd.DataFrame(times_data)
    times_df.to_csv("enrichment_times.csv", index=False)