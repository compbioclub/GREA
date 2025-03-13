import pandas as pd
import numpy as np
from benchmarking import benchmark_parallel
import library
import os


library = library.read_gmt("DATA/Enrichr.KEGG_2021_Human.gmt")
signature = pd.read_csv("DATA/ageing_muscle_gtex.tsv")

sub_library = {}
for i, key in enumerate(library.keys()):
    if i > 5:
        break
    sub_library[key] = library[key]


def run_bench(signature, library,rep_n, output_dir='result',perm_list=[250,500,750,]):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    res = benchmark_parallel(signature, library, rep_n=rep_n,perm_list=perm_list)

    return res
if __name__ == '__main__':
    run_bench(signature, sub_library, rep_n=3, output_dir='result')