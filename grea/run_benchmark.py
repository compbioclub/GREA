import pandas as pd
import numpy as np
from benchmarking import benchmark_parallel
import grea.library as library
import os


library = library.read_gmt("DATA/Enrichr.KEGG_2021_Human.gmt")
signature = pd.read_csv("DATA/GSE52428_H1N1_expr")


def run_bench(signature, library,rep_n, output_dir='result'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    res = benchmark_parallel(signature, library, rep_n=rep_n)

    return res
if __name__ == '__main__':
    run_bench(signature, library,rep_n=11, output_dir='result')