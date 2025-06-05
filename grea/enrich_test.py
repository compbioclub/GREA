from statsmodels.nonparametric.smoothers_lowess import lowess
from collections import Counter
from scipy import interpolate
import pandas as pd
import numpy as np
import random
from scipy.stats import gamma
from scipy.stats import kstest
import multiprocessing
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpmath import mp, mpf
from multiprocessing import Pool


import grea.enrich_signal as enrich_signal
import grea.prob as prob


def enrich(obj):

    enrich_signal.get_overlap(obj) 
    enrich_signal.get_sorted(obj)
    enrich_signal.get_running_sum(obj)  
    if obj.stop == 'running_sum':
        return
    
    enrich_signal.get_ES_ESD(obj)  
    if obj.get_lead_sigs:
        enrich_signal.get_leading_edge(obj)  
    enrich_signal.get_AUC(obj)
    # === p-values ===
    if obj.get_pval:
        enrich_signal.get_metrics_null(obj)
        prob.get_prob(obj)

