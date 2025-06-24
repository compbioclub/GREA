from typing import List
import itertools
import urllib.request
import shutil
import json
import os
import re
import ssl
import pandas as pd
from copy import deepcopy
from bioservices import KEGG

def get_library_from_names(
        libraries, add_lib_key=True,
        min_size=None, max_size=None):
    mylibs = list_libraries()
    mylibs += ['KO_PATHWAY_KO_TERM']  
    mylibs += ['Mouse.MitoCarta3.0'] 
    mylibs += ['c5.go.bp.v2025.1.Hs', 'c5.go.cc.v2025.1.Hs', 'c5.go.mf.v2025.1.Hs']
    mylibs += ['m5.go.bp.v2025.1.Mm', 'm5.go.cc.v2025.1.Mm', 'm5.go.mf.v2025.1.Mm']

    term_dict = {}
    for key in libraries:
        if key not in mylibs:
            print(f'---WARMING: "{key}" is not a valid library name, filter it out.')
            continue
        lib = get_library(key)
        mylib = {}
        for term, genes in lib.items():
            if min_size is not None and len(genes) < min_size:
                print(f'---WARMING: "{key}-{term}" has {len(genes)} genes, less than min_size {min_size}, filter it out.')
                continue
            if max_size is not None and len(genes) > max_size:
                print(f'---WARMING: "{key}-{term}" has {len(genes)} genes, larger than max_size {max_size}, filter it out.')
                continue
            if add_lib_key:
                mylib[f'{key}|{term}'] = genes
            else:
                mylib[term] = genes
        n_term = len(mylib.keys())
        print(f'---Finished: Load {key} with {n_term} terms.')
        term_dict.update(mylib)
    return term_dict

def get_library(library: str):
    """
    Load gene set library from Enrichr as dictionary.

    Parameters:
    signature (string): Name of Enrichr gene set library.

    Returns:
    Gene set library as dictionary.
    """
    if library == 'KO_PATHWAY_KO_TERM':
        return get_ko_library()
    elif library == 'Mouse.MitoCarta3.0':
        return get_mt_library()
    elif library.startswith('c5') or library.startswith('m5'):
        wdr = os.path.dirname(os.path.abspath(__file__))
        return read_gmt(f'{wdr}/db/{library}.symbols.gmt')
    else:
        return read_gmt(load_library(library))


def get_ko_library():

    # Initialize KEGG service
    kegg = KEGG()

    # Retrieve all Pathway to KO mappings
    pathway_to_ko = kegg.link("ko", "pathway")

    # Process into dictionary format with pathways as keys
    pathway_ko_dict = {}
    for line in pathway_to_ko.splitlines():
        pathway, ko = line.split('\t')
        if pathway.startswith('path:map'):
            continue
        pathway = pathway.replace("path:", "")
        ko = ko.replace("ko:", "")
        pathway_ko_dict.setdefault(pathway, []).append(ko)

    library = {}
    for i, pathway in enumerate(pathway_ko_dict.keys()):
        name = kegg.get(pathway).split('\n')[1].replace('NAME', '').strip()
        library[name] = pathway_ko_dict[pathway]
    return library

def get_mt_library():
    wdr = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(f'{wdr}/db/Mouse.MitoCarta3.0.csv', index_col=0)['Genes'].apply(lambda x: x.split(', ')).to_dict()


def list_libraries():
    #print(get_config())
    return(load_json(get_config()["LIBRARY_LIST_URL"])["library"])

# https://data.broadinstitute.org/gsea-msigdb/msigdb/release/
# v1.0.6 now have the MSlGDB api to download mouse GiT file with gene symbol/entrezid directory
def load_library(library: str, overwrite: bool = False, verbose: bool = False) -> str:
    if not os.path.exists(get_data_path()+library or overwrite):
        if verbose:
            print("Download Enrichr geneset library")
        urlretrieve(get_config()["LIBRARY_DOWNLOAD_URL"]+library, get_data_path()+library)
    else:
        if verbose:
            print("File cached. To reload use load_library(\""+library+"\", overwrite=True) instead.")
    lib = read_gmt(get_data_path()+library)
    if verbose:
        print("# genesets: "+str(len(lib)))
    return(get_data_path()+library)

def print_libraries():
    """
    Retrieve all names of gene set libraries from Enrichr.

    Returns:
    Array of gene set library names.
    """
    libs = list_libraries()
    for i in range(0, len(libs)):
        print(str(i)+" - "+libs[i])

def read_gmt(gmt_file: str, background_genes: List[str]=[], verbose=False):
    with open(gmt_file, 'r') as file:
        lines = file.readlines()
    library = {}
    background_set = {}
    if len(background_genes) > 1:
        background_genes = [x.upper() for x in background_genes]
        background_set = set(background_genes)
    for line in lines:
        sp = line.strip().split("\t")
        sp2 = [re.sub(",.*", "",value) for value in sp[2:]]
        sp2 = [x.upper() for x in sp2 if x]
        if len(background_genes) > 2:
            geneset = list(set(sp2).intersection(background_set))
            if len(geneset) > 0:
                library[sp[0]] = geneset
        else:
            if len(sp2) > 0:
                library[sp[0]] = sp2
    ugenes = list(set(list(itertools.chain.from_iterable(library.values()))))
    if verbose:
        print("Library loaded. Library contains "+str(len(library))+" gene sets. "+str(len(ugenes))+" unique genes found.")
    return library

def load_json(url):
    context = ssl._create_unverified_context()
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=context) as fr:
        r = fr.read()
    return(json.loads(r.decode('utf-8')))

def get_config():
    config_url = os.path.join(
        os.path.dirname(__file__),
        'db/config.json')
    with open(config_url) as json_file:
        data = json.load(json_file)
    return(data)

def get_data_path() -> str:
    path = os.path.join(
        os.path.dirname(__file__),
        'db/'
    )
    return(path)

def urlretrieve(req, filename):
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(req, context=context) as fr:
        with open(filename, 'wb') as fw:
            shutil.copyfileobj(fr, fw)
