import os
import sys
import random
import pandas as pd
import numpy as np
import logging
import json
import argparse
import pathlib
import h5py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns
import src.utils as my_u
from src.utils import data_transformation
from src.utils import run_plot
from src.utils import h5_data_loader


def run(
    datasets,
    latent_model,
    clustering,
    transformations,
    label_filter,
    seed,
    output,
    force
):
    random.seed(seed)
    
    """Data transformation test"""
    X_, y_, b_, file_names = h5_data_loader(datasets, label_filter)
    logging.info(f'Data loaded. {datasets}')
  
    # initialize latent mapper
    if (latent_model == 'tsne'):
        latent_space = TSNE(n_components=2)
    elif (latent_model == 'umap'):
        latent_space = UMAP(n_components=2, init='spectral', random_state=seed)
    else:
        assert(0, 'ldim')


    for it in transformations:
        its = it.split('_')
        this_file_names = file_names + '_' + latent_model + '_' + clustering + '_' + str(seed)
        for it_it in its:
            X_ = data_transformation[it_it](X_)
            this_file_names = this_file_names + '_' + it_it
        
        out_file = f'{output}/{this_file_names}.png'
        if (not force and os.path.exists(out_file)):
            logging.info(f'Skipped: {out_file}')
            continue

        plt.figure(figsize=(2,2), dpi=300)
        ax00 = plt.subplot2grid((1,1), (0,0))

        clu_res = run_plot(X_, ax00, y_, latent_space, clustering)
        ax00.legend(bbox_to_anchor=(1.1,0), loc='lower left',borderaxespad=0)
        logging.info(f'Result summary : {clu_res}')

        plt.savefig(out_file, bbox_inches='tight')
        logging.info(f'Saved [{it}]: {out_file}')


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.INFO,
        format='%(asctime)s %(message)s'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-D', '--datasets',
        nargs='+',
        default=['Baron'],
        type=str,
        help='Single cell RNA sequencing dataset(s)'
    )

    parser.add_argument(
        '-S', '--seed',
        type=int,
        default=1,
        help='Random seed for the reproducibility'
    )

    parser.add_argument(
        '-L', '--latent-model',
        default='umap',
        help='Selects dimensional reduction model'
    )

    parser.add_argument(
        '-C', '--clustering-model',
        default='dbscan',
        help='Selects clustering model'
    )

    parser.add_argument(
        '-T', '--transformations',
        nargs='+',
        default=['total'],
        type=str,
        help='Data transformation method(s)'
    )

    parser.add_argument(
        '-F', '--label-filter',
        nargs='+',
        default=[],
        type=str,
        help='Use only choosen cell-type(s)'
    )

    folder_name = 'preprocessing'

    parser.add_argument(
        '-o', '--output',
        default=pathlib.Path(__file__).resolve().parent / 'results'
                                                        / folder_name,
        type=str,
        help='Output path for storing the results.'
    )

    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='If set, overwrites all files. Else, skips existing files.'
    )

    args = parser.parse_args()

    # Create the output directory for storing all results of the
    # individual combinations.
    os.makedirs(args.output, exist_ok=True)

    logging.info(f'Dataset(s): {args.datasets}')
    logging.info(f'Latent model: {args.latent_model}')
    logging.info(f'Clustering method: {args.clustering_model}')
    logging.info(f'Normalization method: {args.transformations}')
    logging.info(f'Cell type(s): {args.label_filter}')
    logging.info(f'Seed: {args.seed}')
    logging.info(f'Output folder: {args.output}')
    logging.info(f'Override: {args.force}')


    run(
        args.datasets,
        args.latent_model,
        args.clustering_model,
        args.transformations,
        args.label_filter,
        args.seed,
        args.output,
        args.force
    )
