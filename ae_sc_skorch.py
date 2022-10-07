import glob
import os
import random
import logging
import pathlib
import argparse
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch import NeuralNetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from umap import UMAP
from src.model import common as common
from src.model.skorch_ae_model import AE
from src.utils import clustering
from src.utils import data_transformation
from src.utils import h5_data_loader


class AutoEncoderNet(NeuralNetRegressor):
    def __init__(
                self,
                module,
                latent,
                clustering,
                *args,
                criterion=torch.nn.MSELoss,
                **kwargs
        ):
        super(AutoEncoderNet, self).__init__(
            module,
            *args,
            criterion=criterion,
            **kwargs
        )       
        """ TSNE / UMAP """
        self.latent = latent
        self.clustering = clustering

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """ Autoencoder Loss """
        z, r = y_pred
        loss_reconstruction = super().get_loss(r, y_true, *args, **kwargs)
        loss_l1 = 1e-3 * torch.abs(z).sum()
        return loss_reconstruction + loss_l1

    def score(self, X, y, *args, **kwargs):
        z, _ = super().forward(X)
        latent_vector = self.latent.fit_transform(z)
        best, _ = clustering(latent_vector, y, cluster=self.clustering)
        logging.info(f'{best[1]}')
        return best[1]

    def fit(self, X, y):
        return super().fit(X, X)


def run(
        train_datasets,
        test_datasets,
        latent_model,
        clustering_model,
        transformations,
        label_filter,
        grid_cv,
        seed,
        output,
        force
    ):
    random.seed(seed)

    X_train, y_train, b_train, train_file_names = h5_data_loader(train_datasets, label_filter)
    X_test, y_test, b_test, test_file_names = h5_data_loader(test_datasets, label_filter)
    
    # Dataset align
    common_gene = X_train.columns.intersection(X_test.columns)
    if len(common_gene) <= 0:
        logging.warning(f'Train & Test has 0 common gene symbols')
        return
    feature_length = len(common_gene)
    X_train = X_train.filter(items=common_gene,axis=1).to_numpy().astype(np.float32)
    X_test = X_test.filter(items=common_gene,axis=1).to_numpy().astype(np.float32)

    for it in transformations:
        logging.info(f'Transformation: {it}. running...')
        its = it.split('_')
        for it_it in its:
            X_train = data_transformation[it_it](X_train)
            X_test = data_transformation[it_it](X_test)

        # initialize latent mapper
        if (latent_model == 'tsne'):
            latent_space = TSNE(n_components=2)
        elif (latent_model == 'umap'):
            latent_space = UMAP(n_components=2, init='spectral', random_state=seed)
        else:
            assert(0, 'ldim')

        ####### ---- #######
        net = AutoEncoderNet(
            AE,
            latent = latent_space,
            clustering = clustering_model,
            optimizer = torch.optim.Adam,
            batch_size = 32,
            module__c_len = feature_length,
            module__e_dim = [512],
            module__d_dim = [512],
            module__l_dim = 128,
            max_epochs=40,
            lr=0.1,
            iterator_train__shuffle=True,
            device = 'cuda:'+str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
        )
        
        # deactivate skorch-internal train-valid split and verbose logging
        net.set_params(train_split=False, verbose=0)

        """ Grid Search """
        params = {
            'lr': [0.0005, 0.001],
            'max_epochs': [20,40],
            'batch_size' : [32, 64],
            'module__e_dim': [[1024], [512]],
            'module__l_dim': [128, 64],
        }
       
        gs = GridSearchCV(net, params, refit=False, cv=StratifiedKFold(n_splits=grid_cv).split(X_train, y_train))

        gs.fit(X_train, y_train)
        logging.info(f'Grid Search {it} done: {gs.best_params_}')
        logging.info(f'Grid Search {it} done: {gs.best_score_}')


        net = AutoEncoderNet(
            AE,
            latent = latent_space,
            clustering = clustering_model,
            optimizer = torch.optim.Adam,
            batch_size = 32,
            module__c_len = feature_length,
            **gs.best_params_,
            iterator_train__shuffle=True,
            device = 'cuda:'+str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
        )
        net.set_params(train_split=False, verbose=2)

        net.fit(X_train, y_train)
        
        # TODO: net.score(X_test, y_test) ?
        encoded_pred = net.predict(X_test)
        latent = latent_space.fit_transform(encoded_pred)
        best, _ = clustering(latent, y_test, cluster=clustering_model)
        logging.info(f'Clustering result {it}: {best}')
        

if __name__ == '__main__':
    logging.basicConfig(
        level = logging.INFO,
        format='%(asctime)s %(message)s'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-datasets',
        nargs='+',
        default=['Baron'],
        type=str,
        help='For training, Single-cell RNA sequencing dataset(s) - h5py file path'
    )

    parser.add_argument(
        '--test-datasets',
        nargs='+',
        default=['Baron'],
        type=str,
        help='For testing, Single-cell RNA sequencing dataset(s) - h5py file path'
    )

    parser.add_argument(
        '-S', '--seed',
        type=int,
        default=1,
        help='Random seed for the reproducibility'
    )

    parser.add_argument(
        '-G', '--grid-cv',
        type=int,
        default=5,
        help='Cross Validation parameter in GridSearchCV'
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

    logging.info(f'Train Dataset(s): {args.train_datasets}')
    logging.info(f'Test Dataset(s): {args.test_datasets}')
    logging.info(f'Latent model: {args.latent_model}')
    logging.info(f'Clustering method: {args.clustering_model}')
    logging.info(f'Normalization method: {args.transformations}')
    logging.info(f'Cell type(s): {args.label_filter}')
    logging.info(f'Grid CV: {args.grid_cv}')
    logging.info(f'Seed: {args.seed}')
    logging.info(f'Output folder: {args.output}')
    logging.info(f'Override: {args.force}')

    run(
        args.train_datasets,
        args.test_datasets,
        args.latent_model,
        args.clustering_model,
        args.transformations,
        args.label_filter,
        args.grid_cv,
        args.seed,
        args.output,
        args.force
    )
