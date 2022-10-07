import pandas as pd
import numpy as np
import os
import glob
import sys
import copy
import h5py
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
from sklearn.preprocessing import normalize as l2norm

import matplotlib.pyplot as plt
import seaborn as sns


"""
Set of Pre-processing methods

Log2
total20000
    each of the cell could have different number of total read count
    make sum of reads count to 20,000
minmax
    simple min max normalization to make expression level in 0~1 range.
minmax_scaler
    sklearn implementation
    use std for further normalization

"""
def df_cp(x):
    return copy.deepcopy(x)

def df_log(x):
    return np.log2(x+1.0)

def df_total20000_old(x):
    x = x.divide(x.sum(1), axis = 0).mul(20000)
    x = x.replace(np.nan,0)
    return x

def df_total20000(x):    
    np.nan_to_num(x,0)
    x = (x.T/x.sum(axis=1)).T*20000
    np.nan_to_num(x,0)
    return x


def df_minmax(x):
    x = ((x.transpose() - x.min(1))/(x.max(1)-x.min(1)))
    x = x.transpose()
    x = x.replace(np.nan,0)
    return x

def df_minmax_scaler(x):
    x = np.transpose(min_max_scaler.fit_transform(x.transpose()))
    return x

def df_l2norm(x):
    x = l2norm(x, axis=1, copy=True)
    return x

def df_zscore(x):
    x = ((x.transpose() - x.mean(1))/(x.std(1)))
    x = x.transpose()
    x = x.replace(np.nan,0)
    return x

def df_meansquare(x):
    m = np.sqrt(np.power(x,2).mean(1))
    x = x.transpose()/m
    x = x.transpose()
    x = x.replace(np.nan,0)
    return x

data_transformation = {
    "raw" : df_cp,
    "log" : df_log,
    "total" : df_total20000,
    "minmax" : df_minmax_scaler,
    "l2norm" : df_l2norm,
    "zscore" : df_zscore,
    "meansquare" : df_meansquare,
}


def h5_data_loader(datasets, label_filter=None):
    """Data transformation test"""
    X_, y_, b_, file_names = [], [], [], []
    
    for dataset in datasets:
        file_prefix = dataset.split('/')[-1].split('.')[0]
        file_names.append(file_prefix)

        # h5 data load
        hf = h5py.File(dataset, 'r')
        
        total_data = pd.DataFrame(hf['data'][:])
        total_data.columns = hf['column'][:].astype('str')
        total_data.index = hf['index'][:].astype('str')
        total_data = total_data.loc[:,~total_data.columns.duplicated(keep='first')]

        labels = np.char.lower(np.array(hf['label'][:,0].astype('str').tolist()))
        # TODO: better label cleaning
        labels = np.array([x.split('_')[0] for x in labels])
        
        blabels = np.array([file_prefix]*len(labels))

        if label_filter is not None and len(label_filter) > 1:
            baron_filter = [True if x in label_filter else False for x in labels]
            total_data = total_data.loc[baron_filter]
            labels = labels[baron_filter]
            blabels = blabels[baron_filter]

        X_.append(total_data)
        y_.append(labels)
        b_.append(blabels)

        # h5 close
        hf.close()
    del(total_data, labels, blabels)

    if len(X_) > 1:
        common_gene = X_[0].columns
        for x in X_[1:]:
            common_gene = common_gene.intersection(x.columns)
        
        if (len(common_gene) <= 0):
#            logging.info(f'Gene symbol mismatch!')
            exit(-1)
 
        X_ = [x.filter(items=common_gene, axis=1) for x in X_]
        X_ = pd.concat(X_)
        y_, b_ = map(
            np.concatenate, [y_, b_]
        )
        file_names = '_'.join(file_names)
    else:
        X_ = X_[0]
        y_ = y_[0]
        file_names = file_names[0]

    return X_, y_, b_, file_names



def run_plot(exp, ax, labels, latent_space, cluster, blabels=None, b_ax=None):
    """
    Project exp to latent space.
    
    exp -> latent_space -> latent_vector
    latent_vector -> cluster -> pred_label
    
    x = 
    ARI: pred_label ~ labels
    Silhouette: latent_vector ~ pred_label
    """
    raw_result, raw_df = cluster_on_latent(exp, labels, latent_space, cluster, blabels=blabels)
    draw_plot(raw_df, raw_result, ax, labels)
    if ((blabels != None) and (b_ax != None)):
        draw_plot(raw_df, raw_result, b_ax, blabels, hue_='batch')

    del(raw_df)
    return raw_result


def batch_integrate_plot(whole_exp, whole_key, filename='batch_integration.png'):
    """
    Draw a 5x2 plot
    
    [t-sne, UMAP]
        x   [
            Cell Type,
            Batch Type,
            k-means with number of original class,
            k-means,
            dbscan
            ]
    """
    ####### ---- PLOT ---- #######
    plt.figure(figsize=(16,8), dpi=300)
    ax00 = plt.subplot2grid((2,4), (0,0)) 
    ax10 = plt.subplot2grid((2,4), (0,1))  
    ax20 = plt.subplot2grid((2,4), (0,2))  
    ax30 = plt.subplot2grid((2,4), (0,3))  

    ax01 = plt.subplot2grid((2,4), (1,0)) 
    ax11 = plt.subplot2grid((2,4), (1,1))  
    ax21 = plt.subplot2grid((2,4), (1,2))  
    ax31 = plt.subplot2grid((2,4), (1,3))  


    cell_type_labels = [ x.split('_')[0] for x in whole_key]
    print(set(cell_type_labels))
    batch_type_labels = [ x.split('_')[-1] for x in whole_key]

    text_arg='test_baron'
    tsne_df = TSNE(n_components=2).fit_transform(whole_exp.cpu().detach().data)
    #tsne_df = TSNE(n_components=2, init='pca').fit_transform(whole_exp.cpu().detach().data)

    df=pd.DataFrame()
    df['tsne1'] = tsne_df[:,0]
    df['tsne2'] = tsne_df[:,1]

    df['label'] = batch_type_labels
    draw_plot(df, (0,0,0), ax10, df['label'], txt_=False)

    df['label'] = cell_type_labels   
    draw_plot(df, (0,0,0), ax00, df['label'], txt_=False)
        

    best, best_kmeans_label = clustering(tsne_df, cell_type_labels, batch_labels=batch_type_labels, cluster='kmean')
    print('kmean tsne', best)
    df['pred'] = [str(x) for x in best_kmeans_label]
    draw_plot(df, best, ax20, df['pred'], 'pred')
    
    best, best_kmeans_label = clustering(tsne_df, cell_type_labels, batch_labels=batch_type_labels, cluster='dbscan')
    print('dbscan tsne',best)
    df['pred'] = [str(x) for x in best_kmeans_label]
    draw_plot(df, best, ax30, df['pred'], 'pred')

    ##### ---- UMAP ---- #####
    umap_2d = UMAP(n_components=2, init='spectral', random_state=0)
    tsne_df = umap_2d.fit_transform(whole_exp.cpu().detach().data)
    df=pd.DataFrame()
    df['tsne1'] = tsne_df[:,0]
    df['tsne2'] = tsne_df[:,1]

    df['label'] = batch_type_labels
    draw_plot(df, (0,0,0), ax11, df['label'], txt_=False)
    df['label'] = cell_type_labels
    draw_plot(df, (0,0,0), ax01, df['label'], txt_=False)
    

    best, best_kmeans_label = clustering(tsne_df, cell_type_labels, batch_labels=batch_type_labels, cluster='kmean')
    print('kmean umap',best)
    df['pred'] = [str(x) for x in best_kmeans_label]
    draw_plot(df, best, ax21, df['pred'], 'pred')

    best, best_kmeans_label = clustering(tsne_df, cell_type_labels, batch_labels=batch_type_labels, cluster='dbscan')
    print('dbscan umap',best)
    df['pred'] = [str(x) for x in best_kmeans_label]
    draw_plot(df, best, ax31, df['pred'], 'pred')
 
    ax00.set_ylabel('t-sne' , fontsize=14)
    ax01.set_ylabel('umap' , fontsize=14)

    ax01.set_xlabel('cell type', fontsize=13)
    ax11.set_xlabel('batch', fontsize=13)
    ax21.set_xlabel('k-mean', fontsize=13)
    ax31.set_xlabel('dbscan', fontsize=13)
    #ax00.legend(bbox_to_anchor=(0,0), loc='lower right',borderaxespad=0) 
    ax01.legend(bbox_to_anchor=(-0.2,0), loc='lower right',borderaxespad=0) 
    plt.savefig(filename)
    plt.close()


def clustering(latent_df, labels, batch_labels=None, cluster='kmean'):
    best = (0,0,0,10)
    best_kmeans_label = []
    num_clus = len(set(labels))
    if cluster == 'kmean':
        best_kmeans_label = [1] * len(labels)
        for kn in range(num_clus//2, num_clus+4):
            kmeans = KMeans(n_clusters = kn, n_init=20, max_iter=50).fit(latent_df)

            test_ari = adjusted_rand_score(kmeans.labels_, labels)
            batch_ari = (adjusted_rand_score(kmeans.labels_, batch_labels) \
                        if batch_labels != None else 0)
            sil = (silhouette_score(latent_df, kmeans.labels_) \
                        if (len(set(kmeans.labels_)) > 1) else 0)
            
            if best[1] < test_ari:
                best = (len(set(kmeans.labels_)), test_ari, sil, batch_ari)
                best_kmeans_label = kmeans.labels_
            del(kmeans)
        
    elif cluster == 'dbscan':
        best_kmeans_label = [1] * len(labels)
        for kn in range(1,20):
            kmeans = DBSCAN(eps=0.5 * kn, min_samples=8).fit(latent_df)
            
            test_ari = adjusted_rand_score(kmeans.labels_, labels)
            batch_ari = (adjusted_rand_score(kmeans.labels_, batch_labels) \
                        if batch_labels != None else 0)
            sil = (silhouette_score(latent_df, kmeans.labels_) \
                        if (len(set(kmeans.labels_)) > 1) else 0)
            
            if best[1] < test_ari:
                best = (len(set(kmeans.labels_)), test_ari, sil, batch_ari)
                best_kmeans_label = kmeans.labels_
            del(kmeans)
    else :
        pass
    #print(cluster, '#cluster:', best)
    return best, best_kmeans_label
           



def cluster_on_latent(whole_exp, labels, projector, cluster='kmean', blabels=None):
    i = 0
    best = (0,0,0)
    latent_df = projector.fit_transform(whole_exp)

    df=pd.DataFrame()
    df['tsne1'] = latent_df[:,0]
    df['tsne2'] = latent_df[:,1]
    df['label'] = labels
    df['batch'] = blabels

    best, _ = clustering(latent_df, labels, blabels, cluster)
    #print(cluster, '#cluster:', best)
    return best, df
           



def tsne_get(whole_exp, labels, cluster='kmean'):
    i = 0
    best = (0,0,0)
    tsne_df = TSNE(n_components=2).fit_transform(whole_exp)
    #tsne_df = TSNE(n_components=2, init='pca').fit_transform(whole_exp.cpu().detach().data)
    #tsne_df = whole_exp

    df=pd.DataFrame()
    df['tsne1'] = tsne_df[:,0]
    df['tsne2'] = tsne_df[:,1]
    df['label'] = labels

    if cluster:
        best_kmeans_label = []
        #print(whole_exp, whole_exp.shape, len(whole_key))
        for kn in range(2,len(set(labels))+4):
            if cluster == 'kmean':
                kmeans = KMeans(n_clusters = kn, n_init=20, max_iter=50).fit(tsne_df)
            elif cluster == 'dbscan':
                kmeans = DBSCAN(eps=0.5 * kn, min_samples=10).fit(tsne_df)

            test_ari = adjusted_rand_score(kmeans.labels_, labels)
            if (len(set(kmeans.labels_)) > 1):
                sil = silhouette_score(tsne_df, kmeans.labels_)
            if best[1] < test_ari:
                best = (len(set(kmeans.labels_)), test_ari, sil)
                best_kmeans_label = kmeans.labels_

            #print(Counter(kmeans.labels_), Counter(whole_key))
        print(cluster, '#cluster:', best)
    return best, df


def draw_plot(df_, result_, axs_, label_, hue_='label', txt_=True, a=0.6, m='o'):
    sns.scatterplot(
        x="tsne1", y="tsne2",
        size=5,
        sizes=(5,10),
        data=df_,
        legend="full",
        hue=hue_,
        hue_order=sorted(set(label_)),
        alpha=a,
        linewidth=(0 if m=='o' else 0.5),
        edgecolors='none',
        ax=axs_,
        marker=m,
        )
    axs_.get_legend().remove()
    axs_.spines['top'].set_visible(False)
    axs_.spines['right'].set_visible(False)
    axs_.get_xaxis().set_ticks([])
    axs_.get_yaxis().set_ticks([])
    axs_.set_xlabel('')
    axs_.set_ylabel('')

    if txt_:
        if len(result_) == 4:
            txt = "ARI: {ari:.3f}\nSilhouette: {sil:.3f}\nbARI: {bari:.3f}\nClusters: {cl:d}"
            axs_.text(min(df_['tsne1']),max(df_['tsne2']),txt.format(ari = result_[1], sil=result_[2], bari=result_[3], cl=result_[0]), fontsize=10, horizontalalignment='left', verticalalignment='bottom')
        else:
            txt = "ARI: {ari:.3f}\nSilhouette: {sil:.3f}\nClusters: {cl:d}"
            axs_.text(min(df_['tsne1']),max(df_['tsne2']),txt.format(ari = result_[1], sil=result_[2], cl=result_[0]), fontsize=10, horizontalalignment='left', verticalalignment='bottom')

