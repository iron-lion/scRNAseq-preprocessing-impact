import glob
import os
import random
import numpy as np
import pandas as pd 
import torch.nn as nn
import torch
from src.model import common as common
from src.model.vae_model import variationalautoencoder as vae
from src.args_parser import get_parser
import src.utils as utils


BARON_DIR = str("./data/scPan/baron/CLASS_WISE/")
MURARO_DIR = str("./data/scPan/muraro/CLASS_WISE/")
XIN_DIR = str("./data/scPan/xin/CLASS_WISE/")
SEG_DIR = str("./data/scPan/segerstolpe/CLASS_WISE/")
WANG_DIR =  str("./data/scPan/wang/CLASS_WISE/")
PAN_LIST = [BARON_DIR, MURARO_DIR, XIN_DIR, SEG_DIR, WANG_DIR]

celltypes = ['epsilon', 'alpha', 'beta', 'duct', 'activated', 'schwann', 'gamma', 'quiescent', 'delta', 'macrophage', 'endothelial', 'acinar', 'mast']

def main(latent, fileout):
    params = get_parser().parse_args()
    print(params)
    params.device = 'cuda:0' if params.cuda and torch.cuda.is_available() else 'cpu'

    # TODO: common gene set
    hp_gene_set = common.hp_symbol_set_load(string_gene_list_pwd = './dataset/human_pancreas_gene_list.txt', delimiter='\n')
    hp_gene_set = list(hp_gene_set)
    hp_gene_set.sort()
    
    ####### ---- #######
    model = vae(params, len(hp_gene_set), [1024], [1024], latent)

    for ro in range(1):
        i = -1
        testgeo = common.geo_data_loader(BARON_DIR, i, hp_gene_set, utils.df_total20000)
        kk = list(testgeo.keys())
        for k in kk:
            if k.split('_')[0] not in celltypes:
                print(k)
                testgeo.pop(k, None)
        #model.fit_dic(testgeo)
        #model.fit_maml(testgeo)
        model.fit_maml_proto(testgeo)

        whole_exp = torch.Tensor()
        whole_key = []
        for target_dir in PAN_LIST:
            print(target_dir)
            i += 1

            testgeo = common.geo_data_loader(target_dir, i, hp_gene_set, utils.df_total20000)
            kk = list(testgeo.keys())
            for k in kk:
                if k.split('_')[0] not in celltypes:
                    print(k)
                    testgeo.pop(k, None)

            a_df, a_label = model.transform_dic(testgeo)
            if 0:#train_dir == target_dir:
                utils.batch_integrate_plot(a_df, a_label, target_dir.split('/')[3] + '_each_HP_vae__log2.total' + str(ro) + '_1024-128.png')
            whole_exp = torch.cat((whole_exp, a_df), 0)
            whole_key += a_label
            print(whole_exp.shape, len(whole_key))

        utils.batch_integrate_plot(whole_exp, whole_key, fileout)
 


if __name__ == '__main__':
    main(128, 'VAE_HP_total_proto_1024_128_1.png')
    main(128, 'VAE_HP_total_proto_1024_128_2.png')
    main(128, 'VAE_HP_total_proto_1024_128_3.png')
    main(128, 'VAE_HP_total_proto_1024_128_4.png')
    main(128, 'VAE_HP_total_proto_1024_128_5.png')
    main(128, 'VAE_HP_total_proto_1024_128_6.png')
    main(128, 'VAE_HP_total_proto_1024_128_7.png')
    main(128, 'VAE_HP_total_proto_1024_128_8.png')
    main(128, 'VAE_HP_total_proto_1024_128_9.png')
    main(128, 'VAE_HP_total_proto_1024_128_0.png')

