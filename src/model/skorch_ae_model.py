import glob
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from src.model import rl_model as L2C
from src.model import common as common
from sklearn.metrics.cluster import adjusted_rand_score


class AE(nn.Module):
    def __init__(self, c_len, e_dim = [1024], d_dim = [1024], l_dim=128):
        super(AE, self).__init__()

        self.channel_len = c_len
        self.embeding = e_dim
        if len(d_dim) == 0:
            self.decoding = e_dim[::-1] 
        else:
            self.decoding = d_dim
        self.latent = l_dim

        self.feature_encoder = L2C.AEncoder(self.channel_len, self.embeding, self.latent, bn=True)
        self.feature_decoder = L2C.ADecoder(self.channel_len, self.decoding, self.latent, bn=True)

        L2C.weights_init(self.feature_encoder)
        L2C.weights_init(self.feature_decoder)


    def model_train(self):
        self.feature_encoder.train()
        self.feature_decoder.train()


    def model_eval(self):
        self.feature_encoder.eval()
        self.feature_decoder.eval()


    def forward(self, x):
        z = self.feature_encoder(x)
        r = self.feature_decoder(z)
        return z, r

