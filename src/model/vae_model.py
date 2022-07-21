import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from src.model import basic_model as bmodel
from src.model import common as common
from sklearn.metrics.cluster import adjusted_rand_score


class variationalautoencoder():
    def __init__(self,params, clen, edim, ddim, ldim):
        self.model = VAE(params, clen, edim, ddim, ldim)

    def fit(self, x):
        self.model.init_model()
        self.model.train_vae(x)

    def fit_dic(self, dic: dict):
        self.model.init_model()

        whole_exp = torch.Tensor()
        for key, exp in dic.items():
            x = torch.Tensor(exp.transpose().values.astype(float))
            whole_exp = torch.cat((whole_exp, x), 0)
        self.model.train_vae(whole_exp)

    def fit_maml(self, dic: dict):
        self.model.init_model()
        self.model.train_vae_maml(dic)

    def fit_maml_proto(self, dic: dict):
        self.model.init_model()
        self.model.train_vae_maml_proto(dic)

    def fit_transform(self, x):
        self.model.init_model()
        x = torch.Tensor(x).float()
        self.model.train_vae(x)
        return self.model.encode(x)

    def transform(self, x):
        x = torch.Tensor(x).float()
        mu, logv = self.model.encode(x)
        return mu

    def transform_dic(self, dic:dict):
        whole_exp = torch.Tensor()
        whole_key = []
        for key, exp in dic.items():
            x = torch.Tensor(exp.transpose().values.astype(float))
            whole_exp = torch.cat((whole_exp, self.transform(x)), 0)
            whole_key += [key.lower()] * exp.shape[1]
        return whole_exp, whole_key



class VAE(nn.Module):
    def __init__(self, params, c_len, e_dim = [1024], d_dim = [8], l_dim=16):
        super(VAE,self).__init__()

        self.channel_len = c_len
        self.embeding = e_dim
        if len(d_dim) == 0:
            self.decoding = e_dim[::-1] 
        else:
            self.decoding = d_dim
        self.latent = l_dim

        self.device = params.device
        self.epochs = params.epochs
        self.epochs_test = params.epochs_test
        self.lr = params.learning_rate
        self.lrS = params.lr_scheduler_step
        self.lrG = params.lr_scheduler_gamma
        self.batch_log = params.batch_log

        self.optim_init = False


    def init_model(self):
        print('init! DNN')
        self.feature_encoder = bmodel.VGEPEncoder(self.channel_len, self.embeding, self.latent).to(self.device)
        self.feature_decoder = bmodel.ADecoder(self.channel_len, self.decoding, self.latent).to(self.device)

        bmodel.weights_init(self.feature_encoder)
        bmodel.weights_init(self.feature_decoder)


    def model_train(self):
        self.feature_encoder.train()
        self.feature_decoder.train()


    def model_eval(self):
        self.feature_encoder.eval()
        self.feature_decoder.eval()


    def reparameterize(self, noise, mu, logvar):
        if noise:
            sigma = torch.exp(logvar).to(self.device)
            eps = torch.FloatTensor(logvar.size()[0],1).normal_(0,0.25).to(self.device)
            eps  = eps.expand(sigma.size())
            return mu + sigma * eps
        else:
            return mu


    def forward(self, x):
        mu, logv = self.feature_encoder(x)
        z = self.reparameterize(True, my, logv)
        r = self.feature_decoder(z)
        return r


    def encode(self, x):
        self.model_eval()
        z = self.feature_encoder(x)
        return z


    def init_optim(self, fe_param, fd_param, learning_rate):
        if self.optim_init is False:
            self.fe_optim = torch.optim.Adam(params=fe_param, lr=learning_rate)
            self.fd_optim = torch.optim.Adam(params=fd_param, lr=learning_rate)

            self.feature_encoder_scheduler = StepLR(self.fe_optim,step_size=self.lrS,gamma=self.lrG) # decay LR
            self.feature_decoder_scheduler = StepLR(self.fd_optim,step_size=self.lrS,gamma=self.lrG) # decay LR
            
            self.optim_init = True
        else:
            pass
        
        return self.fe_optim, self.fd_optim,\
                self.feature_encoder_scheduler, self.feature_decoder_scheduler

    
    def train_vae(self, exp, nonneg=False):
        print("> init sc training...")
        loss_log = []

        feature_encoder_optim, feature_decoder_optim \
        , feature_encoder_scheduler, feature_decoder_scheduler \
            = self.init_optim(self.feature_encoder.parameters(), \
                self.feature_decoder.parameters(), self.lr)

        BATCH_SIZE = 64
        li = list(range(exp.shape[0]))
        random.shuffle(li)
 
        self.model_train()
        for epoch in range(self.epochs):
            loss_epoch = []
            for b in range(exp.shape[0]//BATCH_SIZE):
                batch_exp = exp[li[b*BATCH_SIZE:min((b+1)*BATCH_SIZE, exp.shape[0])],:]

                loss = self.vae_recon_loss(batch_exp)
                
                # training
                self.feature_encoder.zero_grad()
                self.feature_decoder.zero_grad()
                  
                loss.backward()
                
                feature_encoder_optim.step()
                feature_decoder_optim.step()

                feature_encoder_scheduler.step()
                feature_decoder_scheduler.step()
                
                # Non-negative encoder
                if nonneg:
                    for par in self.feature_encoder.parameters():
                        par.data.clamp_(0)
                    for par in self.feature_decoder.parameters():
                        par.data.clamp_(0)
                
                loss_epoch.append(loss.cpu().detach().data)

            if (epoch+1)%self.batch_log == 0:
                print("episode:",epoch+1," loss:",np.mean(loss_epoch))
                loss_log.append(np.mean(loss_epoch))

        return loss_log



    def train_vae_maml(self, train_data_dic, nonneg=False):
        print("> init sc training...")
        loss_log = []

        feature_encoder_optim, feature_decoder_optim \
        , feature_encoder_scheduler, feature_decoder_scheduler \
            = self.init_optim(self.feature_encoder.parameters(), \
                self.feature_decoder.parameters(), self.lr)

        self.model_train()        
        for epoch in range(self.epochs):
            # class balance loader
            samples, sample_labels, batches, batch_labels, label_converter \
                = common.sample_test_split(train_data_dic, 5,5,5)
            samples = torch.Tensor(samples.transpose()).float()
            batches = torch.Tensor(batches.transpose()).float()
            batch_labels = torch.LongTensor(batch_labels.values)
            
            loss = self.vae_recon_loss(samples)

            # training
            self.feature_encoder.zero_grad()
            self.feature_decoder.zero_grad()
              
            loss.backward()
            
            feature_encoder_optim.step()
            feature_decoder_optim.step()

            feature_encoder_scheduler.step()
            feature_decoder_scheduler.step()

            # Non-negative encoder
            if nonneg:
                for par in self.feature_encoder.parameters():
                    par.data.clamp_(0)
                for par in self.feature_decoder.parameters():
                    par.data.clamp_(0)
            
            loss_log.append(loss.cpu().detach().data)
            if (epoch+1)%self.batch_log == 0:
                print("episode:",epoch+1," loss:",loss.data)
        
        return loss_log


    def vae_recon_loss(self, samples, use_kld=True):
        #mse = nn.L1Loss(reduction='mean').to(self.device)
        mse = nn.MSELoss().to(self.device)

        v_samples = Variable(samples.float()).to(self.device)

        mu_sample, logvar_sample = self.feature_encoder(v_samples)
        z_from_samples = self.reparameterize(True, mu_sample, logvar_sample)
        recon_samples = self.feature_decoder(z_from_samples)
                
        loss = mse(v_samples, recon_samples)

        if use_kld:
            KLD = (0.5 * torch.sum(1 + logvar_sample - mu_sample.pow(2) - logvar_sample.exp()))
            #KLD /= samples.shape[0] * self.channel_len
            loss -= KLD * 0.00001
         
        return loss

 
    def train_vae_maml_proto(self, train_data_dic, nonneg=False):
        print("> init training...")
        last_acc = 0.0

        feature_encoder_optim, feature_decoder_optim, feature_encoder_scheduler, feature_decoder_scheduler \
                = self.init_optim(self.feature_encoder.parameters(), self.feature_decoder.parameters(), self.lr)

        self.model_train()
        for epoch in range(self.epochs):
            # class balance loader
            samples, sample_labels, batches, batch_labels, label_converter \
                = common.sample_test_split(train_data_dic, 5, 5, 5)
            samples = torch.Tensor(samples.transpose()).float()
            batches = torch.Tensor(batches.transpose()).float()
            batch_labels = torch.LongTensor(batch_labels.values)
            
            #loss = self.recon_loss(samples)
            loss, predict_labels = self.proto_loss(samples, sample_labels, batches, batch_labels)
            #loss+=lossv

            rewards = [1 if predict_labels[j]==batch_labels[j] else 0 for j in range(len(predict_labels))]
            acc = np.sum(rewards) / len(batch_labels)

            # training
            self.feature_encoder.zero_grad()
            self.feature_decoder.zero_grad()
              
            loss.backward()
            
            feature_encoder_optim.step()
            feature_decoder_optim.step()

            feature_encoder_scheduler.step()
            feature_decoder_scheduler.step()
            
            if nonneg:
                # Non-negative encoder
                for par in self.feature_encoder.parameters():
                    par.data.clamp_(0)
                for par in self.feature_decoder.parameters():
                    par.data.clamp_(0)
            
            if (epoch+1)%self.batch_log == 0:
                print("episode:",epoch+1," loss:",loss.data, " acc:", acc)

        return  last_acc


    def proto_loss(self, samples, sample_labels, batches, batch_labels):
        v_samples = Variable(samples.float()).to(self.device)
        v_batches = Variable(batches.float()).to(self.device)
        """
        for k in range(v_samples.shape[0]):
            v_samples[k,] = F.threshold(v_samples[k,], np.quantile(v_samples[k,], 0.5), 0)
        for k in range(v_batches.shape[0]):
            v_batches[k,] = F.threshold(v_batches[k,], np.quantile(v_batches[k,], 0.5), 0)
        """
        mu_sample, logvar_sample = self.feature_encoder(v_samples)
        KLDs = (0.5 * torch.sum(1 + logvar_sample - mu_sample.pow(2) - logvar_sample.exp())).sum(-1).mean()
        mu_batch, logvar_batch = self.feature_encoder(v_batches)
        KLDb = (0.5 * torch.sum(1 + logvar_batch - mu_batch.pow(2) - logvar_batch.exp())).sum(-1).mean()

        z_from_sample = self.reparameterize(True, mu_sample, logvar_sample)
        z_from_batch = self.reparameterize(True, mu_batch, logvar_batch)

        sample_features = z_from_sample.view(5,5, self.latent)
        sample_features = torch.mean(sample_features,1).squeeze(1)
        sample_features_ext = sample_features.repeat(5*5, 1, 1)

        test_features_ext = z_from_batch.unsqueeze(0).repeat(5,1,1)
        test_features_ext = torch.transpose(test_features_ext,0,1)

        # euc distance protonet
        eu_dists = torch.pow(test_features_ext - sample_features_ext, 2).sum(2) # (e * c, c)
        log_p_y = F.log_softmax(-eu_dists, dim=1).view(5, 5, -1)
        loss_val = -log_p_y.gather(2, torch.LongTensor(batch_labels).to(self.device).view(5,5,-1)).squeeze().view(-1).mean()
        _, predict_labels = torch.max(log_p_y, 2)
        
        loss = loss_val - 0.001 * (KLDs+KLDb)

        return loss, predict_labels.view(-1,1)

