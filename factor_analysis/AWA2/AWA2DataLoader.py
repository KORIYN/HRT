# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:23:18 2019

@author: badat
"""

import os,sys
#import scipy.io as sio
import torch
import numpy as np
import h5py
import time
import pickle
from sklearn import preprocessing
from global_setting import NFS_path
#%%
import scipy.io as sio
import pandas as pd
#%%
import pdb
#%%
attr_path =  os.path.join(NFS_path,'attribute/AWA2/new_des.csv')
img_dir = os.path.join(NFS_path,'AWA2/')
mat_path = os.path.join(NFS_path,'xlsa17/data/AWA2/res101.mat')


class AWA2DataLoader():
    def __init__(self, data_path, is_scale = False, is_unsupervised_attr = False,is_balance =True):

        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path

        self.dataset = 'AWA2'
        print('$'*30)
        print(self.dataset)
        print('$'*30)
        self.datadir = self.data_path + 'data/{}/'.format(self.dataset)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_scale = is_scale
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.is_unsupervised_attr = is_unsupervised_attr
        self.read_matdataset()
     

    def read_matdataset(self):

        path= self.datadir + '1feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        print('_____')
        print(path)
        tic = time.clock()
        hf = h5py.File(path, 'r')

        
        if self.is_unsupervised_attr:
            print('Unsupervised Attr')
            class_path = './w2v/{}_class.pkl'.format(self.dataset)
            with open(class_path,'rb') as f:
                w2v_class = pickle.load(f)
            assert w2v_class.shape == (50,300)
            w2v_class = torch.tensor(w2v_class).float()
            
            U, s, V = torch.svd(w2v_class)
            reconstruct = torch.mm(torch.mm(U,torch.diag(s)),torch.transpose(V,1,0))
            print('sanity check: {}'.format(torch.norm(reconstruct-w2v_class).item()))
            
            print('shape U:{} V:{}'.format(U.size(),V.size()))
            print('s: {}'.format(s))
            
            self.w2v_att = torch.transpose(V,1,0).cuda()
            self.att = torch.mm(U,torch.diag(s)).cuda()
            self.normalize_att = torch.mm(U,torch.diag(s)).cuda()
            
        else:
            print('Expert Attr')
            att = np.array(hf.get('att'))
            
            print("threshold at zero attribute with negative value")
            att[att<0]=0
            
            self.att = torch.from_numpy(att).float().cuda()
            
            original_att = np.array(hf.get('original_att'))
            self.original_att = torch.from_numpy(original_att).float().cuda()
            
            w2v_att = np.array(hf.get('w2v_att'))
            self.w2v_att = torch.from_numpy(w2v_att).float().cuda()
            
            self.normalize_att = self.original_att/100
        
        print('Finish loading data in ',time.clock()-tic)