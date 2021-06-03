#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os,sys
#import scipy.io as sio
import torch
import numpy as np
import h5py
import time
import pickle
from sklearn import preprocessing
from core.helper_func import mix_up
#%%
import pdb
import pandas as pd
import random
#%%

class SUNDataLoader():
    def __init__(self, data_path,  is_scale = False, is_unsupervised_attr = False,is_balance=True):

        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.dataset = 'SUN'
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
        self.get_idx_classes()
        self.I = torch.eye(self.allclasses.size(0)).cuda()

    def next_batch(self, batch_size):
        if self.is_balance:
            idx = []
            n_samples_class = max(batch_size //self.ntrain_class,1)
            sampled_idx_c = np.random.choice(np.arange(self.ntrain_class),min(self.ntrain_class,batch_size),replace=False).tolist()
            for i_c in sampled_idx_c:
                idxs = self.idxs_list[i_c]
                idx.append(np.random.choice(idxs,n_samples_class))
            idx = np.concatenate(idx)
            idx = torch.from_numpy(idx)
        else:
            idx = torch.randperm(self.ntrain)[0:batch_size]
        
        shape = idx.shape[0]
        idx = idx.numpy().tolist()
        idx1 = random.sample(idx,shape//2)
        idx2 = list(set(idx).difference(set(idx1)))
        idx1 = torch.tensor(idx1)
        idx2 = torch.tensor(idx2)

        batch_feature1 = self.data['train_seen']['resnet_features'][idx1].cuda()
        batch_label1 =  self.data['train_seen']['labels'][idx1].cuda()

        batch_feature2 = self.data['train_seen']['resnet_features2'][idx2].cuda()
        batch_label2 =  self.data['train_seen']['labels'][idx2].cuda()
        
        batch_feature = torch.cat((batch_feature1,batch_feature2),dim = 0)
        batch_label = torch.cat((batch_label1,batch_label2),dim = 0)

        #shuffle
        num = len(batch_label)
        order = torch.randperm(num)
        batch_feature = batch_feature[order,...]
        batch_label = batch_label[order]

        batch_att = self.att[batch_label].cuda()


        return batch_label, batch_feature, batch_att
    
    def get_idx_classes(self):
        n_classes = self.seenclasses.size(0)
        self.idxs_list = []
        train_label = self.data['train_seen']['labels']
        for i in range(n_classes):
            idx_c = torch.nonzero(train_label == self.seenclasses[i].cpu()).cpu().numpy()
            idx_c = np.squeeze(idx_c)
            self.idxs_list.append(idx_c)
        return self.idxs_list
    

        
    def read_matdataset(self):

        path= self.datadir + '1feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        path2 = self.datadir + '2feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        print('_____')
        print(path)


        tic = time.clock()
        hf = h5py.File(path, 'r')
        hf2 = h5py.File(path2, 'r')
        features = np.array(hf.get('feature_map'))
        features2 = np.array(hf2.get('feature_map'))

        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))
        
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
            self.att = torch.from_numpy(att).float().cuda()
            
            original_att = np.array(hf.get('original_att'))
            self.original_att = torch.from_numpy(original_att).float().cuda()
            
            w2v_att = np.array(hf.get('w2v_att'))
            self.w2v_att = torch.from_numpy(w2v_att).float().cuda()
            
            self.normalize_att = self.original_att/100
        
        print('Finish loading data in ',time.clock()-tic)
        
        train_feature = features[trainval_loc]
        test_seen_feature = features[test_seen_loc]
        test_unseen_feature = features[test_unseen_loc]

        train_feature2 = features2[trainval_loc]

        if self.is_scale:
            scaler = preprocessing.MinMaxScaler()
    
            train_feature = scaler.fit_transform(train_feature)
            test_seen_feature = scaler.fit_transform(test_seen_feature)
            test_unseen_feature = scaler.fit_transform(test_unseen_feature)

        train_feature = torch.from_numpy(train_feature).float()
        test_seen_feature = torch.from_numpy(test_seen_feature) 
        test_unseen_feature = torch.from_numpy(test_unseen_feature) 

        train_feature2 = torch.from_numpy(train_feature2).float()

        self.train_label = torch.from_numpy(labels[trainval_loc]).long()
        self.test_unseen_label = torch.from_numpy(labels[test_unseen_loc])
        self.test_seen_label = torch.from_numpy(labels[test_seen_loc])

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.cpu().numpy())).cuda()
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.cpu().numpy())).cuda()
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()


        self.att_test_seen = self.att[self.test_seen_label]
        self.att_test_unseen = self.att[self.test_unseen_label]


        path_fa = self.data_path +'factor_analysis/SUN_init_w2v_att_fa.hdf5'
        hf_fa = h5py.File(path_fa, 'r')
        w2v_att_fa = np.array(hf_fa.get('init_w2v_att_fa'))
        self.w2v_att_fa = torch.from_numpy(w2v_att_fa).float().cuda() 

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['resnet_features2'] = train_feature2
        self.data['train_seen']['labels']= self.train_label


        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['resnet_features_pca'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = self.test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen']['labels'] = self.test_unseen_label
