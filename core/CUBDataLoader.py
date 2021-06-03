# -*- coding: utf-8 -*-
import os,sys
import torch
import numpy as np
import h5py
import time
import pickle
import pdb
from sklearn import preprocessing
from global_setting import NFS_path
#%%
import scipy.io as sio
import pandas as pd
import random

attr_path =  os.path.join(NFS_path,'attribute/CUB/new_des.csv')
img_dir = os.path.join(NFS_path,'data/CUB/')
mat_path = os.path.join(NFS_path,'data/xlsa17/data/CUB/res101.mat')



class CUBDataLoader():
    def __init__(self, data_path,  is_scale = False,is_unsupervised_attr = False,is_balance=True):

        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        
        self.dataset = 'CUB'
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
    
    def augment_img_path(self,mat_path=mat_path,img_dir=img_dir):
        self.matcontent = sio.loadmat(mat_path)
        self.image_files = np.squeeze(self.matcontent['image_files'])
        
        def convert_path(image_files,img_dir):
            new_image_files = []
            for idx in range(len(image_files)):
                image_file = image_files[idx][0]
                image_file = os.path.join(img_dir,'/'.join(image_file.split('/')[5:]))
                new_image_files.append(image_file)
            return np.array(new_image_files)
        
        self.image_files = convert_path(self.image_files,img_dir)
        
        path= self.datadir + '1feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        hf = h5py.File(path, 'r')
        
        trainval_loc = np.array(hf.get('trainval_loc'))
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))
        
        self.data['train_seen']['img_path'] = self.image_files[trainval_loc]
        self.data['test_seen']['img_path'] = self.image_files[test_seen_loc]
        self.data['test_unseen']['img_path'] = self.image_files[test_unseen_loc]
        
        self.attr_name = pd.read_csv(attr_path)['new_des']
        
        
        
    def next_batch_img(self, batch_size,class_id,is_trainset = False):
        features = None
        labels = None
        img_files = None
        if class_id in self.seenclasses:
            if is_trainset:
                features = self.data['train_seen']['resnet_features']
                labels = self.data['train_seen']['labels']
                img_files = self.data['train_seen']['img_path']
            else:
                features = self.data['test_seen']['resnet_features']
                labels = self.data['test_seen']['labels']
                img_files = self.data['test_seen']['img_path']
        elif class_id in self.unseenclasses:
            features = self.data['test_unseen']['resnet_features']
            labels = self.data['test_unseen']['labels']
            img_files = self.data['test_unseen']['img_path']
        else:
            raise Exception("Cannot find this class {}".format(class_id))
        
        #note that img_files is numpy type !!!!!
        
        idx_c = torch.squeeze(torch.nonzero(labels == class_id))
        
        features = features[idx_c]
        labels = labels[idx_c]
        img_files = img_files[idx_c.cpu().numpy()]
        
        batch_label = labels[:batch_size].cuda()
        batch_feature = features[:batch_size].cuda()
        batch_files = img_files[:batch_size]
        batch_att = self.att[batch_label].cuda()
        
        return batch_label, batch_feature,batch_files, batch_att
    

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
            idx = idx.numpy().tolist()
        idx1 = random.sample(idx,batch_size//2)
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

        path = self.datadir + '1feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
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
            temp = np.array(hf.get('att'))
            print(w2v_class.shape,temp.shape)

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

        path_fa = self.data_path +'factor_analysis/CUB_init_w2v_att_fa.hdf5'
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


