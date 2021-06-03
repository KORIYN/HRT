import numpy as np
import torch 
from CUBDataLoader import CUBDataLoader
from global_setting import NFS_path
import h5py
from sklearn.datasets.samples_generator import make_classification
from sklearn.decomposition import FactorAnalysis
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '3'



def fa(input,k):
    
    input = input.cpu().numpy()

    fa = FactorAnalysis(n_components=k)
    X_reduced = fa.fit_transform(input)

    return X_reduced

if __name__ == '__main__':


    # X, y = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=3, n_informative=2, n_clusters_per_class=1,class_sep =0.5, random_state =10)
    # data = lda(X, y, 2)
    # print(y.shape)
    # exit()
    # X[1000,3] y[1000]

    seed = 214#215#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    
    dataloader = CUBDataLoader(NFS_path,is_unsupervised_attr=False,is_balance=False)
    init_w2v_att = dataloader.w2v_att 


    init_w2v_att_fa = fa(init_w2v_att, 16)
    print(init_w2v_att_fa.shape)

    f1 = h5py.File('CUB_init_w2v_att_fa.hdf5', 'w')
    f1.create_dataset('init_w2v_att_fa', data=init_w2v_att_fa)
    f1.close()

    





