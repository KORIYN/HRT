import numpy as np
import torch 
from SUN import SUNDataLoader
from global_setting import NFS_path
import h5py
from sklearn.datasets.samples_generator import make_classification
from sklearn.decomposition import FactorAnalysis
import os 
from setproctitle import setproctitle
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def fa(input,k):
    
    input = input.cpu().numpy()

    fa = FactorAnalysis(n_components=k)
    X_reduced = fa.fit_transform(input)

    return X_reduced

if __name__ == '__main__':



    seed = 214#215#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    
    dataloader = SUNDataLoader(NFS_path,is_unsupervised_attr=False,is_balance=False)

    init_w2v_att = dataloader.w2v_att #[102,300]

    init_w2v_att_fa = fa(init_w2v_att, 16)
    print(init_w2v_att_fa.shape)

    f1 = h5py.File('SUN_init_w2v_att_fa.hdf5', 'w')
    f1.create_dataset('init_w2v_att_fa', data=init_w2v_att_fa)
    f1.close()
    





