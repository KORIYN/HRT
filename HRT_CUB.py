# coding: utf-8
# In[1]:

import os,sys
pwd = os.getcwd()

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
parent = '/'.join(pwd.split('/')[:-1])
sys.path.insert(0,parent)
os.chdir(parent)

#%%
print('-'*30)
print(os.getcwd())
print('-'*30)

#%%
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from core.model import HRT
from core.CUBDataLoader import CUBDataLoader
from core.helper_func import eval_zs_gzsl,visualize_attention,eval_zs_gzsl,eval_zs_gzsl_loss#,get_attribute_attention_stats
from core.utils import init_log, progress_bar ,adjust_learning_rate
from global_setting import NFS_path
from visual import Logger
import importlib
import pdb
import numpy as np
from torch.nn import DataParallel

# In[3]:

torch.backends.cudnn.benchmark = True


# In[4]:

dataloader = CUBDataLoader(NFS_path,is_unsupervised_attr=False,is_balance=False)


# In[5]:

dataloader.augment_img_path()


# In[6]:

def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return lr


# In[ ]:

seed = 214#215#
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

batch_size = 50
nepoches = 200
dim_f = 2048
dim_v = 300 #词向量的维度

niters = dataloader.ntrain * nepoches//batch_size  
init_w2v_att = dataloader.w2v_att 
init_w2v_att_fa = dataloader.w2v_att_fa
att = dataloader.att 
normalize_att = dataloader.normalize_att 


trainable_w2v = True
lambda_ = 0.1#0.1
bias = 0
prob_prune = 0
uniform_att_1 = False
uniform_att_2 = False

train_numbers = len(dataloader.train_label)


seenclass = dataloader.seenclasses 
unseenclass = dataloader.unseenclasses

desired_mass = 1
report_interval = niters//nepoches 
data = 'CUB'

model = HRT(data,dim_f,dim_v,init_w2v_att,init_w2v_att_fa,att,normalize_att,
            seenclass,unseenclass,
            lambda_,
            trainable_w2v,normalize_V=False,normalize_F=True,is_conservative=True,
            uniform_att_1=uniform_att_1,uniform_att_2=uniform_att_2,
            prob_prune=prob_prune,desired_mass=desired_mass, is_conv=False,
            is_bias=True)
# model = DataParallel(model)
model.cuda()

setup = {'pmp':{'init_lambda':0.1,'final_lambda':0.1,'phase':0.8},
         'desired_mass':{'init_lambda':-1,'final_lambda':-1,'phase':0.8}}
print(setup)


params_to_update = []
params_names = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        params_names.append(name)
        print("\t",name)

#%%
lr = 0.0001
weight_decay = 0.0001#0.000#0.#
momentum = 0.9#0.#
#%%
lr_seperator = 1
lr_factor = 1
print('default lr {} {}x lr {}'.format(params_names[:lr_seperator],lr_factor,params_names[lr_seperator:]))

mode = 'MultiStep33'


if mode == 'normal':
    optimizer  = optim.RMSprop( params_to_update ,lr=lr,weight_decay=weight_decay, momentum=momentum)

elif mode == 'MultiStep33':
    optimizer = optim.RMSprop( params_to_update ,lr=lr,weight_decay=weight_decay, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                        milestones=[ 33], gamma=0.1)


print('-'*30)
print('learing rate {}'.format(lr))
print('trainable V {}'.format(trainable_w2v))
print('lambda_ {}'.format(lambda_))
print('optimized seen only')
print('optimizer: RMSProp with momentum = {} and weight_decay = {}'.format(momentum,weight_decay))
print('-'*30)


number = 'CUBseen-0.5unseen+1'
#visual_path
visual_path = pwd+'/visual/' + '{}{}bz={}lr=0.0001epoch=3000/'.format(number,mode,batch_size)

#store_path
save_dir = pwd+'/result/{}{}bz={},lr=0.0001,epoch=3000.txt'.format(number,mode,batch_size)
path = pwd +'/weight/{}{}bz={},lr=0.0001epoch=3000/'.format(number,mode,batch_size)

if not os.path.exists(path):
      os.makedirs(path)
logger = Logger(visual_path)
logging = init_log(save_dir)
_print = logging.info
# In[8]:
epoch = 0
best_performance = [0,0,0,0]
running_loss = 0
running_loss_CE = 0
running_loss_cal = 0
running_loss_mse = 0


for i in range(0,niters):

    model.train()
    optimizer.zero_grad()
 
    batch_label, batch_feature, batch_att = dataloader.next_batch(batch_size) #[50],[50, 2048, 7, 7],[50, 312]


    out_package = model(batch_feature)

    in_package = out_package
    in_package['batch_label'] = batch_label
    in_package['batch_att'] = batch_att
    out_package = model.compute_loss(in_package)
    loss,loss_CE,loss_cal,loss_mse = out_package['loss'],out_package['loss_CE'],out_package['loss_cal'],out_package['loss_mse']

    loss.backward()
    optimizer.step()

    running_loss += loss.data.cpu()* batch_size
    running_loss_CE += loss_CE.data.cpu()* batch_size
    running_loss_cal += loss_cal.data.cpu()* batch_size
    running_loss_mse += loss_mse.data.cpu()* batch_size


    if i%report_interval==0 and i != 0: 
        if mode != 'normal':
            scheduler.step()
        epoch += 1
        print('-'*30)
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        ##########train loss
        running_loss = running_loss/train_numbers
        running_loss_CE = running_loss_CE/train_numbers
        running_loss_cal = running_loss_cal/train_numbers
        running_loss_mse = running_loss_mse/train_numbers


        logger.scalar_summary('train_loss', running_loss.item(), epoch)
        logger.scalar_summary('train_loss_CE', running_loss_CE.item(), epoch)
        logger.scalar_summary('train_loss_cal', running_loss_cal.item(), epoch)
        logger.scalar_summary('train_loss_mse', running_loss_mse.item(), epoch)
        


        ##############test

        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataloader,model, bias_seen=-bias,bias_unseen=bias)

        loss_seen, loss_CE_seen, loss_cal_seen, loss_mse_seen,loss_unseen, loss_CE_unseen, loss_cal_unseen, loss_mse_unseen = eval_zs_gzsl_loss(dataloader,model,bias_seen=-bias,bias_unseen=bias)

        torch.save(model.state_dict(), path+'epoch-%d.pth'%(epoch))

        logger.scalar_summary('acc_seen', acc_seen.item(), epoch)
        logger.scalar_summary('acc_novel', acc_novel.item(), epoch)
        logger.scalar_summary('H', H.item(), epoch)
        logger.scalar_summary('acc_zs', acc_zs.item(), epoch)



        logger.scalar_summary('test_seen_loss', loss_seen.item(), epoch)
        logger.scalar_summary('test_seen_loss_CE', loss_CE_seen.item(), epoch)
        logger.scalar_summary('test_seen_loss_cal', loss_cal_seen.item(), epoch)
        logger.scalar_summary('test_seen_loss_mse', loss_mse_seen.item(), epoch)

        logger.scalar_summary('test_unseen_loss', loss_unseen.item(), epoch)
        logger.scalar_summary('test_unseen_loss_CE', loss_CE_unseen.item(), epoch)
        logger.scalar_summary('test_unseen_loss_cal', loss_cal_unseen.item(), epoch)
        logger.scalar_summary('test_unseen_loss_mse', loss_mse_unseen.item(), epoch)


        if H > best_performance[2]:
            best_performance = [acc_seen, acc_novel, H, acc_zs]

        _print('--' * 60) 
        _print('epoch:{} - train_loss: {:.4f}        train_loss_CE: {:.4f}        train_loss_cal: {:.4f}      train_loss_mse: {:.4f}'.format(epoch,running_loss,running_loss_CE,running_loss_cal,running_loss_mse))
        _print('epoch:{} - test_seen_loss: {:.4f}    test_seen_loss_CE: {:.4f}    test_seen_loss_cal: {:.4f}       test_seen_loss_mse: {:.4f}'.format(epoch,loss_seen,loss_CE_seen,loss_cal_seen,loss_mse_seen))
        _print('epoch:{} - test_unseen_loss: {:.4f}  test_unseen_loss_CE: {:.4f}  test_unseen_loss_cal: {:.4f}     test_unseen_loss_mse: {:.4f} '.format(epoch,loss_unseen,loss_CE_unseen,loss_cal_unseen,loss_mse_unseen))
        _print('epoch:{} - acc_seen: {:.4f}          acc_novel: {:.4f}            H: {:.4f}                       acc_zs: {:.4f}'.format(epoch,acc_seen,acc_novel,H,acc_zs))


        stats_package1 = {'epoch':epoch, 'train_loss':running_loss.item(), 'train_loss_CE':running_loss_CE.item(),'train_loss_cal': running_loss_cal.item(),'train_loss_mse': running_loss_mse.item()}
        stats_package2 = {'epoch':epoch,'test_seen_loss':loss_seen.item(),'test_seen_loss_CE':loss_CE_seen.item(),'test_seen_loss_cal': loss_cal_seen.item(),'test_seen_loss_mse': loss_mse_seen.item()}
        stats_package3 = {'epoch':epoch,'test_unseen_loss':loss_unseen.item(),'test_unseen_loss_CE':loss_CE_unseen.item(),
                        'test_unseen_loss_cal':loss_cal_unseen.item(),'test_unseen_loss_mse': loss_mse_unseen.item()}
        stats_package4 = {'epoch':epoch,'acc_seen':acc_seen.item(), 'acc_novel':acc_novel.item(), 'H':H.item(), 'acc_zs':acc_zs.item()}


        running_loss = 0
        running_loss_CE = 0
        running_loss_cal = 0
        running_loss_mse = 0

        print('--' * 60)
        print(stats_package1)
        print(stats_package2)
        print(stats_package3)
        print(stats_package4)
        

_print('the best result:' ) 
_print('- acc_seen: {:.4f}          acc_novel: {:.4f}            H: {:.4f}                       acc_zs: {:.4f}'.format(best_performance[0],best_performance[1],best_performance[2],best_performance[3]))

print('the best result:')
stats_package5 = {'acc_seen':best_performance[0], 'acc_novel':best_performance[1], 'H':best_performance[2], 'acc_zs':best_performance[3]}
print(stats_package5)
