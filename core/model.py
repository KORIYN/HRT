# -*- coding: utf-8 -*-  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  
from core import capsule_Attention

#%%  
import pdb  
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torchvision.utils import make_grid
#%%  
  
class HRT(nn.Module):  
    #####  
    # einstein sum notation  
    # b: Batch size \ f: dim feature \ v: dim w2v \ r: number of region \ k: number of classes  
    # i: number of attribute \ h : hidden attention dim  
    #####  
    def __init__(self,data,dim_f,dim_v,  
                 init_w2v_att,init_w2v_att_fa,att,normalize_att,  
                 seenclass,unseenclass, 
                 lambda_,  
                 trainable_w2v = False, normalize_V = False, normalize_F = False, is_conservative = False,
                 prob_prune=0.0,desired_mass = -1,uniform_att_1 = False,uniform_att_2 = False, is_conv = False,
                 is_bias = False,bias = 1,non_linear_act=False,
                 loss_type = 'CE',non_linear_emb = False,
                 is_sigmoid = False):  
        super(HRT, self).__init__()  
        self.dim_f = dim_f  
        self.dim_v = dim_v 
        self.dim_att = att.shape[1]  
        self.nclass = att.shape[0] 
        self.hidden = self.dim_att//2 
        self.init_w2v_att = init_w2v_att 
        self.init_w2v_att_fa = init_w2v_att_fa
        self.non_linear_act = non_linear_act
        self.loss_type = loss_type
        if is_conv:
            r_dim = dim_f//2
            self.conv1 = nn.Conv2d(dim_f, r_dim, 2) #[2x2] kernel with same input and output dims
            print('***Reduce dim {} -> {}***'.format(self.dim_f,r_dim))
            self.dim_f = r_dim
            self.conv1_bn = nn.BatchNorm2d(self.dim_f)
            
            
        if init_w2v_att is None:  
            self.V = nn.Parameter(nn.init.normal_(torch.empty(self.dim_att,self.dim_v)).float(),requires_grad = True)  
        else:
            self.init_w2v_att = F.normalize(init_w2v_att)
            self.V = nn.Parameter(self.init_w2v_att.clone().float(),requires_grad = trainable_w2v)  

        self.att = nn.Parameter(F.normalize(att).float(),requires_grad = False) 
  
        self.W_1 = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v,self.dim_f).float()),requires_grad = True)

        self.W_3 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v,self.dim_f).float()),requires_grad = True)
        self.P = torch.mm(self.att,torch.transpose(self.att,1,0)) 
        assert self.P.size(1)==self.P.size(0) and self.P.size(0)==self.nclass  
        self.weight_ce = nn.Parameter(torch.eye(self.nclass).float(),requires_grad = False)
        
        self.data = data  
        if self.data == 'CUB':
            in_n_capsules = 196
        else:
            in_n_capsules = 49
        out_n_capsules = self.att.shape[1]

        #capsule
        self.CapsModel = capsule_Attention.CapsModel_Attention(in_n_capsules = in_n_capsules, out_n_capsules = out_n_capsules, init_w2v_att_fa = self.init_w2v_att_fa)

        self.normalize_V = normalize_V  
        self.normalize_F = normalize_F   
        self.is_conservative = is_conservative  
        self.is_conv = is_conv
        self.is_bias = is_bias
        
        self.seenclass = seenclass  
        self.unseenclass = unseenclass  
        self.normalize_att = normalize_att  

        if is_bias:
            self.bias = nn.Parameter(torch.tensor(bias),requires_grad = False)
            mask_bias = np.ones((1,self.nclass))
            if self.data == 'AWA2':
                mask_bias[:,self.seenclass.cpu().numpy()] *= -0.8
            else:
                mask_bias[:,self.seenclass.cpu().numpy()] *= -0.5
            mask_bias[:,self.unseenclass.cpu().numpy()] *= 1
            
            self.mask_bias = nn.Parameter(torch.tensor(mask_bias).float(),requires_grad = False)


        if desired_mass == -1:  
            self.desired_mass = self.unseenclass.size(0)/self.nclass
        else:  
            self.desired_mass = desired_mass#nn.Parameter(torch.tensor(desired_mass),requires_grad = False)#nn.Parameter(torch.tensor(self.unseenclass.size(0)/self.nclass),requires_grad = False)#  
        self.prob_prune = nn.Parameter(torch.tensor(prob_prune).float(),requires_grad = False) 
        self.lambda_ = lambda_
        self.loss_att_func = nn.BCEWithLogitsLoss()
        self.log_softmax_func = nn.LogSoftmax(dim=1)  
        self.uniform_att_1 = uniform_att_1
        self.uniform_att_2 = uniform_att_2
        
        self.non_linear_emb = non_linear_emb
        
        
        print('-'*30)  
        print('Configuration')  
        
        print('loss_type {}'.format(loss_type))
        
        if self.is_conv:
            print('Learn CONV layer correct')
        
        if self.normalize_V:  
            print('normalize V')  
        else:  
            print('no constraint V')  
              
        if self.normalize_F:  
            print('normalize F')  
        else:  
            print('no constraint F')  
              
        if self.is_conservative:  
            print('training to exclude unseen class [seen upperbound]')  
        if init_w2v_att is None:  
            print('Learning word2vec from scratch with dim {}'.format(self.V.size()))  
        else:  
            print('Init word2vec')  
        
        if self.non_linear_act:
            print('Non-linear relu model')
        else:
            print('Linear model')
        
        print('loss_att {}'.format(self.loss_att_func))  
        print('Bilinear attention module')  
        print('*'*30)  
        print('Measure w2v deviation')
        if self.uniform_att_1:
            print('WARNING: UNIFORM ATTENTION LEVEL 1')
        if self.uniform_att_2:
            print('WARNING: UNIFORM ATTENTION LEVEL 2')
        print('Compute Pruning loss {}'.format(self.prob_prune))  
        if self.is_bias:
            print('Add one smoothing')
        print('Second layer attenion conditioned on image features')
        print('-'*30)  
        
        if self.non_linear_emb:
            print('non_linear embedding')
            self.emb_func = torch.nn.Sequential(
                                torch.nn.Linear(self.dim_att, self.dim_att//2),
                                torch.nn.ReLU(),
                                torch.nn.Linear(self.dim_att//2, 1),
                            )
        
        self.is_sigmoid = is_sigmoid
        if self.is_sigmoid:
            print("Sigmoid on attr score!!!")
        else:
            print("No sigmoid on attr score")

    
    def compute_loss_rank(self,in_package):  
        # this is pairwise ranking loss  
        batch_label = in_package['batch_label']  
        S_pp = in_package['S_pp']  
        
        batch_label_idx = torch.argmax(batch_label,dim = 1)
        
        s_c = torch.gather(S_pp,1,batch_label_idx.view(-1,1))  
        if self.is_conservative:  
            S_seen = S_pp  
        else:  
            S_seen = S_pp[:,self.seenclass]  
            assert S_seen.size(1) == len(self.seenclass)  
          
        margin = 1-(s_c-S_seen)  
        loss_rank = torch.max(margin,torch.zeros_like(margin))  
        loss_rank = torch.mean(loss_rank)  
        return loss_rank  
      
    def compute_loss_Self_Calibrate(self,in_package):  
        S_pp = in_package['S_pp']  
        Prob_all = F.softmax(S_pp,dim=-1)
        Prob_unseen = Prob_all[:,self.unseenclass]  
        assert Prob_unseen.size(1) == len(self.unseenclass)  
        mass_unseen = torch.sum(Prob_unseen,dim=1)  
        
        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp  
      
    def compute_V(self):
        if self.normalize_V:  
            V_n = F.normalize(self.V)
        else:  
            V_n = self.V  
        return V_n

    def compute_aug_cross_entropy(self,in_package):  
        batch_label = in_package['batch_label']  
        S_pp = in_package['S_pp'] 

        Labels = batch_label

        self.vec_bias = self.mask_bias*self.bias  
        
        if self.is_bias:
            S_pp = S_pp - self.vec_bias
        
        if not self.is_conservative:  
            S_pp = S_pp[:,self.seenclass]  
            Labels = Labels[:,self.seenclass]  
            assert S_pp.size(1) == len(self.seenclass)  
        
        Prob = self.log_softmax_func(S_pp)  

        loss = -torch.einsum('bk,bk->b',Prob,Labels)  
        loss = torch.mean(loss)  
        return loss  

    def compute_att_mse(self,in_package):
        batch_att = in_package['batch_att'] 
        Pred_att_final = in_package['Pred_att_final']

        loss=torch.nn.MSELoss(reduction='sum')
        loss_mse=loss(Pred_att_final,batch_att)/(batch_att.size(0)*batch_att.size(1))

        return loss_mse

    def compute_loss(self,in_package):

        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]  

        ## loss rank  
        if self.loss_type == 'CE':
            loss_CE = self.compute_aug_cross_entropy(in_package)
        elif self.loss_type == 'rank':
            loss_CE = self.compute_loss_rank(in_package)  
        else:
            raise Exception('Unknown loss type')
        
        ## loss self-calibration  
        loss_cal = self.compute_loss_Self_Calibrate(in_package)
        
        loss_mse = self.compute_att_mse(in_package)
        
        ## total loss  
        loss = loss_CE + self.lambda_*loss_cal + 0.033*loss_mse

          
        out_package = {'loss':loss,'loss_CE':loss_CE,
                       'loss_cal':loss_cal,'loss_mse':loss_mse}

        return out_package  
      
    def forward(self,Fs): 

        V_n = self.compute_V() 

        if self.is_conv:
            Fs = self.conv1(Fs)
            Fs = self.conv1_bn(Fs)
            Fs = F.relu(Fs)


        shape = Fs.shape 
        Fs = Fs.reshape(shape[0],shape[1],shape[2]*shape[3]) 

        if self.normalize_F and not self.is_conv:  
            Fs = F.normalize(Fs,dim = 1)  

        _ , agreement = self.CapsModel(Fs) 


        A = agreement.permute(0,2,1) 
        A = F.softmax(A,dim = -1)


        F_p = torch.einsum('bir,bfr->bif',A,Fs) # h 


        A_p = torch.einsum('iv,vf,bif->bi',V_n,self.W_3,F_p) 
        A_p = torch.sigmoid(A_p)  

        ##  
        Pred_att = torch.einsum('iv,vf,bif->bi',V_n,self.W_1,F_p) # e 
        
        Pred_att_final = torch.einsum('bi,bi->bi',Pred_att,A_p)
        
        S_pp = torch.einsum('ki,bi,bi->bik',self.att,A_p,Pred_att) 


            
        if self.non_linear_emb:
            S_pp = torch.transpose(S_pp,2,1)    #[bki] <== [bik]
            S_pp = self.emb_func(S_pp)          #[bk1] <== [bki]
            S_pp = S_pp[:,:,0]                  #[bk] <== [bk1]
        else:
            S_pp = torch.sum(S_pp,axis=1)        #[bk] <== [bik] 

        if self.is_bias:
            self.vec_bias = self.mask_bias*self.bias
            S_pp = S_pp + self.vec_bias

        package = {'S_pp':S_pp,'Pred_att_final':Pred_att_final}  
        return package  