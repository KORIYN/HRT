import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

#### Capsule Layer ####
class CapsuleFC(nn.Module):
    r"""Applies as a capsule fully-connected layer.
    TBD
    """
    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, matrix_pose, dp,init_w2v_att_fa):
        super(CapsuleFC, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules 
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.matrix_pose = matrix_pose

        if matrix_pose:
            self.sqrt_d = int(np.sqrt(self.in_d_capsules))
            self.weight_init_const = np.sqrt(out_n_capsules/(self.sqrt_d*in_n_capsules)) 
            self.w = nn.Parameter(self.weight_init_const* \
                                          torch.randn(in_n_capsules, self.sqrt_d, self.sqrt_d, out_n_capsules))

        else:
            self.weight_init_const = np.sqrt(out_n_capsules/(in_d_capsules*in_n_capsules)) 
            self.w = nn.Parameter(self.weight_init_const* \
                                          torch.randn(in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules))

        self.dropout_rate = dp
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)
        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1. / (out_d_capsules ** 0.5) 
        #attribute semantic vectors
        self.init_w2v_att_fa = init_w2v_att_fa
        self.init_w2v_att_fa = F.normalize(init_w2v_att_fa)
        self.V = nn.Parameter(self.init_w2v_att_fa.clone(),requires_grad = True)  


    def compute_V(self):

        V_n = self.V.reshape(self.V.shape[0],self.sqrt_d,self.sqrt_d)
        return V_n


    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, matrix_pose={}, \
            weight_init_const={}, dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, self.matrix_pose,
            self.weight_init_const, self.dropout_rate
        )   


    def forward(self, input, num_iter, next_capsule_value=None):


        # b: batch size
        # n: num of capsules in current layer
        # a: dim of capsules in current layer
        # m: num of capsules in next layer
        # d: dim of capsules in next layer
        

        if self.matrix_pose:
            w = self.w # nxdm 
            _input = input.view(input.shape[0], input.shape[1], self.sqrt_d, self.sqrt_d).cuda() # bnax
        else:
            w = self.w

        
        V_16 = self.compute_V()

        if next_capsule_value is None:
            
            
            next_capsule_value = V_16

            V = torch.einsum(' nxdm, bnax->bnadm', w, _input)
            V_shape = V.shape

            V = V.reshape(V_shape[0],V_shape[1],V_shape[2] * V_shape[3],V_shape[4])
            V = F.softmax(V, dim=2)
            V = V.reshape(V_shape[0],V_shape[1],self.sqrt_d,self.sqrt_d,V_shape[4])

            query_key = torch.einsum('bnadm, mad ->bnm', V, next_capsule_value )
            agreement = query_key
            query_key = F.softmax(query_key, dim=2) # routing probabilities
            

            if self.matrix_pose:
                next_capsule_value = torch.einsum('bnm, bnadm->bmad', query_key, V) 

            else:
                next_capsule_value = torch.einsum('bnm, bna, namd->bmd', query_key, input, w) 

        else:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0], 
                                       next_capsule_value.shape[1], self.sqrt_d, self.sqrt_d) 
                V = torch.einsum(' nxdm, bnax->bnadm', w, _input) 
                V_shape = V.shape
                V = V.reshape(V_shape[0],V_shape[1],V_shape[2] * V_shape[3],V_shape[4])
                V = F.softmax(V, dim=2)
                V = V.reshape(V_shape[0],V_shape[1],self.sqrt_d,self.sqrt_d,V_shape[4])

                _query_key = torch.einsum('bnadm, bmad ->bnm', V,next_capsule_value )

            else:
                _query_key = torch.einsum('bna, namd, bmd->bnm', input, w, next_capsule_value)
            _query_key.mul_(self.scale) 
            agreement = _query_key
            query_key = F.softmax(_query_key, dim=2)
            query_key = query_key / (torch.sum(query_key, dim=2, keepdim=True) + 1e-10)

            if self.matrix_pose:
                next_capsule_value = torch.einsum('bnm, bnadm->bmad', query_key, V)
            else:
                next_capsule_value = torch.einsum('bnm, bna, namd->bmd', query_key, input, w)

        next_capsule_value = self.drop(next_capsule_value)

        if not next_capsule_value.shape[-1] == 1:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0], 
                                       next_capsule_value.shape[1], self.out_d_capsules) 
                next_capsule_value = self.nonlinear_act(next_capsule_value) #LayerNorm
            else:
                next_capsule_value = self.nonlinear_act(next_capsule_value)

        return next_capsule_value , agreement 