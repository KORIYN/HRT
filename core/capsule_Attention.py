from core import layers
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from core import capsule_EM


# Capsule model
class CapsModel_Attention(nn.Module):
    def __init__(self,
                init_w2v_att_fa,
                in_n_capsules , out_n_capsules,
                dp = 0.0,
                num_routing = 2
                ):

        super(CapsModel_Attention, self).__init__()
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        ## FC Capsule Layers        

        self.capsule_layers = nn.ModuleList([])

        self.capsule_layers.append(
            layers.CapsuleFC(in_n_capsules = in_n_capsules, 
                    in_d_capsules = 16, 
                    out_n_capsules = out_n_capsules, 
                    out_d_capsules = 16, 
                    matrix_pose = True,
                    dp = dp,
                    init_w2v_att_fa = init_w2v_att_fa
                    )
        )
        self.nonlinear_act = nn.LayerNorm(16)
        self.capusle_EM = capsule_EM.capsules_EM()
    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass

        capsule_patch = self.capusle_EM(x)  
        shape = capsule_patch.shape
        capsule_patch  = capsule_patch.reshape(shape[0],shape[1]*shape[2],shape[3]) 

        init_capsule_value = self.nonlinear_act(capsule_patch)

        ## Main Capsule Layers 
        # concurrent routing

        # first iteration
        # perform initilialization for the capsule values as single forward passing
        capsule_values, _val = [init_capsule_value], init_capsule_value
        query_key =[]
        for i in range(len(self.capsule_layers)): #i = 0, 1, 2
            _val,key = self.capsule_layers[i].forward(_val, 0)
            capsule_values.append(_val) # get the capsule value for next layer
            query_key.append(key)

        # second to t iterations
        # perform the routing between capsule layers
        for n in range(self.num_routing-1):
            _capsule_values = [init_capsule_value]
            _query_key =[]
            for i in range(len(self.capsule_layers)):
                _val,key = self.capsule_layers[i].forward(capsule_values[i], n, capsule_values[i+1])
                _capsule_values.append(_val)
                _query_key.append(key)
            capsule_values = _capsule_values
            query_key =_query_key

        return capsule_values[-1] , query_key[-1]
