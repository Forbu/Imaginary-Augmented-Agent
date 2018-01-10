#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:36:36 2017

@author: adrienbufort
"""

"""
Imaginary augmented agent
"""
import numpy as np

import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

from collections import deque

from time import gmtime, strftime

class policy_network_pre_trained(nn.Module):
    """
    Fix policy network 
    INPUT :
        - observation
    OUTPUT :
        - action (hot encoder)
    """
    def __init__(self):
        super(policy_network_pre_trained, self).__init__()
            
        self.lin1 = nn.Linear(4, 24)
        
        self.lin2 = nn.Linear(24, 48)
        
        self.lin3 = nn.Linear(48,2)
        
    def forward(self, data):
        x = F.relu(self.lin1(data))
        x = F.relu(self.lin2(x))
        x = F.softmax(self.lin3(x),dim=-1)
        return x
        
class env_network(nn.Module):
    """
    Network with freeze state
    to modelize the environment
    INPUT :
        - action (hot encoder) dimension 2
        - observation dimension 4
    OUTPUT :
        - observation
        - reward
    """
    def __init__(self):
        super(env_network, self).__init__()
        
        self.lin1 = nn.Linear(6, 24)
        
        self.lin2 = nn.Linear(24, 48)
        
        self.lin3 = nn.Linear(48,5)
        
    def forward(self, data):
        x = F.relu(self.lin1(data))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

class Imagine_Core_network(nn.Module):
    """
    Network 
    INPUT : 
        - Observation t
    OUTPUT :
        - Observation t + 1
        - reward t + 1
    """
    def __init__(self,policy_network,env_network):
        super(Imagine_Core_network, self).__init__()
        
        self.policy_network = policy_network
        self.env_network = env_network
        
    def forward(self, observations):
        
        policy = self.policy_network(observations)
        
        # concat 
        agggreg = torch.cat((observations,policy),1)
        return self.env_network(agggreg)
    
class imagine_future_network(nn.Module):
    """
    Return sequence of observation rewards
    """
    def __init__(self,imagine_core_network,size_rollout):
        super(imagine_future_network, self).__init__()
        self.imagine_core_network = imagine_core_network
        self.size_rollout = size_rollout
    
    def forward(self, observations):
        data = []
        for i in range(self.size_rollout):
            observations_reward = self.imagine_core_network(observations)
            observations = observations_reward[:,:4]
            data.append(observations_reward)
            
        # concat phase
        #print("data : ",data)
        seq_imagined = torch.cat(data,0)
        # proper format for rollout_encoder to come
        seq_imagined = seq_imagined.view((self.size_rollout,-1,5))
        return seq_imagined
        
class rollout_encoder(nn.Module):
    """
    Network : LSTM rollout encoder 
    INPUT : 
        - Observation t
    OUTPUT
        - sequence of reward / observation
    """
    
    def __init__(self,N_rollout):
        super(rollout_encoder, self).__init__()
        """
        What we need is a N_rollout sequence of LSTM
        MANY TO ONE config (stateless + return sequence = false)
        
        """
        input_size = 5
        output_size1 = 24
        output_size2 = 5
        
        self.lstm1 = nn.LSTM(input_size, output_size1, 1)
      
        self.lstm2 = nn.LSTM(output_size1, output_size2, 1)
        
        
    def forward(self, data):

        x, hidden = self.lstm1(data)
        x = F.relu(x)

        x,hidden = self.lstm2(x)
        x = F.relu(x)
        
        return x[-1]
    
class model_free_network(nn.Module):
    """
    Network 
    INPUT :
        - Observation
    OUTPUT :
        - internal state
    """
    def __init__(self):
        super(model_free_network, self).__init__()
        self.lin1 = nn.Linear(4, 24)
        self.drop1 = nn.Dropout(p=0.1)
        self.lin2 = nn.Linear(24, 48)
        self.drop2 = nn.Dropout(p=0.1)
        
    def forward(self, data):
        x = F.relu(self.drop1(self.lin1(data)))
        x = F.relu(self.drop2(self.lin2(x)))
        return x

class final_synthesis_network(nn.Module):
    """
    Network
    
    """
    def __init__(self):
        super(final_synthesis_network, self).__init__()
        self.lin4 = nn.Linear(53, 2)
        self.drop4 = nn.Dropout(p=0.1)
        
    def forward(self, input_s):
        
        new_vector = F.softmax(self.drop4(self.lin4(input_s)))
        
        return new_vector
        
class aggregator_network(nn.Module):
    """
    Network
    
    """
    def __init__(self,Rollout_encoder,Model_free_network,Imagine_future_network, Final_synthesis_network,size_rollout):
        super(aggregator_network, self).__init__()
        self.Rollout_encoder = Rollout_encoder
        self.Model_free_network = Model_free_network
        self.Imagine_future_network = Imagine_future_network
        self.Final_synthesis_network = Final_synthesis_network
        
        self.size_rollout = size_rollout
        
    def forward(self, observations):
        #print("imagine future")
        #print(observations)
        imagined_future = self.Imagine_future_network(observations)
        #imagined_future = imagined_future.detach()
        
        #print("rollout")
        #print(imagined_future)
        info_future = self.Rollout_encoder(imagined_future.detach())
        #print("model free")
        info_model_free = self.Model_free_network(observations)
        #print(info_model_free)
        #print(info_future)
        new_vector = torch.cat((info_future,info_model_free),1)
        #print("final aggregation")
        result = self.Final_synthesis_network(new_vector)
        
        return result
    



