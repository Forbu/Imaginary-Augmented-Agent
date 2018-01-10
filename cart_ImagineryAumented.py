#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:39:17 2017

@author: adrienbufort
"""

"""
MAIN program 
for imaginary agent
"""
from cart_ImagineryNetwork import *
import numpy as np
import math
# test part of the neural network
from torch.distributions import Categorical
    
def init_weights(m):
    if type(m) == nn.Linear:
        # get size of init
        m.weight.data.normal_(0.0, math.sqrt(2. / 100000))
    if type(m) == nn.LSTM:
        m.weight_hh_l0.data.normal_(0.0, math.sqrt(2. / 100000))

# Now the training scession
class cart_pole_solver:
    def __init__(self,size_rollout = 5,n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=0.99, alpha=0.01, batch_size=1, monitor=False, quiet=False):
        self.size_rollout = size_rollout
        
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        
        self.memory = [[] for p in range(batch_size)]
        
        self.alpha = alpha
        
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
        self.size_rollout = size_rollout
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        
        # NEURAL NETWORK MODEL
        self.env_model_ = env_network()
        self.policy_model = policy_network_pre_trained()
        self.imagine_model = Imagine_Core_network(self.policy_model,self.env_model_)
        self.Imagine_future_network = imagine_future_network(self.imagine_model,self.size_rollout)
        
        self.Final_synthesis_network = final_synthesis_network()
        self.Rollout_encoder = rollout_encoder(self.size_rollout)
        self.Model_free_network = model_free_network()
        
        self.all_network = aggregator_network(self.Rollout_encoder, self.Model_free_network, self.Imagine_future_network, self.Final_synthesis_network,size_rollout)
        
        # Pre trained model
        file = "modeleNAC2017-12-18--14:26:11.pt"
        self.policy_model.load_state_dict(torch.load(file))
        
        # Pre trained model
        file = "modelenv2017-12-18--14:06:59.pt"
        self.env_model_.load_state_dict(torch.load(file))
        
        self.all_network.zero_grad()
        self.optimizerRollout = optim.Adam(self.Rollout_encoder.parameters(), lr=1e-2)
        self.optimizerFree = optim.Adam(self.Model_free_network.parameters(), lr=1e-2)
        self.optimizerAgreg = optim.Adam(self.Final_synthesis_network.parameters(), lr=1e-2)
        
        self.Rollout_encoder.apply(init_weights)
        self.Model_free_network.apply(init_weights)
        self.Final_synthesis_network.apply(init_weights)
        self.saved_log_probs = []
            
    def choose_action(self,state):

        action = self.all_network(state)
        function_sampling = Categorical(action)
        action_done = function_sampling.sample()
        #print(action)
        #print(action_done)
        return action_done.data,function_sampling.log_prob(action_done)
    
    def remember(self,log_action,rewards,batch_index,done):
        self.memory[batch_index].append((log_action,rewards,done))
    
    def preprocess_state(self, state):
        return Variable(torch.from_numpy(state).view((1,4)).float())
    
    def run(self):
        print("Beginning trainning")
                
        """
        Main program to run the Imaginary augmented agent
        """
        scores = deque(maxlen=100)
        self.env.render()
        index_batch = 0
        self.all_network.zero_grad()
        for e in range(self.n_episodes):
            
            
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            
            while not done:
                action, function_log_action = self.choose_action(state)
                
                next_state, reward, done, _ = self.env.step(action[0])
                next_state = self.preprocess_state(next_state)
                
                
                self.remember(function_log_action, reward, index_batch, done)
                state = next_state
                
                i += 1
            #exit(-1)
            index_batch = (index_batch + 1) % self.batch_size
            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                return e - 100
            if (e % self.batch_size == 0 and not self.quiet) and e != 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
                self.gradiant_desc()
                
        
        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e
    
    def get_reward_learning(self,rewards):
        R = 0
        rewards_weight = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            rewards_weight.insert(0, R)
        return np.array(rewards_weight)

    def gradiant_desc(self):
         
        delta_J = Variable(torch.FloatTensor(1).fill_(0.0))
        
        self.all_network.zero_grad()
        for i in range(self.batch_size):
            
            log_prob = list(map(lambda x:x[0],self.memory[i]))
            #print(log_prob)
            rewards = np.array(list(map(lambda x:x[1],self.memory[i])))
            rewards = self.get_reward_learning(rewards)
            #print(rewards)
            for t in range(len(self.memory[i])):

                rewards_value = (rewards[t] - rewards.mean())/rewards.std()
                
                delta_J += - log_prob[t] * rewards_value
        #exit(-1)
        delta_J.backward()
        self.optimizerRollout.step()
        self.optimizerFree.step()
        self.optimizerAgreg.step()

        
        self.memory = [[] for p in range(self.batch_size)]
       
        #print("yoloooo")

if __name__ == '__main__':
    agent = cart_pole_solver()
    agent.run()
  
    
    
    
