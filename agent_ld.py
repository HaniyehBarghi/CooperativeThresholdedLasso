# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:06:07 2023

@author: Xiaotong
"""

import numpy as np
import random 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import pdb

class agent_ld:
    def __init__(self,no,neighbor,sn,K,m,d,N,T):
        self.no = no                  # Number of the agent
        self.neighbor = neighbor      # Neighbors who can receive information
        self.sn = sn                  # number of subspaces
        self.K = K                    # number of arms
        self.m = m                    # dimension of each subspaces (sparsity)
        self.d = d                    # real dimension
        self.N = N                    # number of agents
        self.T = T
        s_set = range(int(self.no*self.sn/self.N), int((self.no+1)*self.sn/self.N))
        self.s_set = set(s_set)       # S set
        self.o_index = int(self.no*self.sn/self.N)
        
        self.observed_rewards = np.array([], dtype=np.float32)  # Chosen context-vectors up to time t
        self.chosen_arms = np.empty((0, d), dtype=np.float32)  # Observed rewards up to time t
        
        self.theta_hat = np.zeros(d, dtype=np.float32)
        self.cumulative_regret = np.array([], dtype=np.float32)
        self.cumulative_regret = np.append(self.cumulative_regret, 0.0)

        self.tilde_theta = {i:np.zeros(d) for i in range(self.sn)}
        self.tilde_observed_rewards = {i:np.array([]) for i in range(self.sn)}
        self.tilde_chosen_arms = {i:np.empty((0,self.d)) for i in range(self.sn)}
        
    def lasso(self, lambda_t):
        # print(f'check the size of chosen arms at each dimension reduction round: {np.shape(self.chosen_arms)}')
        lasso = Lasso(alpha=lambda_t)
        # print(np.shape(self.chosen_arms))
        # print(np.shape(self.observed_rewards))
        lasso.fit(self.chosen_arms, self.observed_rewards)

        return lasso.coef_
    
    def explore(self,lnr_bandit,sn_idx):
        context_vectors = lnr_bandit.generate_explore_context(sn_idx,self.m)
        selected_arm_idx = np.random.randint(0,self.K)
        selected_arm = context_vectors[int(selected_arm_idx)]
        self.chosen_arms = np.vstack((self.chosen_arms, selected_arm))
        reward, regret = lnr_bandit.pull(selected_arm)
        self.observed_rewards = np.append(self.observed_rewards, reward)
        self.cumulative_regret = np.append(self.cumulative_regret, self.cumulative_regret[-1] + regret)
        
        self.tilde_chosen_arms[sn_idx] = np.vstack((self.tilde_chosen_arms[sn_idx], selected_arm))
        self.tilde_observed_rewards[sn_idx] = np.append(self.tilde_observed_rewards[sn_idx], reward)   
    
    def update_o(self):
        norm_tilde_theta = {i: 0 for i in self.s_set}
        for i in self.s_set:
            num_i = self.tilde_chosen_arms[i].shape[0]
            if  num_i > 0.1:
                lasso1 = Lasso(alpha = 0)
                lasso1.fit(self.tilde_chosen_arms[i],self.tilde_observed_rewards[i])
                self.tilde_theta[i] = lasso1.coef_
                norm_tilde_theta[i] = np.linalg.norm(self.tilde_theta[i]) 
        self.o_index = max(norm_tilde_theta, key = norm_tilde_theta.get)
    
    def get_o(self):
        return self.o_index 

    def project_ucb(self,lnr_bandit):
        U = np.zeros((self.d, self.m))
        j = 0
        #print('subspace',self.no,self.o_index)
        for i in range(self.o_index*self.m,(self.o_index+1)*self.m):
            U[i][j] = 1
            j += 1
        P = np.matmul(U,U.T)             
                    
        context_vectors = lnr_bandit.generate_context()
        #pdb.set_trace()
        k = self.o_index
        Vkt = np.matmul(np.matmul(P,(np.matmul(self.chosen_arms[k],self.chosen_arms[k].T)+np.eye(self.d))),P)
                
        ridge = Ridge(alpha = 1)
        ridge.fit(self.tilde_chosen_arms[k],self.tilde_observed_rewards[k])
        self.theta_hat = ridge.coef_
        selected_arm_idx = self.select_arm(P,context_vectors,Vkt)
        selected_arm = context_vectors[selected_arm_idx]
        self.chosen_arms = np.vstack((self.chosen_arms, selected_arm))
        reward, regret = lnr_bandit.pull(selected_arm)
        self.observed_rewards = np.append(self.observed_rewards, reward)
        self.cumulative_regret = np.append(self.cumulative_regret, self.cumulative_regret[-1] + regret)
        self.tilde_chosen_arms[k] = np.vstack((self.tilde_chosen_arms[k], selected_arm))
        self.tilde_observed_rewards[k] = np.append(self.tilde_observed_rewards[k], reward)   

    def select_arm(self,P,context_vectors,Vkt):
        est_rewards = np.zeros(context_vectors.shape[0])
        k = self.o_index
        nk = self.tilde_chosen_arms[k].shape[0] 
        beta = 1 + np.sqrt(2*np.log(self.T)+self.m*np.log(1+nk/self.m))
        Vkt_inv = np.linalg.pinv(Vkt)
        Pcontext = np.matmul(context_vectors,P)
        #print(Pcontext.shape[0]-context_vectors.shape[0])
        for i in range(0,context_vectors.shape[0]):
            est_rewards[i] = np.dot(context_vectors[i],self.theta_hat) \
                    +0.1*beta*np.sqrt(np.dot(np.dot(Pcontext[i],Vkt_inv),Pcontext[i]))
        return np.argmax(est_rewards)
    
    def update_s_set(self,o_ag):
        s_hat_set = range(int(self.no*self.sn/self.N)+1, int((self.no+1)*self.sn/self.N)+1)
        s_hat_set = set(s_hat_set) 
        if o_ag not in self.s_set:
            if len(self.s_set) < (self.sn/self.N +2):
                self.s_set.add(o_ag)
            elif abs(len(self.s_set)-(self.sn/self.N +2)) < 0.01:
                b_set = self.s_set - s_hat_set
                norm_tilde_theta = {i: 0 for i in b_set}
                for j in b_set:
                    norm_tilde_theta[j] = np.linalg.norm(self.tilde_theta[j])
                b_index = max(norm_tilde_theta, key = norm_tilde_theta.get)
                self.s_set.add(b_index)
                self.s_set.union(s_hat_set)
                    
        
        

        
        
        

