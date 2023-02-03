# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:48:58 2023

@author: Xiaotong
"""

import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import sparse
import random
import warnings
from linearbandits import LinearBandits
from subgoss import subgoss

# Fixing randomness!!!
random.seed(1)
np.random.seed(2)

# Removing warnings
warnings.filterwarnings("ignore")

num_j = 9
K = 10
d = 20
m = 2
N = 5
num_sim = 3
sA = 1
sigma = 0.01
Ad_matrix = np.ones((N,N)) - np.eye(N)
b = 2
sn = int(d/m)

# Defining model parameter (theta_star) #todo
theta_star = np.zeros(d)
for i in range(0,m):
    theta_star[d-i-1] = random.uniform(0.3, 1)
    
print(theta_star)

# Covariance Matrix Initialization #todo
sigma_sq = 1.
rho_sq = 0.7
M0 = (sigma_sq - rho_sq) * np.eye(K) + rho_sq * np.ones((K, K))  # Covariance Matrix

# Constructing the environment
lnr_bandits = LinearBandits(d, M0, theta_star, sigma, sA)

subgoss = subgoss(num_j,num_sim,Ad_matrix,b,N,sn,m,d,K,lnr_bandits)
subgoss_mean, subgoss_std = subgoss.run()
T = subgoss.T

x = [t for t in range(T)]

plt.plot(subgoss_mean, label=f'SubGoss, N={N}', color='#56b4e9')
# u = [subgoss_mean[i] + subgoss_std[i] for i in range(T)]
# l = [subgoss_mean[i] - subgoss_std[i] for i in range(T)]
# plt.fill_between(x, u, l, color='#9ad2f2', alpha=0.5)
plt.xlabel('Time Horizon (T)')
plt.ylabel(f'Averaged Cumulative-Regret per agent')
plt.title(f's_0/d: {m}/{d} and K: {K}')
plt.legend()
plt.show()

