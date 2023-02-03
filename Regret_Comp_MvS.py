################################################################################
####### This code is implemented to compare cumulative regret in FTL vs. #######
####### some single agent algorithms: SALasso, DRLasso, and THLasso.     #######
################################################################################

import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
import random
import math
import warnings
from linearbandits import LinearBandits
from federatedthlassobandit import FTL
from sparsityaglasso import SparsityAgLasso
from drl import DRL
from environment import Environment
from thlassobandit import ThLassoBandit
from linucb import linucb

# # Fixing randomness!!!
# random.seed(1)
# np.random.seed(2)

# Removing warnings
warnings.filterwarnings("ignore")

# Defining global variables
T = 200  # Time horizon
K = 5  # Number of arms
d = 200  # Dimension
s_0 = 5  # Sparsity parameter
s_2 = 0.3  # Minimum of absolute value of each non-zero dimension
sim_num = 3
sigma = 0.05
sA = 1

# Defining model parameter (theta_star)
theta_star = sparse.random(d, 1, density=s_0 / d).toarray()
for j in range(len(theta_star)):
    if theta_star[j] != 0:
        theta_star[j] = random.uniform(s_2, 1)
print(f'True support set: {np.nonzero(theta_star)[0]}.')

# Covariance Matrix Initialization
sigma_sq = 1.
rho_sq = 0.7
M0 = (sigma_sq - rho_sq) * np.eye(K) + rho_sq * np.ones((K, K))  # Covariance Matrix

# Defining Linear Bandits
lnr_bandits = LinearBandits(d, M0, theta_star, sigma, sA)

# FTL (15)
print('\t\t\t\t\t Federated Thresholded Lasso')
N = 10
c = 0.001  # Positive constant
s_A = 1  # Maximum absolute value of context-vector (component-wise)
lam0 = 4 * sigma * s_A * math.sqrt(c)
env = Environment(d, K, N, lnr_bandits, lam0)
FTL_15 = FTL(T, N, d, sim_num, env)
FTL_mean, FTL_std = FTL_15.run_algorithm()

# SA
print('\t\t\t\t\t Sparsity Agnostic Lasso')
alg_sal = SparsityAgLasso(T, d, sim_num, lnr_bandits)
SAL_mean, SAL_std = alg_sal.run_algorithm()

# THL
print('\t\t\t\t\t Thresholded Lasso')
c = 0.1  # Positive constant
s_A = 1  # Maximum absolute value of context-vector (component-wise)
lam0 = 4 * sigma * s_A * math.sqrt(c)
alg_thl = ThLassoBandit(T, d, sim_num, lnr_bandits, lam0)
THL_mean, THL_std = alg_thl.run_algorithm()

# DR
print('\t\t\t\t\t Doubly Robust Lasso')
alg_drl = DRL(T, K, M0, theta_star, sim_num)
DRL_mean, DRL_std = alg_drl.run_algorithm()

# # Linucb
print('\t\t\t\t\t LinUCB with Log Union')
linucb = linucb(env, sim_num, T, K, d, N)
linucb_mean, linucb_std = linucb.run()

# Plotting the result
x = [t for t in range(T)]

plt.plot(FTL_mean, label=f'FTLasso, N={N}', color='#56b4e9')
# u = [FTL_mean[i] + FTL_std[i] for i in range(T)]
# l = [FTL_mean[i] - FTL_std[i] for i in range(T)]
# plt.fill_between(x, u, l, color='#9ad2f2', alpha=0.2)

plt.plot(SAL_mean, label='SALasso', color='#009E73')
# u = [SAL_mean[i] + SAL_std[i] for i in range(T)]
# l = [SAL_mean[i] - SAL_std[i] for i in range(T)]
# plt.fill_between(x, u, l, color='#00ebab', alpha=0.2)

plt.plot(DRL_mean, label='DRLasso', color='#E69F00')
# u = [DRL_mean[i] + DRL_std[i] for i in range(T)]
# l = [DRL_mean[i] - DRL_std[i] for i in range(T)]
# plt.fill_between(x, u, l, color='#ffc951', alpha=0.2)

plt.plot(THL_mean, label='THLasso', color='red')
# u = [DRL_mean[i] + DRL_std[i] for i in range(T)]
# l = [DRL_mean[i] - DRL_std[i] for i in range(T)]
# plt.fill_between(x, u, l, color='#ffc951', alpha=0.2)

plt.plot(linucb_mean, label='LinUCB', color='pink')
# u = [linucb_mean[i] + linucb_std[i] for i in range(T)]
# l = [linucb_mean[i] - linucb_std[i] for i in range(T)]
# plt.fill_between(x, u, l, color='#ffc951', alpha=0.2)

plt.xlabel('Time Horizon (T)')
plt.ylabel(f'Averaged Cumulative-Regret per agent')
plt.title(f's_0/d: {s_0}/{d} and K: {K}')
plt.legend()
plt.savefig("precision.pdf", format="pdf", bbox_inches="tight")
plt.show()


