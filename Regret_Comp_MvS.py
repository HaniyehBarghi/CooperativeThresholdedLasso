################################################################################
####### This code is implemented to compare cumulative regret in FTL vs. #######
####### some single agent algorithms: SALasso, DRLasso, and THLasso.     #######
################################################################################

import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import sparse
import random
import warnings
from linearbandits import LinearBandits
from environment import Environment
from federatedthlassobandit import FederatedThLassoBandit
from sal import SAL
from drl import DRL
from thlassobandit import ThLassoBandit

# from sa import SA
# from dr import DRi

# Fixing randomness!!!
random.seed(1)
np.random.seed(2)

# Removing warnings
warnings.filterwarnings("ignore")

# Defining global variables
T = 300  # Time horizon
K = 10  # Number of arms
d = 200  # Dimension
s_0 = 5  # Sparsity parameter
s_2 = 0.3  # Minimum of absolute value of each non-zero dimension
num_sim = 2
sA = 1
sigma = 0.05

# Defining model parameter (theta_star) #todo
theta_star = sparse.random(d, 1, density=s_0 / d).toarray()
for j in range(len(theta_star)):
    if theta_star[j] != 0:
        theta_star[j] = random.uniform(s_2, 1)

print(f'True support set: {np.nonzero(theta_star)}.')

# Covariance Matrix Initialization #todo
sigma_sq = 1.
rho_sq = 0.7
M0 = (sigma_sq - rho_sq) * np.eye(K) + rho_sq * np.ones((K, K))  # Covariance Matrix

# Constructing the environment
lnr_bandits = LinearBandits(d, M0, theta_star, sigma, sA)

# Calculating regret [1 * T] for each algorithm: FTL(15) - TL - SA - DR

# # FTL (15)
N = 10
c = 0.001
lam0 = 4 * sigma * sA * math.sqrt(c)
env = Environment(d, K, N, lam0, lnr_bandits)
ftl = FederatedThLassoBandit(env, num_sim, T, K, d, N)
FTL_mean, FTL_std = ftl.run()

# DR
# alg_drl = DRL(T, K, M0, theta_star, sim_num)
# DRL_mean, DRL_std = alg_drl.run_algorithm()
#
# # # TL
# THL = ThLassoBandit(T, K, M0, theta_star, sim_num)
# THL_mean, THL_std = THL.run_algorithm()
# #
# # SA
# alg_sal = SAL(T, K, M0, theta_star, sim_num)
# SAL_mean, SAL_std = alg_sal.run_algorithm()

# Plotting the result
x = [t for t in range(T)]

plt.plot(FTL_mean, label=f'FTLasso, N={N}', color='#56b4e9')
u = [FTL_mean[i] + FTL_std[i] for i in range(T)]
l = [FTL_mean[i] - FTL_std[i] for i in range(T)]
plt.fill_between(x, u, l, color='#9ad2f2', alpha=0.5)

# plt.plot(SAL_mean, label='SALasso', color='#009E73')
# u = [SAL_mean[i] + SAL_std[i] for i in range(T)]
# l = [SAL_mean[i] - SAL_std[i] for i in range(T)]
# plt.fill_between(x, u, l, color='#00ebab', alpha=0.2)
#
# plt.plot(DRL_mean, label='DRLasso', color='#E69F00')
# u = [DRL_mean[i] + DRL_std[i] for i in range(T)]
# l = [DRL_mean[i] - DRL_std[i] for i in range(T)]
# plt.fill_between(x, u, l, color='#ffc951', alpha=0.2)
#
# plt.plot(THL_mean, label='THLasso', color='#0072B2')
# u = [THL_mean[i] + THL_std[i] for i in range(T)]
# l = [THL_mean[i] - THL_std[i] for i in range(T)]
# plt.fill_between(x, u, l, color='#8DCEF3', alpha=0.2)

plt.xlabel('Time Horizon (T)')
plt.ylabel(f'Averaged Cumulative-Regret per agent')
plt.title(f's_0/d: {s_0}/{d} and K: {K}')
plt.legend()
plt.savefig("precision.pdf", format="pdf", bbox_inches="tight")
plt.show()


# plt.plot(FTL_mean, label='FTLasso, N=10', color='#56b4e9')

# plt.plot(SAL_mean, label='SALasso', color='#009E73')
#
# plt.plot(DRL_mean, label='DRLasso', color='#E69F00')
#
# plt.plot(THL_mean, label='THLasso', color='#0072B2')

# plt.xlabel('Time Horizon (T)')
# plt.ylabel(f'Averaged Cumulative-Regret per agent')
# plt.title(f's_0/d: {s_0}/{d} and K: {K}')
# plt.legend()
# plt.savefig("precision.pdf", format="pdf", bbox_inches="tight")
# plt.show()