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
sim_num = 5
sigma = 0.05
N = 10
c = 0.1  # Positive constant
sA = 1  # Maximum absolute value of context-vector (component-wise)
lam0 = 4 * sigma * sA * math.sqrt(c)


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
print('\t\t\t\t\t Federated Thresholded Lasso - Logarithmic Communication')
env = Environment(d, K, N, lnr_bandits, lam0, 'log', 2)
FTL_log2 = FTL(T, N, d, sim_num, env)
mean_log2, std_log2 = FTL_log2.run_algorithm()

print('\t\t\t\t\t Federated Thresholded Lasso - Constant Communication')
lam0 = 4 * sigma * sA * math.sqrt(c)
env = Environment(d, K, N, lnr_bandits, lam0, 'cons', 20)
FTL_cons = FTL(T, N, d, sim_num, env)
mean_cons, std_cons = FTL_cons.run_algorithm()


print('\t\t\t\t\t Federated Thresholded Lasso - Each step Communication')
lam0 = 4 * sigma * sA * math.sqrt(c)
env = Environment(d, K, N, lnr_bandits, lam0, 'each', 1)
FTL_each = FTL(T, N, d, sim_num, env)
mean_each, std_each = FTL_each.run_algorithm()

# Plotting the result

plt.plot(mean_log2, label=f'Logarithmic Communication', color='#56b4e9')
# u = [FTL_mean[i] + FTL_std[i] for i in range(T)]
# l = [FTL_mean[i] - FTL_std[i] for i in range(T)]
# plt.fill_between(x, u, l, color='#9ad2f2', alpha=0.2)


plt.plot(mean_cons, label='Constant delta Communication', color='#009E73')
# u = [SAL_mean[i] + SAL_std[i] for i in range(T)]
# l = [SAL_mean[i] - SAL_std[i] for i in range(T)]
# plt.fill_between(x, u, l, color='#00ebab', alpha=0.2)

plt.plot(mean_each, label='Each step Communication', color='#E69F00')
# u = [DRL_mean[i] + DRL_std[i] for i in range(T)]
# l = [DRL_mean[i] - DRL_std[i] for i in range(T)]
# plt.fill_between(x, u, l, color='#ffc951', alpha=0.2)

plt.xlabel('Time Horizon (T)')
plt.ylabel(f'Averaged Cumulative-Regret per agent')
plt.title(f's_0/d: {s_0}/{d}, K: {K}, and N: {N}')
plt.legend()
plt.savefig("precision.pdf", format="pdf", bbox_inches="tight")
plt.show()


