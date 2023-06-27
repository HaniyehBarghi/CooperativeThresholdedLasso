import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
import random
import math
import warnings
from linearbandits import LinearBandits
from cooprtvthlassobandit import CTL
from sparsityaglasso import SparsityAgLasso
from drl import DRL
from environment import Environment
from thlassobandit import ThLassoBandit

# Fixing randomness!!!
random.seed(1)
np.random.seed(2)

# Removing warnings
warnings.filterwarnings("ignore")

# Defining global variables
T = 400  # Time horizon
K = 5  # Number of arms
d = 500  # Dimension
s_0 = 5  # Sparsity parameter
sim_num = 15
sigma = 0.05
sA = 5

# Defining model parameter (theta_star)
theta_star = sparse.random(d, 1, density=s_0 / d).toarray()
for j in range(len(theta_star)):
    if theta_star[j] != 0:
        theta_star[j] = random.uniform(0.5, 2)
print(f'True support set: {np.nonzero(theta_star)[0]}.')

# Covariance Matrix Initialization
sigma_sq = 1.
rho_sq = 0.7
M0 = (sigma_sq - rho_sq) * np.eye(K) + rho_sq * np.ones((K, K))  # Covariance Matrix

# Defining Linear Bandits
lnr_bandits = LinearBandits(d, M0, theta_star, sigma, sA)

# C-CTL
print('\t\t\t\t\t Centralized Cooperative Thresholded Lasso')
N = 25
c = 1 # Positive constant (required for the theoretical part)
lam0 = 4 * sigma * sA * math.sqrt(c)
xi = 3
env = Environment(d, K, N, lnr_bandits, lam0, 'log', xi, True)
CCTL = CTL(int(T/N) + 1, N, d, sim_num, env)
CCTL_mean, CCTL_std = CCTL.run_algorithm()

# D-CTL
print('\t\t\t\t\t Decentralized Cooperative Thresholded Lasso')
env = Environment(d, K, N, lnr_bandits, lam0, 'log', xi, False)
DCTL = CTL(int(T/N) + 1, N, d, sim_num, env)
DCTL_mean, DCTL_std = DCTL.run_algorithm()

# THL
print('\t\t\t\t\t Thresholded Lasso')
c = 0.0001
lam0 = 4 * sigma * sA * math.sqrt(c)
alg_thl = ThLassoBandit(T, d, sim_num, lnr_bandits, lam0)
THL_mean, THL_std = alg_thl.run_algorithm()

# SA
print('\t\t\t\t\t Sparsity Agnostic Lasso')
alg_sal = SparsityAgLasso(T, d, sim_num, lnr_bandits, sigma)
SAL_mean, SAL_std = alg_sal.run_algorithm()

# DR
print('\t\t\t\t\t Doubly Robust Lasso')
alg_drl = DRL(T, K, M0, theta_star, sim_num)
DRL_mean, DRL_std = alg_drl.run_algorithm()


#################### Plotting the result ####################
x = [t * N for t in range(int(T / N) + 1)]
plt.plot(x, CCTL_mean, label=fr'CCTLasso, N={N}, $\xi ={xi}$', color='red', linestyle='-')
u = [CCTL_mean[i] + CCTL_std[i] for i in range(int(T / N) + 1)]
l = [CCTL_mean[i] - CCTL_std[i] for i in range(int(T / N) + 1)]
plt.fill_between(x, u, l, color='#F9F2C9', alpha=0.2)


plt.plot(x, DCTL_mean, label=fr'DCTLasso, N={N}, $\xi ={xi}$', color='#0057B7', linestyle='-.')
u = [DCTL_mean[i] + DCTL_std[i] for i in range(int(T / N) + 1)]
l = [DCTL_mean[i] - DCTL_std[i] for i in range(int(T / N) + 1)]
plt.fill_between(x, u, l, color='#A8CEF5', alpha=0.2)

x = [t for t in range(T)]
plt.plot(THL_mean, label='THLasso', color='#FFD600', linestyle='--')
u = [THL_mean[i] + THL_std[i] for i in range(T)]
l = [THL_mean[i] - THL_std[i] for i in range(T)]
plt.fill_between(x, u, l, color='#F5E37F', alpha=0.2)

plt.plot(SAL_mean, label='SALasso', color='#16A93B', linestyle='-.')
u = [SAL_mean[i] + SAL_std[i] for i in range(T)]
l = [SAL_mean[i] - SAL_std[i] for i in range(T)]
plt.fill_between(x, u, l, color='#8DDFA1', alpha=0.2)

plt.plot(DRL_mean, label='DRLasso', color='#F9820B', linestyle=':')
u = [DRL_mean[i] + DRL_std[i] for i in range(T)]
l = [DRL_mean[i] - DRL_std[i] for i in range(T)]
plt.fill_between(x, u, l, color='#FACAA4', alpha=0.2)


plt.xlabel('Number of Observations')
plt.ylabel(f'Averaged Cumulative-Regret per agent')
plt.title(f'$s_0$={s_0}, d={d}, K={K}, and corr={rho_sq}')
plt.legend()
plt.grid(color='#E5E7E8')
plt.savefig("MvS.pdf", format="pdf", bbox_inches="tight")
plt.show()
