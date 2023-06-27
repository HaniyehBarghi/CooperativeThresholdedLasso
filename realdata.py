import numpy as np
from matplotlib import pyplot as plt
import random
import warnings
from realbandits import RealBandits
from cooprtvthlassobandit import CTL
from sparsityaglasso import SparsityAgLasso
from environment import Environment
from thlassobandit import ThLassoBandit

# Fixing randomness!!!
random.seed(1)
np.random.seed(2)

# Removing warnings
warnings.filterwarnings("ignore")

# Defining global variables
T = 3000  # Time horizon
K = 10  # Number of arms
d = 69  # Dimension
sim_num = 15

# Defining Linear Bandits
real_bandits = RealBandits(K, d)

# C-CTL
print('\t\t\t\t\t Centralized Cooperative Thresholded Lasso')
N = 20
lam0 = 0.0001
xi = 3
env = Environment(d, K, N, real_bandits, lam0, 'log', xi, True)
CCTL = CTL(int(T/N) + 1, N, d, sim_num, env)
CCTL_mean, CCTL_std = CCTL.run_algorithm()

# D-CTL
print('\t\t\t\t\t Decentralized Cooperative Thresholded Lasso')
env = Environment(d, K, N, real_bandits, lam0, 'log', xi, False)
DCTL = CTL(int(T/N) + 1, N, d, sim_num, env)
DCTL_mean, DCTL_std = DCTL.run_algorithm()

# THL
print('\t\t\t\t\t Thresholded Lasso')
lam0 = 0.0001
alg_thl = ThLassoBandit(T, d, sim_num, real_bandits, lam0)
THL_mean, THL_std = alg_thl.run_algorithm()

# SA
print('\t\t\t\t\t Sparsity Agnostic Lasso')
alg_sal = SparsityAgLasso(T, d, sim_num, real_bandits, None)
SAL_mean, SAL_std = alg_sal.run_algorithm()

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

plt.xlabel('Number of Observations')
plt.ylabel(f'Averaged Cumulative-Regret per agent')
plt.legend()
plt.grid(color='#E5E7E8')
plt.savefig("realdata.pdf", format="pdf", bbox_inches="tight")
plt.show()
