import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
import random
import math
import warnings
from subgoss import subgoss
from linearbandits import LinearBandits
from cooprtvthlassobandit import CTL
from environment import Environment

# Fixing randomness!!!
random.seed(1)
np.random.seed(2)

# Removing warnings
warnings.filterwarnings("ignore")

# Defining global variables
T = 511  # Time horizon
K = 5  # Number of arms
d = 30  # Dimension
s_0 = 2  # Sparsity parameter
sim_num = 15
sigma = 0.05
sA = 5
N = 10

# Defining model parameter (theta_star)
theta_star = sparse.random(d, 1, density=s_0 / d).toarray()
for j in range(len(theta_star)):
    if theta_star[j] != 0:
        theta_star[j] = random.uniform(0.3, 1)
print(f'True support set: {np.nonzero(theta_star)[0]}.')

# Covariance Matrix Initialization
sigma_sq = 1.
rho_sq = 0.3
M0 = (sigma_sq - rho_sq) * np.eye(K) + rho_sq * np.ones((K, K))  # Covariance Matrix

# Defining Linear Bandits
lnr_bandits = LinearBandits(d, M0, theta_star, sigma, sA)

# Subgoss 9
Ad_matrix = np.ones((N, N)) - np.eye(N)
b = 2
num_j = int(math.log(T + 1, b))
print(num_j)
sn = int(d / s_0)
subgoss1 = subgoss(num_j, sim_num, Ad_matrix, b, N, sn, s_0, d, K, lnr_bandits)
subgoss_mean1, subgoss_std1 = subgoss1.run()

# C-CTL
print('\t\t\t\t\t Centralized Cooperative Thresholded Lasso')
c = 0.1  # Positive constant
lam0 = 4 * sigma * sA * math.sqrt(c)
env = Environment(d, K, N, lnr_bandits, lam0, 'log', 2, True)
CCTL = CTL(T, N, d, sim_num, env)
CCTL_mean, CCTL_std = CCTL.run_algorithm()

# D-CTL
print('\t\t\t\t\t Decentralized Cooperative Thresholded Lasso')
env = Environment(d, K, N, lnr_bandits, lam0, 'log', 2, False)
DCTL = CTL(T, N, d, sim_num, env)
DCTL_mean, DCTL_std = DCTL.run_algorithm()

#################### Plotting the result ####################
x = [t for t in range(T)]

plt.plot(x, subgoss_mean1, label=f'Subgoss', color='#16A93B', linestyle='--')
u = [subgoss_mean1[i] + subgoss_std1[i] for i in range(T)]
l = [subgoss_mean1[i] - subgoss_std1[i] for i in range(T)]
plt.fill_between(x, u, l, color='#8DDFA1', alpha=0.2)

plt.plot(x, CCTL_mean, label=f'CCTLasso', color='red', linestyle='-')
u = [CCTL_mean[i] + CCTL_std[i] for i in range(T)]
l = [CCTL_mean[i] - CCTL_std[i] for i in range(T)]
plt.fill_between(x, u, l, color='#F9F2C9', alpha=0.2)

plt.plot(x, DCTL_mean, label=f'DCTLasso', color='#0057B7', linestyle='-.')
u = [DCTL_mean[i] + DCTL_std[i] for i in range(T)]
l = [DCTL_mean[i] - DCTL_std[i] for i in range(T)]
plt.fill_between(x, u, l, color='#A8CEF5', alpha=0.2)

plt.xlabel('Time (T)')
plt.ylabel(f'Averaged Cumulative-Regret per agent')
plt.title(f'$s_0$={s_0}, d={d}, K={K}, and N={N}')
plt.legend()
plt.grid(color='#E5E7E8')
plt.savefig("MvM.pdf", format="pdf", bbox_inches="tight")
plt.show()
