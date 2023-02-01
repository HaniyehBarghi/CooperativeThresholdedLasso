####################################################################
##### This code is implemented to observe retrieval evaluation #####
##### in multi-agent case & when increasing number of agents.  #####
####################################################################

import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import sparse
import random
import warnings
from agent import Agent

# Fixing randomness!!!
random.seed(1)
np.random.seed(2)

# Removing warnings
warnings.filterwarnings("ignore")

# Defining global variables
T = 200  # Time horizon
K = 10  # Number of arms
d = 500  # Dimension
f = 0.1
c = 1  # Positive constant
s_0 = 5  # Support set's cardinality
s_A = 1  # Maximum absolute value of context-vector (component-wise)
sigma = 0.05  # The sub-gaussian parameter of noise: standard-deviation
s_2 = 0.3  # Minimum of absolute value of each non-zero dimension

# Initiating lambda_0
lambda_0 = 4 * sigma * s_A * math.sqrt(c)

# Defining model parameter (theta_star)
theta_star = sparse.random(d, 1, density=s_0 / d).toarray()
support = []
for j in range(len(theta_star)):
    if theta_star[j] != 0:
        theta_star[j] = random.uniform(s_2, 1)
        support.append(j)

# An array for saving all cumulative regret
precision = np.zeros(T)
recall = np.zeros(T)

# Gram Matrix Initialization (from paper Oh,2021)
sigma_sq = 1.
rho_sq = 0.7
M_0 = (sigma_sq - rho_sq) * np.eye(K) + rho_sq * np.ones((K, K))  # Covariance Matrix


# Return TP, FP, and FN for every estimated theta at each time step t
def dimension_explore(vector, new_support):
    tp = 0
    fp = 0
    fn = len(support) - len(new_support)
    for j in range(len(vector)):
        if j in new_support:
            if vector[j] != 0.0:
                tp += 1
            else:
                fn += 1
        else:
            if vector[j] != 0:
                fp += 1
    return tp, fp, fn


# Running the algorithm and retrieval evaluation at time T
def multi_agent(N):
    # Defining agents [total_iteration * N]
    agents = []
    for i in range(total_iteration):
        agents.append([Agent(d, K, M_0, theta_star, sigma) for _ in range(N)])

    # An array for saving all precisions and recalls
    precision = [np.zeros(total_iteration) for _ in range(T)]
    recall = [np.zeros(total_iteration) for _ in range(T)]

    # Iterate each experiment
    for iteration in range(total_iteration):
        print(f'Iteration: {iteration}.')
        new_support = support
        log_counter = 1

        # Iterate time steps
        for t in range(T):
            if t % 200 == 0:
                print(f'\tTime step: {t}')

            # Learning the model via ridge model
            for agent in agents[iteration]:
                agent.observe()
                agent.pull()
                agent.update_parameters()

            # Specifying the mode of algorithm (Lasso or ridge)
            Upd_SS = False
            if math.pow(2, log_counter) <= t < math.pow(2, log_counter + 1):
                Upd_SS = True
                log_counter += 1

            # Applying Lasso to update estimation of the support set
            if Upd_SS:
                lambda_t = lambda_0 * math.sqrt(
                    (2 * math.log(t + 1) * math.log(d)) / (t + 1))  # Regularization-parameter

                # Calculating Lasso estimator for each agent
                lasso_estimators = []
                for agent in agents[iteration]:
                    lasso_estimators.append(agent.lasso(lambda_t))

                # Feature extraction by union
                support_hat = set()  # A set to save non-zero dimensions
                for dimension in lasso_estimators:
                    for j in range(d):
                        if math.fabs(dimension[j]) > N * lambda_t * f:
                            support_hat.add(j)
                support_hat = list(support_hat)
                support_hat.sort()

                # Updating new support set (with respect to dimension reduction)
                new_support = []
                for j in range(len(support_hat)):
                    if support_hat[j] in support:
                        new_support.append(j)

                # 1. Updating extra dimensions
                # 2. Updating ridge parameters
                if len(support_hat) > 0:
                    extra_dimensions = list(np.delete([i for i in range(d)], support_hat, 0))
                    # 1. and 2.
                    for i in range(N):
                        agents[iteration][i].extra_dimensions = extra_dimensions
                        agents[iteration][i].update_parameters()

            # 1. Summing up TP, FP, FN for all agents
            for i in range(N):
                tp, fp, fn = dimension_explore(agents[iteration][i].theta_hat, new_support)
                if tp > 0:
                    precision[t][iteration] += tp / (tp + fp)
                    recall[t][iteration] += tp / (tp + fn)

            precision[t][iteration] /= N
            recall[t][iteration] /= N

    # Releasing RAM
    del agents

    precision_mean = [np.mean(precision[t]) for t in range(T)]
    recall_mean = [np.mean(recall[t]) for t in range(T)]

    precision_std = [np.std(precision[t]) for t in range(T)]
    recall_std = [np.std(recall[t]) for t in range(T)]

    return precision_mean, precision_std, recall_mean, recall_std


# Running the algorithm for different number of agents
total_iteration = 50

N = 5
pre_mean_A, pre_std_A, rec_mean_A, rec_std_A = multi_agent(N)

N = 25
pre_mean_B, pre_std_B, rec_mean_B, rec_std_B = multi_agent(N)

N = 40
pre_mean_C, pre_std_C, rec_mean_C, rec_std_C = multi_agent(N)

# Plot precision for different number of agents
plt.plot(pre_mean_A, label='N = 5', color='#E69F00')
# u = [pre_mean_A[i] + pre_std_A[i] for i in range(len(pre_mean_A))]
# d = [pre_mean_A[i] - pre_std_A[i] for i in range(len(pre_mean_A))]
# x = [t for t in range(T)]
# plt.fill_between(x, u, d, color='gray', alpha=0.2)
plt.plot(pre_mean_B, label='N = 5', color='#56B4E9')
plt.plot(pre_mean_C, label='N = 5', color='#009E73')
plt.xlabel('Time Horizon (T)')
plt.ylabel(f'Averaged precision per agent')
plt.title(f's_0/d: {s_0} / {d}, K: {K}, and $\sigma$: {sigma}')
plt.legend()
plt.savefig("precision.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Plot recall for different number of agents
plt.plot(rec_mean_A, label='N = 5', color='#E69F00')
# u = [rec_mean_A[i] + rec_std_A[i] for i in range(len(rec_mean_A))]
# d = [rec_mean_A[i] - rec_std_A[i] for i in range(len(rec_mean_A))]
# x = [t for t in range(T)]
# plt.fill_between(x, u, d, color='gray', alpha=0.2)
plt.plot(rec_mean_B, label='N = 5', color='#56B4E9')
plt.plot(rec_mean_C, label='N = 5', color='#009E73')
plt.xlabel('Time Horizon (T)')
plt.ylabel(f'Averaged recall per agent')
plt.title(f's_0/d: {s_0} / {d}, K: {K}, and $\sigma$: {sigma}')
plt.legend()
plt.savefig("recall.pdf", format="pdf", bbox_inches="tight")
plt.show()
