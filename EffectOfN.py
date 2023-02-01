#####################################################################
######## This code is implemented to observe the effect of ##########
######## number of agents to the total cumulative regret.  ##########
#####################################################################

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
total_iteration = 1
T = 750     # Time horizon
K = 30      # Number of arms
d = 100     # Dimension
c = 10      # Positive constant
s_0 = 5     # Support set's cardinality
s_A = 1     # Maximum absolute value of context-vector (component-wise)
sigma = 0.05    # The sub-gaussian parameter of noise: standard-deviation
s_2 = 0.3       # Minimum of absolute value of each non-zero dimension

# Initiating lambda_0
lambda_0 = 4 * sigma * s_A * math.sqrt(c)

# Defining model parameter (theta_star)
theta_star = sparse.random(d, 1, density=s_0 / d).toarray()
for j in range(len(theta_star)):
    if theta_star[j] != 0:
        theta_star[j] = random.uniform(s_2, 1)

# Gram Matrix Initialization (from paper Oh,2021)
sigma_sq = 1.
rho_sq = 0.7
M_0 = (sigma_sq - rho_sq) * np.eye(K) + rho_sq * np.ones((K, K))  # Covariance Matrix


# Running the algorithm and return cumulative regret at time T
def multi_agent(N):
    # Defining agents [total_iteration * N]
    agents = []
    for i in range(total_iteration):
        agents.append([Agent(d, K, M_0, theta_star, sigma) for i in range(N)])

    # A variable for saving cumulative regret
    cumulative_regret = 0

    # Iterate each experiment
    for iteration in range(total_iteration):
        print(f'\tIteration: {iteration}.')
        log_counter = 1

        # Iterate time steps
        for t in range(T):
            if t % 200 == 0:
                print(f'\t\tTime step: {t}')

            # Learning the model via ridge model
            for agent in agents[iteration]:
                agent.observe()
                agent.pull()
                agent.calculate_regret()
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
                        if math.fabs(dimension[j]) >= lambda_t * N:
                            support_hat.add(j)

                # 1. Updating extra dimensions
                # 2. Updating ridge parameters
                if len(support_hat) > 0:
                    extra_dimensions = list(np.delete([i for i in range(d)], list(support_hat), 0))
                    # 1. and 2.
                    for i in range(N):
                        agents[iteration][i].extra_dimensions = extra_dimensions
                        agents[iteration][i].update_parameters()

        # 1. Summing the cumulative regret of agents and store it
        # 2. Deleting each used agent #todo agent = null
        for i in range(N):
            cumulative_regret += agents[iteration][i].cumulative_regret[-1]
            agents[iteration][i] = Agent(d, K, M_0, theta_star, sigma)

    # Releasing RAM
    del agents

    return cumulative_regret / (total_iteration * N)


# Running the algorithm for different number of agents
N = 1
compare = []
while N < 80:
    print(f'Number of agents: {N}.')
    compare.append(multi_agent(N))
    N += 7


plt.plot(compare, color='darkgreen')
plt.xlabel('Number of agents (N)')
plt.ylabel(f'Averaged cumulative regret at T = {T}')
plt.title(f's_0/d: {s_0} / {d}, K: {K}, and $\sigma$: {sigma}')
plt.legend()
plt.savefig("cmp.pdf", format="pdf", bbox_inches="tight")
plt.show()