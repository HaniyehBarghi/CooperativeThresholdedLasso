#############################################################################
##### This code is implemented to return mean and standard deviation of #####
##### cumulative regret per agent for alg. Sparsity Agnostic Lasso.     #####
#############################################################################

import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import sparse
import random
import warnings
from agent import Agent


class SAL:
    def __init__(self, T, K, M_0, theta_star, sim_num):
        self.T = T  # Time horizon
        self.d = len(theta_star)  # Dimension
        self.sim_num = sim_num

        # other initialization
        x_max = 15      # upper bound on l_2 norm of every context-vector
        self.lambda_0 = 2 * 0.05 * x_max

        # Defining agents [sim_num]
        self.agents = []
        for i in range(self.sim_num):
            self.agents.append(Agent(self.d, K, M_0, theta_star, 0.05))

    def run_algorithm(self):
        # An array for saving all cumulative regret
        cumulative_regret = [np.zeros(self.sim_num) for _ in range(self.T)]

        # Iterate each experiment
        for sim in range(self.sim_num):
            print(f'Iteration: {sim}.')

            # Iterate time steps
            for t in range(self.T):
                if t % 200 == 0:
                    print(f'\tTime step: {t}')

                # Learning the model via lasso model
                # Regularization-parameter
                lambda_t = self.lambda_0 * math.sqrt((4 * math.log(t + 1) + 2 * math.log(self.d)) / (t + 1))
                self.agents[sim].observe()
                self.agents[sim].observe()
                self.agents[sim].pull()
                self.agents[sim].calculate_regret()
                cumulative_regret[t][sim] += self.agents[sim].cumulative_regret[-1]
                self.agents[sim].lasso(lambda_t)

        regret_mean = [np.mean(cumulative_regret[t]) for t in range(self.T)]
        regret_std = [np.std(cumulative_regret[t]) for t in range(self.T)]

        return regret_mean, regret_std



