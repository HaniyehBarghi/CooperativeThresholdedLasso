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


class SparsityAgLasso:
    def __init__(self, T, d, sim_num, lnr_bandits, sigma):
        self.T = T
        self.d = d
        self.sim_num = sim_num
        self.lnr_bandits = lnr_bandits

        # other initialization
        x_max = np.linalg.norm(lnr_bandits.generate_context()[0])   # upper bound on l_2 norm of every context-vector
        self.lam0 = 2 * sigma * x_max

    def run_algorithm(self):
        # An array for saving all cumulative regret
        cumulative_regret = [np.zeros(self.sim_num) for _ in range(self.T)]

        # Iterate each experiment
        for sim in range(self.sim_num):
            print(f'Iteration: {sim}.')

            # Defining agents
            agent = Agent(self.d, [])

            # Iterate time steps
            for t in range(self.T):
                if t % 200 == 0:
                    print(f'\tTime step: {t}')

                # Choose arm
                agent.choose_arm(self.lnr_bandits)
                cumulative_regret[t][sim] += agent.cumulative_regret[-1]
                # Update Lasso parameter
                lambda_t = self.lam0 * math.sqrt((4 * math.log(t + 1) + 2 * math.log(self.d)) / (t + 1))
                agent.theta_hat = agent.lasso(lambda_t)

        regret_mean = [np.mean(cumulative_regret[t]) for t in range(self.T)]
        regret_std = [np.std(cumulative_regret[t]) for t in range(self.T)]

        return regret_mean, regret_std



