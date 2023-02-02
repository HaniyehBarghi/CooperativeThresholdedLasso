#############################################################################
##### This code is implemented to return mean and standard deviation of #####
##### cumulative regret per agent for alg. Sparsity Agnostic Lasso.     #####
#############################################################################

import numpy as np
import math
from agent import Agent


class SparsityAgLasso:
    def __init__(self, T, d, sim_num, sigma, sA, lnr_bandits):
        self.T = T  # Time horizon
        self.d = d
        self.sim_num = sim_num
        self.lnr_bandits = lnr_bandits

        # other initialization
        x_max = np.round(np.max(np.linalg.norm(self.lnr_bandits.generate_context())))
        self.lambda_0 = 2 * sigma * x_max

    def run_algorithm(self):
        # An array for saving all cumulative regret
        cumulative_regret = [np.zeros(self.sim_num) for _ in range(self.T)]

        # Iterate each experiment
        for sim in range(self.sim_num):
            print(f'Iteration: {sim}.')

            agent = Agent(self.d)

            # Iterate time steps
            for t in range(self.T):
                if t % 200 == 0:
                    print(f'\tTime step: {t}')

                # Learning the model via lasso model
                agent.choose_arm(self.lnr_bandits)
                cumulative_regret[t][sim] = agent.cumulative_regret[-1]

                # Updating theta_hat
                lambda_t = self.lambda_0 * math.sqrt((4 * math.log(t + 1) + 2 * math.log(self.d)) / (t + 1))
                agent.lasso(lambda_t)

        regret_mean = [np.mean(cumulative_regret[t]) for t in range(self.T)]
        regret_std = [np.std(cumulative_regret[t]) for t in range(self.T)]

        return regret_mean, regret_std
