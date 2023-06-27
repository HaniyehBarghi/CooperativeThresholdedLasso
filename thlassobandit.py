import numpy as np
import math
from sklearn import linear_model
from agent import Agent


class ThLassoBandit:
    def __init__(self, T, d, sim_num, lnr_bandits, lam0):
        self.T = T      # Time horizon
        self.d = d     # Number of arms
        self.lnr_bandits = lnr_bandits
        self.sim_num = sim_num
        self.lam0 = lam0

    def run_algorithm(self):
        # A matrix for saving all cumulative regret [T * sim_num]
        cumulative_regret = [np.zeros(self.sim_num) for _ in range(self.T)]

        # Iterating each simulation
        for sim in range(self.sim_num):
            print(f'Simulation: {sim}.')

            # Defining an agent for current simulation round
            agent = Agent(self.d, [])

            # Iterating time steps
            for t in range(self.T):
                if t % 200 == 0:
                    print(f'\tTime step: {t}')

                agent.choose_arm(self.lnr_bandits)
                cumulative_regret[t][sim] += agent.cumulative_regret[-1]

                ### Updating support set ###
                # Lasso estimator
                lam_t = self.lam0 * math.sqrt(
                        (2 * math.log(t + 1) * math.log(self.d)) / (t + 1))  # Regularization-parameter
                lasso_estimator = agent.lasso(lam_t)
                # Feature extraction
                S0 = []  # A set to save non-zero dimensions
                for j in range(self.d):
                    if math.fabs(lasso_estimator[j]) >= 4 * lam_t:
                        S0.append(j)
                S1 = []
                for j in S0:
                    if math.fabs(lasso_estimator[j]) >= 4 * lam_t * math.sqrt(len(S0)):
                        S1.append(j)
                if len(S1) > 0:
                    agent.extra_dimensions = list(np.delete(np.arange(self.d), list(S1), 0))

                # Updating theta_hat
                aa = agent.chosen_arms
                aa = list(np.delete(aa, agent.extra_dimensions, 1))
                reg = linear_model.LinearRegression()
                reg.fit(aa, agent.observed_rewards)
                agent.theta_hat = reg.coef_

        regret_mean = [np.mean(cumulative_regret[t]) for t in range(self.T)]
        regret_std = [np.std(cumulative_regret[t]) for t in range(self.T)]

        return regret_mean, regret_std

