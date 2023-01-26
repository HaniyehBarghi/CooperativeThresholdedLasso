#######################################################################################################
###### This code implements the algorithm Thresholded Lasso(Ariu, 2022) as a class. The function ######
###### run_algorithm returns mean and standard deviation of cumulative regret of an agent.       ######
#######################################################################################################

import numpy as np
import math
from sklearn import linear_model
from agent import Agent


class ThLassoBandit:
    def __init__(self, T, K, M0, theta_star, sim_num):
        self.T = T  # Time horizon
        self.d = len(theta_star)  # Dimension
        self.sim_num = sim_num
        self.K = K
        self.M0 = M0
        self.theta_star = theta_star

        # other initialization
        c = 10  # Positive constant
        s_A = 1  # Maximum absolute value of context-vector (component-wise)
        self.lambda_0 = 4 * 0.05 * s_A * math.sqrt(c)

    def run_algorithm(self):
        # A matrix for saving all cumulative regret [T * sim_num]
        cumulative_regret = [np.zeros(self.sim_num) for _ in range(self.T)]

        # Iterating each simulation
        for sim in range(self.sim_num):
            print(f'Simulation: {sim}.')

            # Defining an agent for current simulation round
            agent = Agent(self.d, self.K, self.M0, self.theta_star, 0.05)

            # Iterating time steps
            for t in range(self.T):
                if t % 200 == 0:
                    print(f'\tTime step: {t}')

                agent.observe()
                agent.pull()
                agent.calculate_regret()
                cumulative_regret[t][sim] += agent.cumulative_regret[-1]

                ### Updating support set ###
                # Lasso estimator
                lambda_t = self.lambda_0 * math.sqrt(
                        (2 * math.log(t + 1) * math.log(self.d)) / (t + 1))  # Regularization-parameter
                lasso_estimator = agent.lasso(lambda_t)
                # Feature extraction
                S0 = []  # A set to save non-zero dimensions
                for j in range(self.d):
                    if math.fabs(lasso_estimator[j]) >= 4 * lambda_t:
                        S0.append(j)
                S1 = []
                for j in S0:
                    if math.fabs(lasso_estimator[j]) >= 4 * lambda_t * math.sqrt(len(S0)):
                        S1.append(j)
                if len(S1) > 0:
                    agent.extra_dimensions = list(np.delete([j for j in range(self.d)], list(S1), 0))
                ##########

                # Updating theta_hat
                aa = agent.A_train
                aa = list(np.delete(aa, agent.extra_dimensions, 1))
                yy = np.ravel(agent.Y_train)
                reg = linear_model.LinearRegression()
                reg.fit(aa, yy)
                agent.theta_hat = reg.coef_

        regret_mean = [np.mean(cumulative_regret[t]) for t in range(self.T)]
        regret_std = [np.std(cumulative_regret[t]) for t in range(self.T)]

        return regret_mean, regret_std



