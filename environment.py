from linearbandits import LinearBandits
from agent import Agent
import math
import numpy as np


class Environment:
    def __init__(self, d, K, N, lam0, lnr_bandits):
        self.d = d
        self.K = K
        self.lam0 = lam0
        self.N = N
        self.lnr_bandits = lnr_bandits
        self.agents = [Agent(self.K, self.d) for _ in range(self.N)]
        self.log_counter = 0

    def reset(self):
        self.agents = [Agent(self.K, self.d) for _ in range(self.N)]
        self.log_counter = 0

    def run_step(self, t):
        if t % 200 == 0:
            print(f'\tTime step: {t}')

        # print('-----------------------------')
        # print(f'\tTime step: {t}')
        # print(self.agents[0].chosen_arms)
        # print(self.agents[0].observed_rewards)
        # print(self.agents[0].theta_hat)
        # print(self.agents[0].extra_dimensions)
        # print(self.agents[0].M)
        # print(self.agents[0].b)
        # print('--------------')

        # Learning the model via ridge model
        for agent in self.agents:
            agent.choose_arm(self.lnr_bandits)
            agent.update_parameters()

        # Specifying the mode of algorithm (Lasso or ridge)
        Upd_SS = False
        if math.pow(2, self.log_counter) <= t < math.pow(2, self.log_counter + 1):
            Upd_SS = True
            self.log_counter += 1

        # Applying Lasso to update estimation of the support set
        support_hat = set()  # A set to save non-zero dimensions
        if Upd_SS:
            # Regularization-parameter
            lambda_t = self.lam0 * math.sqrt((2 * math.log(t + 1) * math.log(self.d)) / (t + 1))

            # Calculating Lasso estimator for each agent
            lasso_estimators = []
            for agent in self.agents:
                lasso_estimators.append(agent.lasso(lambda_t))

            # Feature extraction by union
            for estimator in lasso_estimators:
                for j in range(self.d):
                    if math.fabs(estimator[j]) > 0:
                        support_hat.add(j)

            print(f'Estimated support at time {t},   {support_hat}.')

        # 1. Updating extra dimensions
        # 2. Updating ridge parameters
        extra_dimensions = list(np.delete(np.arange(self.d), list(support_hat)))
        if set(extra_dimensions) != set(self.agents[0].extra_dimensions):
            # 1. and 2.
            for agent in self.agents:
                agent.extra_dimensions = extra_dimensions
                agent.dimension_reduction_parameters()

