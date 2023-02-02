from agent import Agent
import math
import numpy as np


class Environment:
    def __init__(self, d, K, N, lnr_bandits, lam0):
        self.d = d
        self.K = K
        self.N = N
        self.lam0 = lam0
        self.lnr_bandits = lnr_bandits

        self.log_counter = 1
        self.agents = [Agent(self.d) for _ in range(N)]

    def reset(self):
        self.log_counter = 1
        self.agents = [Agent(self.d) for _ in range(self.N)]

    def run_step(self, t):
        if t % 200 == 0:
            print(f'\tTime step: {t}')

        regret = 0.0

        # Learning the model via ridge model
        for agent in self.agents:
            agent.choose_arm(self.lnr_bandits)
            regret += agent.cumulative_regret[-1] / self.N
            agent.update_parameters()

        # Specifying the mode of algorithm (Lasso or ridge)
        Upd_SS = False
        if math.pow(2, self.log_counter) <= t < math.pow(2, self.log_counter + 1):
            print(set(np.arange(self.d)) - set(self.agents[0].extra_dimensions))
            Upd_SS = True
            self.log_counter += 1

        # Applying Lasso to update estimation of the support set
        if Upd_SS:
            lambda_t = self.lam0 * math.sqrt(
                (2 * math.log(t + 1) * math.log(self.d)) / (t + 1))  # Regularization-parameter

            # Calculating Lasso estimator for each agent
            lasso_estimators = []
            for agent in self.agents:
                lasso_estimators.append(agent.lasso(lambda_t))

            # Feature extraction by union
            support_hat = set()  # A set to save non-zero dimensions
            for estimator in lasso_estimators:
                for j in range(self.d):
                    if math.fabs(estimator[j]) >= lambda_t * self.N:
                        support_hat.add(j)

            # 1. Updating extra dimensions
            # 2. Updating ridge parameters
            extra_dimensions = list(np.delete([i for i in range(self.d)], list(support_hat), 0))
            if extra_dimensions != self.agents[0].extra_dimensions:
                # 1. and 2.
                for agent in self.agents:
                    agent.extra_dimensions = extra_dimensions
                    agent.dimension_reduction_parameters()

        return regret
