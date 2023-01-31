import numpy as np
import math


class FedThLassoServer:
    def __int__(self, environment, num_sim, T):
        self.environment = environment
        self.num_sim = num_sim
        self.T = T

    def run_algorithm(self):
        # An array for saving all cumulative regret
        cumulative_regret = [np.zeros(self.sim_num) for _ in range(self.T)]

        # Iterate each experiment
        for sim in range(self.sim_num):
            print(f'Iteration: {sim}.')

            agents = [Agent(self.d, self.K, self.M0, self.theta_star, 0.05)]
            log_counter = 1

            # Iterate time steps
            for t in range(self.T):
                if t % 200 == 0:
                    print(f'\tTime step: {t}')

                # Learning the model via ridge model
                for agent in agents:
                    agent.observe()
                    agent.pull()
                    agent.calculate_regret()
                    cumulative_regret[t][sim] += agent.cumulative_regret[-1] / self.N
                    agent.update_ridge_parameters()

                # Specifying the mode of algorithm (Lasso or ridge)
                Upd_SS = False
                if math.pow(2, log_counter) <= t < math.pow(2, log_counter + 1):
                    Upd_SS = True
                    log_counter += 1

                # Applying Lasso to update estimation of the support set
                if Upd_SS:
                    lambda_t = self.lam0 * math.sqrt(
                        (2 * math.log(t + 1) * math.log(self.d)) / (t + 1))  # Regularization-parameter

                    # Calculating Lasso estimator for each agent
                    lasso_estimators = []
                    for agent in agents:
                        lasso_estimators.append(agent.lasso(lambda_t))

                    # Feature extraction by union
                    support_hat = set()  # A set to save non-zero dimensions
                    for dimension in lasso_estimators:
                        for j in range(self.d):
                            if math.fabs(dimension[j]) >= lambda_t * self.N:
                                support_hat.add(j)

                    # 1. Updating extra dimensions
                    # 2. Updating ridge parameters
                    if len(support_hat) > 0:
                        extra_dimensions = list(np.delete([i for i in range(self.d)], list(support_hat), 0))
                        # 1. and 2.
                        for i in range(self.N):
                            agents[i].extra_dimensions = extra_dimensions
                            agents[i].update_ridge_parameters()

        regret_mean = [np.mean(cumulative_regret[t]) for t in range(self.T)]
        regret_std = [np.std(cumulative_regret[t]) for t in range(self.T)]

        return regret_mean, regret_std

