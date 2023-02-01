import numpy as np
from sklearn.linear_model import Lasso


class Agent:
    def __init__(self, K, d):
        self.K = K
        self.d = d
        self.observed_rewards = np.array([], dtype=np.float32)  # Chosen context-vectors up to time t
        self.chosen_arms = np.empty((0, d), dtype=np.float32)  # Observed rewards up to time t
        self.theta_hat = np.zeros(d, dtype=np.float32)
        self.cumulative_regret = np.array([], dtype=np.float32)
        self.cumulative_regret = np.append(self.cumulative_regret, 0.0)
        self.extra_dimensions = []

        # Ridge case:
        self.M = np.identity(d, dtype=np.float32)  # Initialization of M with identity matrix with shape (d,d)
        self.b = np.zeros((d, 1), dtype=np.float32)  # Initialization of b with shape (d,1) and filling with zeros

    def choose_arm(self, lnr_bandit):
        context_vectors = lnr_bandit.generate_context()
        aa = np.delete(context_vectors, self.extra_dimensions, 1)
        selected_arm = context_vectors[np.argmax(np.dot(aa, self.theta_hat))]
        self.chosen_arms = np.vstack((self.chosen_arms, selected_arm))
        reward, regret = lnr_bandit.pull(selected_arm)
        self.observed_rewards = np.append(self.observed_rewards, reward)
        self.cumulative_regret = np.append(self.cumulative_regret, self.cumulative_regret[-1] + regret)

    # 1. Update ridge parameters M, b
    # 2. Update theta_hat
    def update_parameters(self):
        # 1.
        aa = self.chosen_arms[-1]
        aa = np.delete(aa, self.extra_dimensions, 0)
        self.M += np.dot(aa, aa.T)
        self.b += self.observed_rewards[-1] * np.reshape(aa, (-1, 1))
        # 2.
        self.theta_hat = np.ravel(np.matmul(np.linalg.inv(self.M), self.b))
        # print(self.M)
        # print(self.b)

    # 1. Reduce the dimension of ridge parameters after lasso estimation. (only in log_2 t time steps)
    # 2. Reduce the dimension of theta_hat
    def dimension_reduction_parameters(self):
        # 1.
        aa = self.chosen_arms
        aa = list(np.delete(aa, self.extra_dimensions, 1))
        self.M = np.identity(self.d - len(self.extra_dimensions))
        self.b = np.zeros([self.d - len(self.extra_dimensions), 1])
        for i in range(len(aa)):
            temp = aa[i].reshape([-1, 1])
            self.M += np.dot(temp, temp.T)
            self.b += self.observed_rewards[i] * temp

        # 2.
        self.theta_hat = np.ravel(np.matmul(np.linalg.inv(self.M), self.b))
        # self.theta_hat = np.delete(self.theta_hat, self.extra_dimensions, 0)

    # 1. Return lasso estimator with respect to observed data up to time step t
    def lasso(self, lambda_t):
        # print(f'check the size of chosen arms at each dimension reduction round: {np.shape(self.chosen_arms)}')
        lasso = Lasso(alpha=lambda_t)
        # print(np.shape(self.chosen_arms))
        # print(np.shape(self.observed_rewards))
        lasso.fit(self.chosen_arms, self.observed_rewards)

        return lasso.coef_
