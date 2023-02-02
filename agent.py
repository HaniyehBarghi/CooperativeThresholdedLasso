import numpy as np
from sklearn.linear_model import Lasso


class Agent:
    def __init__(self, d):
        self.d = d
        self.theta_hat = np.ones(d)
        self.chosen_arms = np.empty((0, d), dtype=np.float32)  # Selected context-vectors up to time t
        self.observed_rewards = np.array([], dtype=np.float32)  # Observed rewards up to time t
        self.cumulative_regret = np.array([], dtype=np.float32)
        self.cumulative_regret = np.append(self.cumulative_regret, 0.0)
        self.extra_dimensions = []  # non-main dimensions at time t

        # Ridge case:
        self.M = np.identity(d, dtype=np.float32)  # Initialization of M with identity matrix with shape (d,d)
        self.b = np.zeros((d, 1), dtype=np.float32)  # Initialization of b with shape (d,1) and filling with zeros

    # 1. Select arm with respect to current theta_hat (If required we delete extra_dimensions)
    # 2. Add selected arm to chosen_arms
    # 3. Add associated reward to observed_rewards
    def choose_arm(self, lnr_bandits):
        # 1.
        context_vectors = lnr_bandits.generate_context()
        aa = np.delete(context_vectors, self.extra_dimensions, 1)
        selected_arm = context_vectors[np.argmax(np.dot(aa, self.theta_hat))]
        reward, regret = lnr_bandits.pull(selected_arm)
        # 2.
        self.chosen_arms = np.vstack((self.chosen_arms, selected_arm))
        # 3.
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

    # 1. Return lasso estimator with respect to observed data up to time step t
    def lasso(self, lambda_t):
        lasso = Lasso(alpha=lambda_t)
        lasso.fit(self.chosen_arms, self.observed_rewards)

        return lasso.coef_
