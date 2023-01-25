import numpy as np
from sklearn import linear_model


class Agent:
    def __init__(self, d, K, true_Sigma, theta_star, sigma):
        self.d = d  # observed data's number of dimensions
        self.K = K  # number of arms/actions
        self.true_Sigma = true_Sigma  # Covariance matrix to sample context vectors, which is same for all agents
        self.sigma = sigma  # The sub-gaussian parameter of noise
        self.theta_star = theta_star  # Main linear parameter of the model
        self.theta_hat = np.ones(d)  # Estimated linear parameters of the model at time t
        self.A_train = []  # Chosen context-vectors up to time t
        self.Y_train = []  # Observed rewards up to time t
        self.cumulative_regret = [0.0]  # Cumulative regret up to time t
        self.context_vectors = []  # Observed context-vectors at time t
        self.extra_dimensions = []  # non-main dimensions at time t
        # Ridge case:
        self.M = np.identity(d)  # Ridge estimator parameter M at time t
        self.b = np.zeros([d, 1])  # Ridge estimator parameter b at time t

    # 1. Set the matrix of observed context-vectors K * d once at begging of every time step t
    def observe(self):
        self.context_vectors = np.random.multivariate_normal(np.zeros(self.K), self.true_Sigma, self.d).T

    # 1. Select arm maximizing linear reward with respect to current theta_hat
    # If len(extra_dimensions) > 0, we do not consider the dimensions in self.extra_dimensions
    # 2. Add selected arm to A_train
    # 3. Add associated reward to Y_train
    def pull(self):
        # 1.
        estimated_rewards = []
        for context_vector in self.context_vectors:
            temp = context_vector
            temp = np.delete(temp, self.extra_dimensions, 0)
            temp = temp.reshape([-1, 1])
            estimated_rewards.append(np.matmul(self.theta_hat.T, temp))
        selected_context_vector = self.context_vectors[estimated_rewards.index(max(estimated_rewards))]
        # 2.
        self.A_train.append(selected_context_vector)
        # 3.
        self.Y_train.append(np.matmul(self.theta_star.T, selected_context_vector) + np.random.normal(0, self.sigma))

    # 1. Set regret at time step t
    def calculate_regret(self):
        self.cumulative_regret.append(float(self.cumulative_regret[-1] + self.best() - self.Y_train[-1]))

    # 1. Return best reward
    def best(self):
        best_reward = []
        for context_vector in self.context_vectors:
            if (context_vector == self.A_train[-1]).all():
                best_reward.append(self.Y_train[-1])
            else:
                best_reward.append(np.matmul(self.theta_star.T, context_vector) + np.random.normal(0, self.sigma))
        return max(best_reward)

    # 1. Update ridge parameters M, b
    # 2. Update theta_hat
    def update_ridge_parameters(self):
        # 1.
        aa = self.A_train
        aa = list(np.delete(aa, self.extra_dimensions, 1))
        self.M = np.identity(self.d - len(self.extra_dimensions))
        self.b = np.zeros([self.d - len(self.extra_dimensions), 1])
        for i in range(len(aa)):
            temp = aa[i].reshape([-1, 1])
            self.M += np.dot(temp, temp.T)
            self.b += self.Y_train[i] * aa[i].reshape([-1, 1])
        # 2.
        self.theta_hat = np.ravel(np.matmul(np.linalg.inv(self.M), self.b))

    # 1. Return lasso estimator with respect to observed data up to time step t
    def lasso(self, lambda_t):
        yy = np.ravel(self.Y_train)
        lasso = linear_model.Lasso(alpha=lambda_t)
        lasso.fit(self.A_train, yy)
        self.theta_hat = lasso.coef_

        return self.theta_hat
