import numpy as np


class LinearBandits:
    def __init__(self, d, M0, theta_star, sigma, sA):
        self.d = d
        self.sA = sA  # Maximum infinity-norm of each context vector
        self.sigma = sigma  # The sub-gaussian parameter of noise
        self.K = len(M0)  # Number of arms
        self.theta_star = theta_star  # Sparse parameter of the linear model
        self.M0 = M0
        self.context_vectors = None

    def generate_context(self):
        self.context_vectors = np.random.multivariate_normal(np.zeros(self.K), self.M0, self.d).T
        inf_norms = np.max(np.abs(self.context_vectors), axis=1)
        self.context_vectors[inf_norms > self.sA] = self.context_vectors[inf_norms > self.sA] * self.sA / inf_norms[
                                                                                                              inf_norms > self.sA][
                                                                                                          :, None]
        return self.context_vectors

    def generate_explore_context(self, sn_idx, m):
        self.context_vectors = self.generate_context()
        explore_context = self.context_vectors.copy()
        for i in range(0, self.context_vectors.shape[0]):
            for j in range(0, self.context_vectors.shape[1]):
                if j < sn_idx * m:
                    explore_context[i][j] = 0
                elif j >= (sn_idx + 1) * m:
                    explore_context[i][j] = 0
        return explore_context

    def pull(self, context):
        reward = np.dot(context, self.theta_star)
        regret = np.max(np.dot(self.context_vectors, self.theta_star)) - reward
        return reward + np.random.normal(0, self.sigma), regret
