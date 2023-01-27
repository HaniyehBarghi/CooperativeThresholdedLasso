####################################################################################################################
##### This code implements the linear bandits to generate a matrix of size [K*d] in each time step as contexts #####
##### vectors and return the linear reward and corresponding regret after choosing a context.                  #####
####################################################################################################################

import numpy as np

class LinearBandits:
    def __init__(self, d, M0, theta_star, sigma, sA):
        self.d = d      # Dimension of each context_vector
        self.sA = sA    # Maximum infinity-norm of each context vector
        self.sigma = sigma      # The sub-gaussian parameter of noise
        self.K = len(M0)    # Number of arms
        self.theta_star = theta_star    # Main linear parameter of the model
        self.M0 = M0    
        self.context_vectors = None
        
    def generate_context(self):
        self.context_vectors = np.random.multivariate_normal(np.zeros(self.K), self.M0, self.d).T
        inf_norms = np.max(np.abs(self.context_vectors), axis=1)
        self.context_vectors[inf_norms > self.sA] = self.context_vectors[inf_norms > self.sA] * self.sA / inf_norms[inf_norms > self.sA][:, None]
        return self.context_vectors

    def pull(self, context):
        reward = np.dot(self.x[context], self.beta)
        regret = np.max(np.dot(self.x, self.beta)) - reward
        return reward + np.random.normal(0, self.sigma), regret
