# Method of Kim and Paik (2019). Doubly-Robust Lasso Bandit.

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import sparse
import random

class DRAgent:
    def __init__(self, lambda_1, lambda_2, d, K, tc, tr, zt):
        self.x = []
        self.r = []
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.d = d
        self.K = K
        self.theta_hat = np.zeros(d)
        self.tc = tc
        self.tr = tr
        self.zt = zt

    def choose_a(self, t, x):  # x is K*d matrix
        if t < self.zt:
            self.action = np.random.choice(range(self.K))
            self.pi = 1. / self.K
        else:
            uniformp = self.lambda_1 * np.sqrt((np.log(t) + np.log(self.d)) / t)
            uniformp = np.minimum(1.0, np.maximum(0., uniformp))
            choice = np.random.choice([0, 1], p=[1. - uniformp, uniformp])
            est = np.dot(x, self.theta_hat)
            if choice == 1:
                self.action = np.random.choice(range(self.K))
                if self.action == np.argmax(est):
                    self.pi = uniformp / self.K + (1. - uniformp)
                else:
                    self.pi = uniformp / self.K
            else:
                self.action = np.argmax(est)
                self.pi = uniformp / self.K + (1. - uniformp)
        self.x.append(np.mean(x, axis=0))
        self.rhat = np.dot(x, self.theta_hat)
        return (self.action)

    def update_beta(self, rwd, t):
        pseudo_r = np.mean(self.rhat) + (rwd - self.rhat[self.action]) / self.pi / self.K
        if self.tr:
            pseudo_r = np.minimum(3., np.maximum(-3., pseudo_r))
        self.r.append(pseudo_r)
        if t > 5:
            if t > self.tc:
                lam2_t = self.lambda_2 * np.sqrt((np.log(t) + np.log(self.d)) / t)
            lasso = linear_model.Lasso(alpha=lam2_t)
            lasso.fit(self.x, self.r)
            self.theta_hat = lasso.coef_


class DRL:
    def __init__(self, T, K, M_0, theta_star, sim_num):
        self.T = T  # Time horizon
        self.K = K
        self.d = len(theta_star)  # Dimension
        self.sim_num = sim_num
        self.M_0 = M_0
        self.theta_star = theta_star

        # Defining agents [sim_num]
        self.agents = []
        for i in range(self.sim_num):
            self.agents.append(DRAgent(lambda_1=1., lambda_2=0.5, d=self.d, K=K, tc=1, tr=True, zt=10))

    def run_algorithm(self):
        cumulated_regret_DR = []

        for sim in range(self.sim_num):
            RWD3 = list()
            optRWD = list()
            for t in range(self.T):
                x = np.random.multivariate_normal(np.zeros(self.K), self.M_0, self.d).T
                a3 = self.agents[sim].choose_a(t + 1, x)
                rwd3 = np.dot(x[a3], self.theta_star) + np.random.normal(0, 0.05)
                RWD3.append(np.dot(x[a3], self.theta_star))
                self.agents[sim].update_beta(rwd3, t + 1)
                optRWD.append(np.amax(np.dot(x, self.theta_star)))

            cumulated_regret_DR.append(np.cumsum(optRWD) - np.cumsum(RWD3))

        cumulated_regret_DR = np.array(cumulated_regret_DR).T
        regret_mean = [np.mean(cumulated_regret_DR[t]) for t in range(self.T)]
        regret_std = [np.std(cumulated_regret_DR[t]) for t in range(self.T)]

        return regret_mean, regret_std
#lfdjlfdg