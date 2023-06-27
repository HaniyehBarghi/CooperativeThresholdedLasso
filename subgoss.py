import numpy as np
from agent_ld import agent_ld
import random


class subgoss:
    def __init__(self, num_j, num_sim, Ad_matrix, b, N, sn, m, d, K, lnr_bandits):
        self.num_j = num_j  # number of phases
        self.num_sim = num_sim  # number of simulations
        self.Ad_matrix = Ad_matrix  # communication matrix
        self.N = N  # number of agents
        self.sn = sn
        self.m = m
        self.d = d
        self.K = K
        self.lnr_bandits = lnr_bandits

        self.b = b
        self.T = 0
        for i in range(0, self.num_j):
            self.T += int(np.ceil(self.b ** i))

        self.agents = [agent_ld(i, self.Ad_matrix[i], self.sn,
                                self.K, self.m, self.d, self.N, self.T) for i in range(self.N)]

    def reset(self):
        self.agents = [agent_ld(i, self.Ad_matrix[i], self.sn,
                                self.K, self.m, self.d, self.N, self.T) for i in range(self.N)]

    def run(self):
        # An array for saving all cumulative regret
        cumulative_regret = []

        # Iterate each simulation
        for sim in range(self.num_sim):
            print(f'Iteration: {sim}.')
            self.reset()
            current_t = np.zeros(self.N)
            for j in range(0, self.num_j):
                for i in range(0, self.N):
                    for t in range(0, int(np.ceil(self.b ** j))):
                        k_set = len(self.agents[i].s_set)
                        current_t[i] += 1
                        explore_steps = k_set * self.m * np.ceil(self.b ** (0.5 * (j - 1)))
                        explore_per_subspace = self.m * np.ceil(self.b ** (0.5 * (j - 1)))
                        if t < explore_steps:
                            sn = np.floor(t / explore_per_subspace)
                            sn_idx = list(self.agents[i].s_set)[int(sn)]
                            self.agents[i].explore(self.lnr_bandits, sn_idx)
                            self.agents[i].update_o()
                        else:
                            self.agents[i].project_ucb(self.lnr_bandits)

                for i in range(0, self.N):
                    neighbor_indicator = self.Ad_matrix[i]
                    neighbors = np.asarray(np.where(neighbor_indicator > 0))
                    neighbor_set = list(neighbors.flatten())
                    ag = random.choice(neighbor_set)
                    o_ag = self.agents[ag].get_o()
                    self.agents[i].update_s_set(o_ag)

            for i in range(1, self.N):
                self.agents[0].cumulative_regret += self.agents[i].cumulative_regret

            cumulative_regret.append(self.agents[0].cumulative_regret / self.N)

        cumulative_regret = np.transpose(cumulative_regret)
        print(np.shape(cumulative_regret))
        regret_mean = [np.mean(cumulative_regret[t]) for t in range(self.T)]
        regret_std = [np.std(cumulative_regret[t]) for t in range(self.T)]

        return regret_mean, regret_std
