import numpy as np
from agent import Agent
import random


class CTL:
    def __init__(self, T, N, d, sim_num, env):
        self.T = T  # Time horizon
        self.N = N  # Number of agents
        self.d = d  # Dimension
        self.sim_num = sim_num
        self.env = env

        # Defining agents [sim_num * N]
        self.agents = []
        for i in range(self.sim_num):
            if self.env.cen:
                self.agents = [Agent(self.d, []) for _ in range(N)]
            else:
                neighbour_list = self.env.connected_graph(random.randint(self.N, 2 * self.N))
                self.agents = [Agent(self.d, neighbour_list[i]) for i in range(self.N)]

    def run_algorithm(self):
        # An array for saving all cumulative regret
        cumulative_regret = [np.zeros(self.sim_num) for _ in range(self.T)]

        # Iterate each simulation
        for sim in range(self.sim_num):
            print(f'Iteration: {sim}.')
            self.env.reset()

            # Iterate time steps
            for t in range(self.T):
                cumulative_regret[t][sim] += self.env.run_step(t)

        regret_mean = [np.mean(cumulative_regret[t]) for t in range(self.T)]
        regret_std = [np.std(cumulative_regret[t]) for t in range(self.T)]

        return regret_mean, regret_std
