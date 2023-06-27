import numpy as np

class linucb:
    def __init__(self, environment, sim_num, T, K, d, N):
        self.K = K
        self.d = d
        self.N = N
        self.env = environment
        self.sim_num = sim_num
        self.T = T

    def run(self):
        # An array for saving all cumulative regret
        cumulative_regret = [np.zeros(self.sim_num) for _ in range(self.T)]

        # Iterate each simulation
        for sim in range(self.sim_num):
            print(f'Iteration: {sim}.')

            self.env.reset()

            # Iterate time steps
            for t in range(self.T):
                cumulative_regret[t][sim] += self.env.run_linucb(t)

            for i in range(1, self.N):
                self.env.agents[0].cumulative_regret += self.env.agents[i].cumulative_regret

        regret_mean = [np.mean(cumulative_regret[t]) for t in range(self.T)]
        regret_std = [np.std(cumulative_regret[t]) for t in range(self.T)]

        return regret_mean, regret_std
