import numpy as np

class FederatedThLassoBandit:
    def __init__(self, environment, num_sim, T, K, d, N):
        self.K = K
        self.d = d
        self.N = N
        self.env = environment
        self.num_sim = num_sim
        self.T = T

    def run(self):
        # An array for saving all cumulative regret
        cumulative_regret = []

        # Iterate each simulation
        for sim in range(self.num_sim):
            print(f'Iteration: {sim}.')

            self.env.reset()

            # Iterate time steps
            for t in range(self.T):
                self.env.run_step(t)

            for i in range(1, self.N):
                self.env.agents[0].cumulative_regret += self.env.agents[i].cumulative_regret

            cumulative_regret.append(self.env.agents[0].cumulative_regret / self.N)

        cumulative_regret = np.transpose(cumulative_regret)
        print(np.shape(cumulative_regret))
        regret_mean = [np.mean(cumulative_regret[t]) for t in range(self.T)]
        regret_std = [np.std(cumulative_regret[t]) for t in range(self.T)]

        return regret_mean, regret_std
