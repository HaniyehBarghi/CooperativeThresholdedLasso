from agent import Agent
import math
import numpy as np
import networkx as nx
import random


class Environment:
    def __init__(self, d, K, N, lnr_bandits, lam0, mode, par, cen):
        self.d = d
        self.K = K
        self.N = N
        self.lam0 = lam0
        self.lnr_bandits = lnr_bandits
        self.mode = mode
        self.par = par
        self.cen = cen

        self.log_counter = 1
        self.agents = [Agent(self.d, []) for _ in range(N)]

    def reset(self):
        self.log_counter = 1
        if not self.cen:
            neighbour_list = self.connected_graph(random.randint(self.N, 2 * self.N))
            self.agents = [Agent(self.d, neighbour_list[i]) for i in range(self.N)]
        else:
            self.agents = [Agent(self.d, []) for _ in range(self.N)]

    def run_step(self, t):
        if t % 200 == 0:
            print(f'\tTime step: {t}')

        regret = 0.0

        # Learning the model via ridge model
        for agent in self.agents:
            agent.choose_arm(self.lnr_bandits)
            regret += agent.cumulative_regret[-1] / self.N
            agent.update_parameters()

        # Specifying the mode of algorithm (Lasso or ridge)
        Upd_SS = False
        if self.mode == 'log':
            if math.pow(self.par, self.log_counter) <= t < math.pow(self.par, self.log_counter + 1):
                # print(set(np.arange(self.d)) - set(self.agents[0].extra_dimensions))
                Upd_SS = True
                self.log_counter += 1
        elif self.mode == 'cons':
            if t % self.par == 0:
                # print(set(np.arange(self.d)) - set(self.agents[0].extra_dimensions))
                Upd_SS = True
        else:
            # print(set(np.arange(self.d)) - set(self.agents[0].extra_dimensions))
            Upd_SS = True

        # Applying Lasso to update estimation of the support set
        if Upd_SS:
            lambda_t = self.lam0 * math.sqrt(
                (2 * math.log(t + 1) * math.log(self.d)) / (t + 1))  # Regularization-parameter

            # Calculating Lasso estimator for each agent
            lasso_estimators = []
            for agent in self.agents:
                lasso_estimators.append(agent.lasso(lambda_t))

            if self.cen:
                # Feature extraction by union
                support_hat = set()  # A set to save non-zero dimensions
                for estimator in lasso_estimators:
                    for j in range(self.d):
                        if math.fabs(estimator[j]) >= lambda_t * self.N * 0.1:
                            support_hat.add(j)

                # 1. Updating extra dimensions
                # 2. Updating ridge parameters
                extra_dimensions = list(np.delete([i for i in range(self.d)], list(support_hat), 0))
                if extra_dimensions != self.agents[0].extra_dimensions:
                    # 1. and 2.
                    for agent in self.agents:
                        agent.extra_dimensions = extra_dimensions
                        agent.dimension_reduction_parameters()
            else:
                for i in range(self.N):
                    neighbour = random.choice(self.agents[i].neighbours)

                    # Feature extraction by union
                    support_hat = set()  # A set to save non-zero dimensions
                    for j in range(self.d):
                        if math.fabs(lasso_estimators[i][j]) >= lambda_t * self.N * 0.1:
                            support_hat.add(j)
                        if math.fabs(lasso_estimators[neighbour][j]) >= lambda_t * self.N * 0.1:
                            support_hat.add(j)
                    self.agents[i].extra_dimensions = list(np.delete([i for i in range(self.d)], list(support_hat), 0))
                    self.agents[i].dimension_reduction_parameters()

        return regret

    def run_linucb(self, t):
        if t % 200 == 0:
            print(f'\tTime step: {t}')

        regret = 0
        # Learning the model via ridge model
        for agent in self.agents:
            agent.choose_arm(self.lnr_bandits)
            regret += agent.cumulative_regret[-1] / self.N
            agent.update_parameters()

        # Specifying the mode of algorithm (Lasso or ridge)
        Upd_SS = False
        if math.pow(2, self.log_counter) <= t < math.pow(2, self.log_counter + 1):
            Upd_SS = True
            self.log_counter += 1

        if Upd_SS:
            theta_hat_uni = np.zeros(self.agents[0].theta_hat.shape)
            for agent in self.agents:
                theta_hat_uni += agent.theta_hat

            theta_hat_uni = theta_hat_uni / self.N
            for agent in self.agents:
                agent.theta_hat = theta_hat_uni

        return regret

    def connected_graph(self, num_edges):
        # Check if num_edges is less than num_nodes - 1
        if num_edges < self.N - 1:
            raise ValueError("Number of edges should be at least one less than the number of nodes")

        # Create a graph with num_nodes nodes
        G = nx.gnm_random_graph(self.N, num_edges)

        # Add edges until the graph is connected
        while not nx.is_connected(G):
            G = nx.gnm_random_graph(self.N, num_edges)

        # Get the adjacency matrix of the graph
        adj_matrix = nx.to_numpy_array(G)

        neighbor_lists = []
        for i in range(self.N):
            neighbors = [j for j in range(self.N) if adj_matrix[i, j] == 1]
            neighbor_lists.append(neighbors)
        # Return the neighbor lists
        return neighbor_lists
