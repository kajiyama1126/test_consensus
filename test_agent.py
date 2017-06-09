import copy

import numpy as np


class Agent(object):
    def __init__(self, n, name, weight):
        # self.x_i = random.random(-1,1)
        self.n = n
        self.z_i = np.zeros(self.n)
        self.z_i[name] = 1
        self.z_j = np.zeros((self.n, self.n))
        self.weight = weight
        self.name = name
        self.tildez_ij = np.zeros((self.n, self.n))
        self.tildez_ji = np.zeros((self.n, self.n))
        self.trigger = np.ones(self.n)

    def send(self,j):
        if self.trigger[j] == 1:
            self.tildez_ij[j] = self.z_i
            return (self.z_i)
        else:
            return None

    def receive(self, z, j):
        if z == None:
            pass
        else:
            self.tildez_ji[j] = z[0]


    def update(self):
        sum = 0
        for i in range(self.n):
            if self.name != i:
                sum += self.weight[i] * (self.tildez_ji[i] - self.tildez_ij[i])
        self.z_i = self.z_i + sum
        self.trigger_check(k)

    def trigger_check(self,k):
        for i in range(self.n):
            if np.linalg.norm(self.tildez_ij[i] - self.z_i) >= self.E(k):
                self.trigger[i] = 1
            else:
                self.trigger[i] = 0

    def E(self,k):
        return 0.99**k



class Agent_opt(object):
    def __init__(self, n, name, weight, eta):
        # self.x_i = random.random(-1,1)
        self.n = n
        self.z_i = np.zeros(self.n)
        self.z_i[name] = 1
        self.x_i = copy.copy(self.z_i)
        self.A_i = np.identity(n)
        self.x_j = np.zeros((self.n, self.n))
        self.tildex_ji = np.zeros((self.n, self.n))
        self.tildex_ij = np.zeros((self.n, self.n))
        self.weight = weight
        self.name = name
        self.s_i = self.d_i()
        self.s_j = np.zeros((self.n, self.n))
        self.tildes_ij = np.zeros((self.n, self.n))
        self.tildes_ji = np.zeros((self.n, self.n))
        self.eta = eta
        self.trigger = np.ones(self.n)

    def d_i(self):
        return -(np.dot(self.A_i, self.x_i) - self.z_i)

    def send(self, j):
        if self.trigger[j] == 1:
            self.tildex_ij[j] = self.x_i
            self.tildes_ij[j] = self.s_i
            return (self.x_i, self.s_i)
        else:
            return None

    def receive(self, z, j):
        if z == None:
            pass
        else:
            self.tildex_ji[j] = z[0]
            self.tildes_ji[j] = z[1]

    def update(self,k):
        # self.x_ji[self.name] = self.x_i
        sum = 0
        for i in range(self.n):
            if self.name != i:
                sum += self.weight[i] * (self.tildex_ji[i] - self.tildex_ij[i])
        d_bf = self.d_i()
        # print(self.weight, self.z_j)
        self.x_i = self.x_i + sum - self.eta * self.s_i
        self.s_update(d_bf)
        self.trigger_check(k)

    def trigger_check(self,k):
        for i in range(self.n):
            if np.linalg.norm(self.tildex_ij[i] - self.x_i) >= self.E(k):
                self.trigger[i] = 1
            else:
                self.trigger[i] = 0

    def E(self,k):
        return 0.9**k

    def s_update(self, d_bf):
        # self.s_j[self.name] = self.s_i
        sum = 0
        for i in range(self.n):
            if self.name != i:
                sum += self.weight[i] * (self.tildes_ji[i] - self.tildes_ij[i])
        self.s_i = self.s_i + sum + self.d_i() - d_bf


n = 4
# weight = np.array([[0.5, 0, 0.25, 0.25],
#                    [0, 0.5, 0.25, 0.25],
#                    [0.25, 0.25, 0.5, 0],
#                    [0.25, 0.25, 0, 0.5]])
weight = np.array([[0.5, 0, 0.25, 0.25],
                   [0, 0.5, 0.25, 0.25],
                   [0.5, 0.25, 0.25, 0],
                   [0.5, 0.25, 0, 0.25]])

all_agent = [Agent(n, i, weight[i]) for i in range(n)]
all_agent_opt = [Agent_opt(n, i, weight[i], 0.05) for i in range(n)]
for k in range(200):
    for i in range(n):
        for j in range(n):
            all_agent[i].receive(all_agent[j].send(i), j)
            all_agent_opt[i].receive(all_agent_opt[j].send(i), j)

    for i in range(n):
        all_agent[i].update()
        all_agent_opt[i].update(k)
    for i in range(n):
        print(all_agent[i].z_i, all_agent_opt[i].x_i)
