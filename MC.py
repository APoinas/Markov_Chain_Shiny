import numpy as np

class Markov_Chain:
    def __init__(self, P, ini=None):
        self.P = P
        self.n = P.shape[0]
        if ini is None:
            ini = 0
        self.step = 0
        self.max_step = 0
        self.dist_hist = np.zeros((1, self.n))
        self.dist_hist[0, ini] = 1
        self._infinity = None
    
    def forward(self, k=1):
        if k >= 1:
            if self.step == self.max_step:
                self.max_step += 1
                self.dist_hist = np.vstack((self.dist_hist, (self.dist[None, :]).dot(self.P)))
            self.step += 1
            if k >= 2:
                self.forward(k-1)
    
    def backward(self, k=1):
        #if k > self.step:
        #    raise ValueError("Going too far backward.")
        self.step -= k
        self.step = max(0, self.step)
    
    @property
    def dist(self):
        return self.dist_hist[self.step, :]
    
    @property
    def infinity(self):
        if self._infinity is None:
            _, eigval = np.linalg.eig((self.P).T)
            v = eigval[:, 0]
            self._infinity = abs(v / sum(v))
        return self._infinity
            