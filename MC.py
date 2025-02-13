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
    
class MCA_node:
    def __init__(self, id, center, radius):
        self.id = id
        self.center = center

class MCA_vertex_type:
    BOTH = 0
    FORWARD = 1

class MCA_vertex:
    def __init__(self, id_node1, id_node2, vertex_type):
        self.id_node1 = id_node1
        self.id_node2 = id_node2
        self.type = vertex_type

class Markov_Chain_Anim:
    def __init__(self, MC, center_coordinates=None, global_radius=None):
        self.P = MC.P
        self.n = MC.n
        self.ini = MC.ini
        self.global_radius = global_radius if global_radius is not None else 1/(2*self.n)
        
        self.nodes = [MCA_node(i, cc) for i, cc in zip(range(self.n), center_coordinates)]
        if center_coordinates is None:
            self.nodes = [MCA_node(i, (np.cos(2 * np.pi * i / self.n), np.sin(2 * np.pi * i / self.n))) for i in range(self.n)]
        else:
            self.nodes = [MCA_node(i, cc) for i, cc in zip(range(self.n), center_coordinates)]
        
        self.vertices = []
        self._setup_vertices()
    
    def _setup_vertices(self):
        for i in range(self.n):
            for j in range(i, self.n):
                vertex = None
                if self.P[i, j] > 0 and self.P[j, i] > 0:
                    vertex = MCA_vertex(i, j, MCA_vertex_type.BOTH)
                elif self.P[i, j] > 0:
                    vertex = MCA_vertex(i, j, MCA_vertex_type.FORWARD)
                elif self.P[j, i] > 0:
                    vertex = MCA_vertex(j, i, MCA_vertex_type.FORWARD)
                if vertex is not None:
                    self.vertices.append(vertex)
                