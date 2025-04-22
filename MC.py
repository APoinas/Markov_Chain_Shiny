import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch

class Markov_Chain:
    def __init__(self, P, ini=0):
        self.P = P
        self.n = P.shape[0]
        self.ini = ini
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
        self.radius = radius
        self.patch = Circle(center, radius, edgecolor="black", facecolor="white", lw=2)
    
    def __eq__(self, other):
        if type(other) is MCA_node:
            return self.id == other.id
        return False
    
    def __repr__(self):
        return f'Node id {self.id} with center {self.center} and radius {self.radius}'

class MCA_vertex:
    def __init__(self, node0, node1, weights):
        self.node0 = node0
        self.node1 = node1
        self.weights = weights
        if node1 is None:
            self.angle = np.pi/2
        else:
            self.angle = np.angle(self.node1.center[0] - self.node0.center[0] + (self.node1.center[1] - self.node0.center[1]) * 1j)
        self.patches = []
        self.texts = []
        self._setup_patches_and_texts()
    
    def __repr__(self):
        if self.node1 is None:
            return f'Vertex {self.node0.id}<-->{self.node0.id} with weight {self.weights} and angle {self.angle}'
        return f'Vertex {self.node0.id}<-->{self.node1.id} with weights {self.weights} and angle {self.angle}'
    
    def _setup_patches_and_texts(self):
        center0 = np.array(self.node0.center)
        ang = self.angle
        mov1 = np.array([np.cos(ang - 0.3), np.sin(ang - 0.3)])
        mov2 = np.array([np.cos(ang + 0.3), np.sin(ang + 0.3)])
        if self.node1 is not None:
            center1 = np.array(self.node1.center)
            center = (center0 + center1)/2
            ang_arr = 0
            rad_text = 0.3
            
            self.patches.append(FancyArrowPatch(center0 + self.node0.radius * mov1, center1 - self.node1.radius * mov2, 
                                                connectionstyle="arc3, rad=" + str(ang_arr), arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="black"))
            self.texts.append({"loc": center + rad_text * (mov1 - mov2)/2, "lab": str(self.weights[0]), "rot": ang * 180 / np.pi})
            if self.weights[1] != 0:
                self.patches.append(FancyArrowPatch(center1 - self.node1.radius * mov1, center0 + self.node0.radius * mov2, 
                                                    connectionstyle="arc3, rad=" + str(ang_arr), arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="black"))
                self.texts.append({"loc": center - rad_text * (mov1 - mov2)/2, "lab": str(self.weights[1]), "rot": ang * 180 / np.pi})
        else:
            mov1 = np.array([np.cos(ang - 0.5), np.sin(ang - 0.5)])
            mov2 = np.array([np.cos(ang + 0.5), np.sin(ang + 0.5)])
            ang_arr = 2
            rad_text = 0.5
            self.patches.append(FancyArrowPatch(center0 + self.node0.radius * mov1, center0 + self.node0.radius * mov2, 
                                                connectionstyle="arc3, rad=" + str(ang_arr), arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="black"))
            self.texts.append({"loc": center0 - rad_text * (mov1 - mov2)/2, "lab": str(self.weights), "rot": ang * 180 / np.pi})
    
    def update_solo_angle(self, new_angle):
        if self.node1 is None:
            self.angle = new_angle
            rad_text = 0.25
            mov1 = np.array([np.cos(new_angle - 0.5), np.sin(new_angle - 0.5)])
            mov2 = np.array([np.cos(new_angle + 0.5), np.sin(new_angle + 0.5)])
            self.patches[0].set_positions(np.array(self.node0.center) + self.node0.radius * mov1, np.array(self.node0.center) + self.node0.radius * mov2)
            self.texts[0]["loc"] = np.array(self.node0.center) + rad_text * np.array([np.cos(new_angle), np.sin(new_angle)])
            self.texts[0]["rot"] = new_angle * 180 / np.pi - 90

class Markov_Chain_Anim:
    def __init__(self, MC, center_coordinates=None, global_radius=None):
        self.P = MC.P
        self.n = MC.n
        self.ini = MC.ini
        self.previous_state = None
        self.state = MC.ini
        self.global_radius = global_radius if global_radius is not None else 1/(2*self.n)
        
        if center_coordinates is None:
            self.nodes = [MCA_node(i, (np.cos(2 * np.pi * i / self.n), np.sin(2 * np.pi * i / self.n)), self.global_radius) for i in range(self.n)]
        else:
            self.nodes = [MCA_node(i, cc, self.global_radius) for i, cc in zip(range(self.n), center_coordinates)]
        
        self.vertices = []
        self._setup_vertices()
    
    def _setup_vertices(self):
        for i in range(self.n):
            for j in range(i, self.n):
                if i == j and self.P[i, i] > 0:
                    self.vertices.append(MCA_vertex(self.nodes[i], None, self.P[i, i]))
                elif self.P[i, j] > 0:
                    self.vertices.append(MCA_vertex(self.nodes[i], self.nodes[j], (self.P[i, j], self.P[j, i])))
                elif self.P[j, i] > 0:
                    self.vertices.append(MCA_vertex(self.nodes[j], self.nodes[i], (self.P[j, i], 0)))
        
        for vertex in self.vertices:
            if vertex.node1 is None:
                L = np.sort(np.mod([vert.angle if vert.node0==vertex.node0 else vert.angle - np.pi for vert in self.connected_vertices(vertex.node0, self_=False)], 2 * np.pi))
                vertex.update_solo_angle(Best_Angle(L))
    
    def connected_vertices(self, node, self_=True):
        connexions = []
        for vertex in self.vertices:
            if vertex.node1 == node:
                connexions.append(vertex)
            elif vertex.node0 == node:
                if vertex.node1 is not None:
                    connexions.append(vertex)
                elif self_:
                    connexions.append(vertex)
        return connexions
    
    def update_state(self):
        old = self.state
        self.state = np.random.choice(self.n, 1, p=self.P[self.state, :])[0]
        self.previous_state = old
    
    def find_vertex(self, node0, node1=None):
        if node1 is None or node0 == node1:
            for vertex in self.vertices:
                if node0 == vertex.node0:
                    return vertex
        else:
            for vertex in self.vertices:
                if node0 == vertex.node0 and node1 == vertex.node1:
                    return vertex
                if node0 == vertex.node1 and node1 == vertex.node0:
                    return vertex
        return None
    
    def find_vertex_patch(self, node0, node1=None):
        if node1 is None or node0 == node1:
            for vertex in self.vertices:
                if node0 == vertex.node0 and vertex.node1 is None:
                    return vertex.patches[0]
        else:
            for vertex in self.vertices:
                if node0 == vertex.node0 and node1 == vertex.node1:
                    return vertex.patches[0]
                if node0 == vertex.node1 and node1 == vertex.node0:
                    return vertex.patches[1]
        return None
    
    def draw(self, fig=None):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.gca()
            Clear_all(ax)
        magnifying_factor = 1.1
        offset = 0.2
        xmin = min([node.center[0] - magnifying_factor * node.radius for node in self.nodes])
        xmax = max([node.center[0] + magnifying_factor * node.radius for node in self.nodes])
        ymin = min([node.center[1] - magnifying_factor * node.radius for node in self.nodes])
        ymax = max([node.center[1] + magnifying_factor * node.radius for node in self.nodes])
        ax.set(xlim=[xmin - offset, xmax + offset], ylim=[ymin - offset, ymax + offset], aspect="equal")
        ax.set_axis_off()

        for circ in self.nodes:
            ax.add_patch(circ.patch)
        
        for vert in self.vertices:
            for patch in vert.patches:
                ax.add_patch(patch)
            for text in vert.texts:
                rot = text["rot"]
                if rot < -90:
                    rot += 180
                if rot > 90:
                    rot -= 180
                ax.text(text["loc"][0], text["loc"][1], text["lab"], rotation=rot, ha="center", va="center")
                
        return fig, ax
    
    def Animate(self):
        fig, _ = self.draw()
        nb_frame = 30
        
        def update(frame):
            if frame == 0:
                if self.previous_state is not None:
                    self.find_vertex_patch(self.nodes[self.previous_state], self.nodes[self.state]).set_color("black")
                    self.nodes[self.previous_state].patch.set_edgecolor("black")
                self.nodes[self.state].patch.set_edgecolor("red")
            elif frame % 2 == 1:
                self.nodes[self.state].patch.set_edgecolor("black")
                self.update_state()
                self.find_vertex_patch(self.nodes[self.previous_state], self.nodes[self.state]).set_color("red")
            else:
                self.find_vertex_patch(self.nodes[self.previous_state], self.nodes[self.state]).set_color("black")
                self.nodes[self.state].patch.set_edgecolor("red")
        
        return animation.FuncAnimation(fig=fig, func=update, frames=nb_frame, interval=240)

def Best_Angle(L):
    if len(L) == 0:
        return 0
    if len(L) == 1:
        return np.mod(L[0] + np.pi, 2 * np.pi)
    if len(L) >= 2:
        best_id = np.argmax(np.diff(L, append = L[0] + 2 * np.pi))
        if best_id == len(L) - 1:
            return np.mod((L[best_id] + L[0] + 2 * np.pi)/2, 2 * np.pi)
        return (L[best_id] + L[best_id + 1])/2

def Clear_patches(ax):
    for p in ax.patches:
        p.set_visible(False)
        p.remove()

def Clear_texts(ax):
    for p in ax.texts:
        p.set_visible(False)
        p.remove()

def Clear_all(ax=None):
    if ax is None:
        Clear_patches(ax)
        Clear_texts(ax)
    else:
        try:
            Clear_patches(plt.gca())
        except:
            pass
        try:
            Clear_texts(plt.gca())
        except:
            pass