import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import networkx as nx

class SimplicialComplex:
    """
    Two dimensional simplicial complex
    """
    def __init__(self, num_side, delaunay=False, random=True, weighted=True) -> None:
        simps = self.get_simps(num_side, delaunay, random)
        self.nodes, self.edges, self.faces = simps
        self.num_nodes, self.num_edges, self.num_faces = [len(x) for x in simps]
        self.node_vec = np.zeros(self.num_nodes, dtype=int)
        self.edge_vec = np.zeros(self.num_edges, dtype=int)
        self.face_vec = np.zeros(self.num_faces, dtype=int)
        self.weighted = weighted
        self.B0_main, self.B1_main = self.make_boundaries()
        self.add_simplices({2:np.arange(self.num_faces)})
        self.update_nodes()

    @staticmethod
    def triangulate(num_side, delaunay, random):
        if random:
            pts = 2 * np.random.random((num_side ** 2, 2)) - 1
        else:
            x, y = np.linspace(-1, 1, num_side), np.linspace(-1, 1, num_side)
            xv, yv = np.meshgrid(x,y)
            xv, yv = xv.ravel(), yv.ravel()
            pts = np.array([(xv[i], yv[i]) for i in range(xv.shape[0])])
        
        if delaunay:
            tri = Delaunay(pts).simplices
        else:
            tri = []
            for j in range(num_side-1):
                for i in range(num_side-1):
                    idx, idxx = i * num_side, (i+1) * num_side
                    tl,tr,bl,br = idx+j, idx+j+1, idxx+j, idxx+j+1
                    if (j % 2) == (i % 2):
                        tri.append([tl, tr, bl])
                        tri.append([tr, bl, br])
                    else: 
                        tri.append([tl, tr, br])
                        tri.append([tl, bl, br])
        tri = [tuple(sorted(x)) for x in tri if len(set(x)) == 3]
        return pts, tri
            
    def get_simps(self, num_side, delaunay, random):
        nodes, faces = self.triangulate(num_side, delaunay, random)
        edge_idx_dict, edge_dict = {}, {}
        num_edge = 0

        for face in faces:
            n1, n2, n3 = face
            face_edges = [(n1, n2), (n1, n3), (n2, n3)]

            for edge in face_edges:
                if edge not in edge_dict:
                    edge_dict[edge] = num_edge
                    edge_idx_dict[num_edge] = edge
                    num_edge += 1

        edges = [edge_idx_dict[i] for i in range(num_edge)]

        edges.sort(key=lambda x:x[0])
        faces.sort(key=lambda x:x[0])

        return nodes, edges, faces

    def make_boundaries(self):
        B0 = np.zeros((self.num_nodes, self.num_edges))
        B1 = np.zeros((self.num_edges, self.num_faces))

        for i, edge in enumerate(self.edges):
            v1, v2 = edge
            B0[v1, i] = -1
            B0[v2, i] = 1
        
        for i, face in enumerate(self.faces):
            v1, v2, v3 = face
            e1, e2, e3 = (v1, v2), (v1, v3), (v2, v3)
            e1_idx, e2_idx, e3_idx = self.edges.index(e1), self.edges.index(e2), self.edges.index(e3)
            B1[e1_idx, i] = 1
            B1[e2_idx, i] = -1
            B1[e3_idx, i] = 1

        self.B0, self.B1 = B0, B1

        return B0, B1
    
    def add_simplices(self, simp_set):
        """
        simplices: dictionary of simplices to be added.
        dictionary keys can be 0, 1, and/or 2
        e.g. simplices[0] = [1, 3, 4] 
        (set containing 1st, 3rd and 4th node in mesh) 

        Want to add simplices in such a way that simplicial complex structure is 
        maintained. Therefore, if we add a face, we must ensure to add its 
        edges and nodes if they aren't already in the complex.
        """
        simp_dict = {0:set(), 1:set(), 2:set()}
        
        for i in range(3):
            if i in simp_set:
                simp_dict[i] = simp_dict[i].union(simp_set[i])

        # Add faces
        for face in simp_dict[2]:
            n1, n2, n3 = self.faces[face]
            self.face_vec[face] = 1

            e1, e2, e3 = (n1, n2), (n1, n3), (n2, n3)

            e1_idx, e2_idx, e3_idx = self.edges.index(e1), self.edges.index(e2), self.edges.index(e3)

            nodes_needed, edges_needed = set([n1, n2, n3]), set([e1_idx, e2_idx, e3_idx])

            simp_dict[0] = simp_dict[0].union(nodes_needed)
            simp_dict[1] = simp_dict[1].union(edges_needed)

        # Add edges
        for edge in simp_dict[1]:
            self.edge_vec[edge] = 1
            n1, n2 = self.edges[edge]

            nodes_needed = set([n1, n2])
            simp_dict[0] = simp_dict[0].union(nodes_needed)
            
        # Add nodes
        for node in simp_dict[0]:
            self.node_vec[node] = 1

        self.update_boundaries()
        self.update_G()
        self.update_H()
        self.update_nodes()

    def del_simplices(self, simp_set):
        simp_dict = {0:set(), 1:set(), 2:set()}
        
        for i in range(3):
            if i in simp_set:
                simp_dict[i] = simp_dict[i].union(simp_set[i])

        for node in simp_dict[0]:
            self.node_vec[node] = 0

            for i, edge in enumerate(self.edges):
                if node in edge:
                    simp_dict[1] = simp_dict[1].union(set([i]))
                
            for i, face in enumerate(self.faces):
                if node in face:
                    simp_dict[2] = simp_dict[2].union(set([i]))

        for edge in simp_dict[1]:
            self.edge_vec[edge] = 0
            n1, n2 = self.edges[edge]
            for i, face in enumerate(self.faces):
                if (n1 in face) and (n2 in face):
                    simp_dict[2] = simp_dict[2].union(set([i]))
    
        for face in simp_dict[2]:
            self.face_vec[face] = 0
        
        self.update_boundaries()
        self.update_G()
        self.update_H()
        self.update_nodes()

    def update_nodes(self):
        self.active_nodes = np.where(self.node_vec == 1)[0]

    def update_G(self, weights=None):
        G = nx.Graph()
        
        for i, edge in enumerate(self.edges):
            if self.edge_vec[i]:
                u, v = edge
                if weights is not None:
                    w = weights[i]
                else:
                    w = np.linalg.norm(self.nodes[u] - self.nodes[v]) if self.weighted else 1
                G.add_edge(u, v, weight=w, idx=i)
        
        self.G = G

    def update_boundaries(self):
        B0 = self.B0_main[self.node_vec.astype(bool),:]
        B0 = B0[:,self.edge_vec.astype(bool)]
        B1 = self.B1_main[self.edge_vec.astype(bool),:]
        B1 = B1[:,self.face_vec.astype(bool)]
        self.B0, self.B1 = B0, B1
        return B0, B1

    def update_H(self, thresh=1e-12):
        L = self.B0.T @ self.B0 + self.B1 @ self.B1.T

        eig_w, eig_v = np.linalg.eigh(L)
        ev_list = [(eig_w[i], eig_v[i]) for i in range(len(eig_w))]
        ev_list.sort(key=lambda var:var[0], reverse=False)
        eig_w, eig_v = np.array([tup[0] for tup in ev_list]), np.array([tup[1] for tup in ev_list])
        num_holes = np.sum(np.abs(eig_w) < thresh)
        H = eig_v[:,:num_holes].T
        self.H, self.num_holes = H, num_holes

    def num_simplices(self, k):
        if k == 0:
            res = np.sum(self.node_vec)
        elif k == 1:
            res = np.sum(self.edge_vec)
        else:
            res = np.sum(self.face_vec)
        return int(res)

    def edge_idx_parity(self, edge):
        e1, e2 = edge
        idx = self.edges.index((min(edge), max(edge)))
        parity = -2 * int(e1 > e2) + 1
        return idx, parity

    def make_holes(self, hole_locs, r=0.25):
        hole_nodes = set()
        
        for i, hole in enumerate(hole_locs):
            dist_from_nodes = np.linalg.norm(self.nodes - hole, axis=1, ord=1)
            hole_idx = set(np.where((dist_from_nodes < r) == True)[0])
            hole_nodes = hole_nodes.union(hole_idx)
        
        nodes_to_remove = {0:set(hole_nodes)}
        self.del_simplices(nodes_to_remove)


def get_polygon_order(pts):
    pts_arr = np.zeros((len(pts), 2))
    for i, pt in enumerate(pts):
        pts_arr[i,:] = pt
    pts_mean = np.mean(pts_arr, axis=0)
    pts_arr -= pts_mean
    polar_ang = np.arctan2(pts_arr[:,1], pts_arr[:,0])
    polar_mag = pts_arr[:,0]**2 + pts_arr[:,1]**2
    ind = np.lexsort((polar_mag, polar_ang))
    return pts[ind], polar_ang[ind], polar_mag[ind]

def poly_hole(SC, hole_pts):
    hole_nodes = set()
    active_nodes = SC.nodes[SC.node_vec == 1]
    active_nodes_idx = np.where(SC.node_vec == 1)[0]
    num_pts = len(hole_pts)
    edges = [(hole_pts[i%num_pts], hole_pts[(i+1)%num_pts]) for i in range(num_pts)]

    for (idx, (xp, yp)) in zip(active_nodes_idx, active_nodes):
        intersections = 0
        for ((x1, y1), (x2,y2)) in edges:
            if (y1 < yp) == (y2 < yp):
                continue
            if (y2 != y1) and (xp < x1 + ((yp - y1) / (y2-y1)) * (x2 - x1)):
                intersections += 1
        if (intersections % 2) == 1:
            hole_nodes.add(idx)
    
    return hole_nodes

def rect_hole(left, top, width, height):
    tl = (left, top)
    bl = (left, top - height)
    br = (left + width, top - height)
    tr = (left + width, top)
    return np.array([tl, bl, br, tr])

class Path:
    def __init__(self, SC, nodes, seq=False) -> None:
        self.SC = SC
        self.nodes = nodes
        self.node_coords = [SC.nodes[x] for x in nodes]
        self.edges = [tuple(x) for x in zip(nodes[:-1], nodes[1:])]
        self.to_chain(seq)
        self.project(seq)
        self.get_weight(seq)

    def to_chain(self, seq=False):
        num_edges, num_path_nodes = self.SC.edge_vec.shape[0], len(self.nodes)
        self.chain = np.zeros((num_edges, 1))
        self.chain_seq = np.zeros((num_edges, num_path_nodes)) if seq else None
        
        for i, edge in enumerate(self.edges):
            idx, parity = self.SC.edge_idx_parity(edge)
            self.chain[idx] += parity
            if seq:
                self.chain_seq[idx, i+1] = parity
        
        if seq:
            self.chain_seq = np.cumsum(self.chain_seq, axis=1)

    def project(self, seq=False):
        if seq and self.chain_seq is None:
            self.to_chain(True)
        x_hat = self.chain_seq if seq else self.chain
        x_hat = x_hat[self.SC.edge_vec.astype(bool)] 
        y = (self.SC.H @ x_hat).T
        if seq:
            if self.chain_seq is None:
                self.to_chain(True)
            self.proj_seq = y
            self.proj = y[-1].ravel()
        else:
            self.proj = y.ravel()
            self.proj_seq = None
        return y.ravel()

    def proj_diff(self, other, seq=False):
        if not seq:
            return np.linalg.norm(self.proj - other.proj, 2)
        else:
            if self.proj_seq is None:
                self.project(True)
            return np.linalg.norm(self.proj_seq - other.proj, 2, axis=1)

    def get_weight(self, seq=False):
        f = np.cumsum if seq else np.sum
        res = f([self.SC.G[u][v]['weight'] for (u,v) in self.edges])
        if seq:
            self.weight_seq = np.zeros(len(self))
            self.weight_seq[1:] = res
            self.weight = self.weight_seq[-1]
        else:
            self.weight = res
            self.weight_seq = None
        return res

    def cost(self, ref_proj, alpha, seq=False):
        if seq:
            if self.proj_seq is None:
                self.project(True)
            if self.weight_seq is None:
                self.get_weight(True)
            proj_diff_seq = np.linalg.norm(self.proj_seq - ref_proj, axis=1)
            return self.weight_seq + alpha * proj_diff_seq
        else:
            proj_diff = np.linalg.norm(self.proj - ref_proj)
            return self.weight + alpha * proj_diff

    def __neg__(self):
        return Path(self.SC, self.nodes[::-1], False)
    
    def __add__(self, other):
        if isinstance(other, int):
            assert(other in self.SC.G.neighbors(self.nodes[-1]))
            return Path(self.SC, self.nodes + [other], False)
        elif isinstance(other, list) and len(other) == 1:
            assert(other[0] in self.SC.G.neighbors(self.nodes[-1]))
            return Path(self.SC, self.nodes + [other], False)
        elif isinstance(other, list) and len(other) > 1:
            assert(other[0] == self.nodes[-1])
            return Path(self.SC, np.append(self.nodes, other), False)
        elif isinstance(other, Path):
            assert(self.nodes[-1] == other.nodes[0])
            return Path(self.SC, np.append(self.nodes, other.nodes[1:]), False)

    def __sub__(self, other):
        if isinstance(other, int) or (isinstance(other, list) and len(other) == 1):
            return self + other
        else:
            return self + (-other)

    def __getitem__(self, i) -> int:
        return self.nodes[i]

    def __repr__(self) -> str:
        res = "-".join([str(node) for node in self.nodes])
        return res

    def __len__(self) -> int:
        return len(self.nodes)


def nodes_from_coords(SC, coords):
    node_set = []

    for i, coord in enumerate(coords):
        node_dists = np.linalg.norm(SC.nodes - coord, axis=1)
        node_dists = [(x if SC.node_vec[i] else np.inf) for i, x in enumerate(node_dists)]
        node_order = np.argmin(node_dists)
        node_set.append(node_order)
    
    return np.array(node_set)

def path_from_coords(SC, coords):
    nodes_to_visit = nodes_from_coords(SC, coords)
    num_nodes = len(nodes_to_visit)
    path_nodes = [nodes_to_visit[0]]

    for i in range(num_nodes-1):
        start, end = nodes_to_visit[i], nodes_to_visit[i+1]
        interp_path_nodes = nx.dijkstra_path(SC.G, start, end)
        path_nodes += interp_path_nodes[1:]
    
    path = Path(SC, path_nodes)

    return path