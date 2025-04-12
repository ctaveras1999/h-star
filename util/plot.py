import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx 

def plot_sc(SC, ax=None, figsize=(6,6)):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)    
    for i, face in enumerate(SC.faces):
        if SC.face_vec[i]:
            continue
        color = 'darkgray'
        n1, n2, n3 = face
        tri = np.vstack([SC.nodes[n1], SC.nodes[n2], SC.nodes[n3]])
        ti = plt.Polygon(tri, color=color)#, alpha=alpha_val)
        ax.add_patch(ti)

    for i, edge in enumerate(SC.edges):
        if SC.edge_vec[i] == 0:
            continue
        n1, n2 = edge
        line = np.vstack([SC.nodes[n1], SC.nodes[n2]])

        ax.plot(line[:,0], line[:,1], color='black', linewidth=1, alpha=0.2)
        ax.axis('off')

    active_nodes = SC.node_vec.astype(bool)

    ax.scatter(SC.nodes[active_nodes,0], SC.nodes[active_nodes,1], color='black', s=1.5)
    return ax

def plot_path(SC, path, color, label, ax=None, figsize=(6,6)):
    matplotlib.style.use('default')
    if ax is None:
        ax = plot_sc(SC, ax, figsize)
    for edge in path.edges:
        n1, n2 = edge
        x1, y1 = SC.nodes[n1]
        x2, y2 = SC.nodes[n2]
        dx, dy = x2 - x1, y2 - y1
        ax.arrow(x1, y1, dx, dy, color=color, width=0.01, length_includes_head=True)

        start_and_end = np.array([path.nodes[0], path.nodes[-1]])

        ax.scatter(SC.nodes[start_and_end, 0], SC.nodes[start_and_end, 1], color='black', s=50, marker='*')

    return ax


def several_path_plot(SC, paths, labels=None, ax=None, colors=None, figsize=(5,5), legend=False):
    matplotlib.style.use('default')
    if ax is None:
        ax = plot_sc(SC)

    num_paths = len(paths)

    if colors is None:
        colors = [plt.cm.Set1(i) for i in range(len(paths))]

    for path, color in zip(paths, colors):
        plot_path(SC, path, color, '', ax, figsize)

    if labels is None:
        labels = [f"Path {i+1}" for i in range(num_paths)] 
    
    if legend:
        leg = ax.legend(labels)
        
        for i in range(num_paths):
            leg.legendHandles[i].set_color(colors[i])

    return ax

def plot_bases(SC, font=None, figsize=(5,5), arrowmult=7, node_size=5, edge_color="blue", node_color='gray'):
    if font is None:
        font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 16,
            }
    node_pos = {i:SC.nodes[i] for i in (np.where(SC.node_vec == 1)[0])}
    num_bases = SC.H.shape[0]
    
    for i in range(num_bases):
        harm_vec = SC.H[i,:]
        harm_vec_sign = np.sign(harm_vec)
        harm_vec_weight = np.abs(harm_vec)
        G_oriented = nx.DiGraph()
        j = 0
        for k, edge in enumerate(SC.edges):
            if not SC.edge_vec[k]:
                continue
            sign, weight = harm_vec_sign[j], harm_vec_weight[j] * 10
            oriented_edge = (edge[0], edge[1]) if (sign > 0) else (edge[1], edge[0])
            G_oriented.add_edge(oriented_edge[0], oriented_edge[1], weight=weight)
            j += 1

        oriented_weights = [G_oriented[u][v]['weight'] for u,v in G_oriented.edges]
        plt.figure(figsize=figsize)
        ax = plt.gca()

        for j, face in enumerate(SC.faces):
            #alpha_val = 0 if self.face_vec[i] else 0.2
            #color = 'white' if self.face_vec[i] else 'darkgray'
            if SC.face_vec[j]:
                continue
            color = 'darkgray'
            n1, n2, n3 = face
            tri = np.vstack([SC.nodes[n1], SC.nodes[n2], SC.nodes[n3]])
            ti = plt.Polygon(tri, color=color, alpha=1)
            ax.add_patch(ti)

        nx.draw(G_oriented, pos=node_pos, node_size=node_size, node_color=node_color, width=oriented_weights, edge_color=edge_color, arrows=True, arrowstyle='-|>', arrowsize=[arrowmult * i for i in oriented_weights])

        plt.title(f"Harmonic Basis Vector $h_{i+1}$", fontdict=font)
        plt.show()

def plot_projections(paths, colors, proj_axes=(0,1), ax=None, arrow_width = 0.009, star_size = 300,  origin_size = 250, figsize=(6,6)):
    matplotlib.style.use('ggplot')
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
    for i, path in enumerate(paths):
        color = colors[i]
        projections = path.project(True)
        plot_projs = np.zeros((projections.shape[0], 2))
        num_holes = projections.shape[1]
        if num_holes == 1:
            plot_projs[:,0] = projections.ravel()
            arrow_width = 0.001
        if num_holes == 2:
            plot_projs = projections
        if num_holes > 2:
            plot_projs[:,0] = projections[:,proj_axes[0]]
            plot_projs[:,1] = projections[:,proj_axes[1]]

        if num_holes == 1:
            ax.scatter(plot_projs[1:,0], plot_projs[1:,1], color=color, s=origin_size/10, marker='.')
        else:
            for j in range(plot_projs.shape[0]-1):
                curr_proj, next_proj = plot_projs[j], plot_projs[j+1]
                curr_x, curr_y = curr_proj
                next_x, next_y = next_proj
                dx, dy = next_x - curr_x, next_y - curr_y
                ax.arrow(curr_x, curr_y, dx, dy, color=color, width=arrow_width, length_includes_head=True)
        ax.scatter(plot_projs[-1,0], plot_projs[-1,1], s=star_size, color=color, marker='*')
    ax.scatter([0], [0], s=origin_size, color='black', marker='.')