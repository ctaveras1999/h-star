import matplotlib 
import matplotlib.pyplot as plt
import numpy as np

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


def several_path_plot(SC, paths, labels=None, ax=None, colors=None, figsize=(5,5)):
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
    leg = ax.legend(labels)
    
    for i in range(num_paths):
        leg.legendHandles[i].set_color(colors[i])

    return ax