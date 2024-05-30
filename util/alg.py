import numpy as np
from util.sc import *
from util.pqdict import pqdict


def backtrace(prev, start, end):
    node = end
    path = []
    while node != start:
        path.append(node)
        node = prev[node]
    path.append(node) 
    path.reverse()
    return path


def next_node(pq, proj, edge_weight, visit_order, visited):
    u = pq.pop()
    proj_u = proj[u]
    weight_u = edge_weight[u]
    visit_order.append((u, proj_u))
    visited.add(u)
    return u, proj_u, weight_u


def update_node(u, v, new_cost, weight_alt, proj_alt, pq, prev, dist, edge_weight, proj):
    prev[v] = u
    dist[v] = new_cost
    edge_weight[v] = weight_alt
    proj[v] = proj_alt
    pq[v] = new_cost


def proj_cost(proj1, proj2):
    proj_diff = proj1 - proj2
    return np.linalg.norm(proj_diff, ord=2)


def almost_equal(x, y, eps=1e-8):
    return np.linalg.norm(np.array(x) - np.array(y)) < eps


def hstar(SC, start, end, ref_proj, alpha, verbose=False, other=False, get_proj=False):
    prev = {}
    dist = {v:np.inf for v in range(SC.node_vec.shape[0])}
    proj = {v:np.inf * np.ones(ref_proj.shape) for v in range(SC.node_vec.shape[0])}
    edge_weight = {v:np.inf for v in range(SC.node_vec.shape[0])}
    visited = set()
    pq = pqdict()

    # initialize node cost
    for u in SC.active_nodes: 
        pq[u] = float("inf") if (u != start) else 0

    dist[start], edge_weight[start], proj[start] = 0, 0, np.zeros(ref_proj.shape)

    visit_order = []
    node_children = {node:0 for node in np.where(SC.node_vec == 1)[0]}

    while len(pq) > 0:
        u, proj_u, weight_u = next_node(pq, proj, edge_weight, visit_order, visited)
        u_children = 0
        if (u == end) and (other is False):
            break
        if verbose:
            print('. . . . . . . . . . . '*3)
            print(f'visiting {u} with cost {dist[u]}')
            print(f'node position = {SC.nodes[u]}')
        
        for v in SC.G.neighbors(u):
            if v in visited:
                continue
            weight_v_alt = weight_u + SC.G[u][v]['weight']
            weight_v_now = edge_weight[v]
            proj_alt = proj_u + Path(SC, [u, v]).proj
            cost_now = dist[v]
            cost_alt = weight_v_alt + alpha * proj_cost(proj_alt, ref_proj)

            if verbose:
                proj_diffs = [proj_cost(proj[v], ref_proj), proj_cost(proj_alt, ref_proj)]
                print(f'curr neighbor = {v}, cost now = {cost_now}, cost alt : {cost_alt}')
                print(f'curr proj diff = {proj_diffs[0]}, new proj diff = {proj_diffs[1]}')
                print(f'neighbor position = {SC.nodes[v]}')
                print(f"old edge weights: {weight_v_now} & new edge weights: {weight_v_alt}")
            
            if (almost_equal(cost_now, cost_alt) and u < prev[v]) or cost_alt < cost_now:
                if v in prev:
                    node_children[prev[v]] -= 1
                u_children += 1
                update_node(u, v, cost_alt, weight_v_alt, proj_alt, pq, prev, dist, edge_weight, proj)

        node_children[u] = u_children

    path = Path(SC, backtrace(prev, start, end))

    if verbose:
        print("=== Dijkstra's Algo Output ===")
        print("Distances")
        print(dist)
        print("Path")
        print(path)

    if other:
        heads = [node for node in node_children if not node_children[node]]
        other_paths = [Path(SC, backtrace(prev, start, head)) for head in heads if head != end]
    else: 
        other_paths = []


    if not get_proj:
        return path, dist[end], prev, visit_order, other_paths

    return path, dist[end], prev, visit_order, other_paths, proj


def complete(SC, partial, ref_path, end, alpha):
    curr, reversed_path = partial[-1], partial[1:][::-1]
    if len(reversed_path) > 0:
        new_ref_proj = Path(SC, reversed_path + list(ref_path.nodes)).proj
    else:
        new_ref_proj = ref_path.proj
    remainder, _, _, visited, _ = hstar(SC, curr, end, new_ref_proj, alpha)
    completed = Path(SC, partial + remainder[1:])
    num_visited = len(visited)
    return completed, num_visited


def rollout_rec(SC, partial, curr_node, end, ref_path, alpha, depth, max_depth, k, pruning, eps, verbose=False):
    if (curr_node == end) or (depth == max_depth):
        path, num_visited = (Path(SC, partial), 1) if curr_node == end else complete(SC, partial, ref_path, end, alpha)
        cost = path.cost(ref_path.proj, alpha, False)
        return path, cost, num_visited

    path_options = []
    costs = []
    total_visited = 0

    if pruning: ### <----- Left off here
        path_proj_curr = Path(SC, partial).proj
        ref_proj = ref_path.proj
        proj_diff_curr = np.linalg.norm(path_proj_curr - ref_proj, ord=2)

    neighbors = [x for x in SC.G.neighbors(curr_node)]
    if (max_depth > 1) and pruning:
        proj_diffs = [Path(SC, partial + [v]).proj for v in neighbors]
        proj_diffs = [np.linalg.norm(x - ref_proj, ord=2) for x in proj_diffs]
        num_closer_idx = np.where(proj_diffs < proj_diff_curr + eps)[0]
        if len(num_closer_idx) == 0:
            neighbors = [neighbors[np.argmin(proj_diffs)]]
        else:
            neighbors = [neighbors[x] for x in num_closer_idx]

    pruned = []
    for v in neighbors:
        if (k > 0) and (v == partial[k-1]):
            continue
        if pruning:
            path_proj_next = Path(SC, partial + [v]).proj
            path_diff_next = np.linalg.norm(path_proj_next - ref_proj, ord=2)
            if path_diff_next > proj_diff_curr + eps:
                pruned.append(v)
                if verbose: 
                    print("-----> " * depth + f"pruning node {v}")
                continue
        path_v, cost_path_v, num_visited = rollout_rec(SC, partial + [v], v, end, ref_path, alpha, depth+1, max_depth, k, pruning, eps, verbose=False)
        path_options += [path_v]
        costs += [cost_path_v]
        total_visited += num_visited
    
    if len(costs) == 0: # eps too small
        for v in pruned:
            path_v, cost_path_v, num_visited = rollout_rec(SC, partial + [v], v, end, ref_path, alpha, depth+1, max_depth, k, pruning, eps, verbose=False)
            path_options += [path_v]
            costs += [cost_path_v]
            total_visited += num_visited

    best_cost, best_idx = np.min(np.array(costs)), np.argmin(np.array(costs))
    best_path = path_options[best_idx]
    if verbose:
        print("-----> " * depth + "best cost:", np.round(best_cost, 3), "best path:", best_path)

    return best_path, best_cost, total_visited


def k_rollout(SC, ref_path, alpha, max_depth, pruning, eps=1e-2, verbose=False):
    start, end = ref_path[0], ref_path[-1]    
    vk, path = start, [start]
    best_paths, best_costs = [], []

    if verbose:
        print('vk =', vk)

    k, num_visited = 0, 0
    while vk != end:
        if verbose:
            print(f"STAGE {k}: vk = {vk}")
            print("-"*300)
        if max_depth == 0:
            path, cost_vk, num_visited = rollout_rec(SC, path, vk, end, ref_path, alpha, 0, max_depth, k, pruning, eps, verbose)
            best_paths, best_costs = [path], [cost_vk]
            break
        
        path_vk, cost_vk, visited_vk = rollout_rec(SC, path, vk, end, ref_path, alpha, 0, max_depth, k, pruning, eps, verbose)
        vk = path_vk[k+1]
        path += [vk]
        best_paths += [path_vk]
        best_costs += [cost_vk]
        num_visited += visited_vk
        k += 1
        if verbose:
            print(f"END STAGE {k-1}: vk+1={vk} w cost {np.round(cost_vk,3)}, {visited_vk} nodes visited \n")

    if max_depth > 0:
        path = Path(SC, path)

    return path, best_costs, best_paths, num_visited


def bhattacharya(SC, ref_path, start, end, eps=1e-6, others=False, verbose=False): # test!!
    vec_dist = lambda x, y: np.linalg.norm(np.array(x)-np.array(y))
    heuristic = lambda u, v: vec_dist(SC.nodes[u], SC.nodes[v])
    proj_ref, zero_proj = ref_path.proj, np.zeros_like(ref_path.proj)
    pq = pqdict()
    pq[(start, tuple(zero_proj))] = 0
    dist = {(start, tuple(zero_proj)):0}
    prev = {(start, tuple(zero_proj)):-1}
    visited = [(start, tuple(zero_proj))]

    i = 0
    while True:
        u, proj_u = pq.pop()
        dist_u = [dist[x] for x in dist if (x[0] == u) and almost_equal(proj_u, x[1], eps)][0]
        cost_to_go = heuristic(u, end)
        proj_diff_u = vec_dist(proj_u, proj_ref)
        visited.append((u, proj_u))
        if verbose and (u == end):
            print(f"Found path with projection difference = {np.round(proj_diff_u, 10)} and length  = {np.round(dist[u, proj_u], 2)}")
        if (u == end) and almost_equal(proj_ref, proj_u, eps):
            break
        
        for v in SC.G.neighbors(u):
            uv_edge = Path(SC, [u,v])
            proj_v = np.array(proj_u) + uv_edge.proj
            dist_v_alt = dist_u + uv_edge.get_weight(False)
            dist_v = np.inf
            found, visited_v = False, False

            for (w, proj_w) in visited: # skip node if its already been visited
                if w == v and almost_equal(proj_v, proj_w):
                    visited_v = True
                    break

            if visited_v: # skip node if its already been visited
                continue

            for w, proj_w in dist: # see if node has been updated by neighbor 
                if w == v and almost_equal(proj_v, proj_w):
                    found, proj_v, dist_v = True, proj_w, dist[(w, proj_w)]
                    break
            
            
            if not found or (dist_v_alt < dist_v):
                v_key = (v, tuple(proj_v))
                prev[v_key] = (u, proj_u)
                dist[v_key] = dist_v_alt
                pq[v_key] = dist_v_alt + cost_to_go  
        i += 1

    def backtrace2(prev, start, end, ref_proj, others=others):
        node, other_nodes = None, []
        for x in prev:
            if (x[0] == end) and almost_equal(x[1],  ref_proj, eps):
                node = x
                break
            elif x[0] == end:
                other_nodes.append(x)
                
        path = []
        while node[0] != start:
            path = [node[0]] + path
            node = prev[node]
        path = Path(SC, [node[0]] + path) # add start node 

        if not others:
            return path

        other_paths = []
        for node in other_nodes:
            other_path = []
            while node[0] != start:
                other_path = [node[0]] + other_path
                node = prev[node]
            other_paths.append(Path(SC, [node[0]] + other_path)) # add start node 

        return path, other_paths

    num_visited = len(visited)

    if not others:
        path = backtrace2(prev, start, end, proj_ref, others)
        return path, num_visited
    else:
        path, other_paths = backtrace2(prev, start, end, proj_ref, others)
        return path, other_paths, num_visited
        
