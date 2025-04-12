import time
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


def hstar(SC, ref_path, alpha, verbose=False, other=False, get_proj_time=False):
    start, end, ref_proj = ref_path[0], ref_path[-1], ref_path.proj
    start_time = time.time()
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
    elapsed_time = time.time() - start_time

    if other:
        heads = [node for node in node_children if not node_children[node]]
        other_paths = [Path(SC, backtrace(prev, start, head)) for head in heads if head != end]
    else: 
        other_paths = []

    if not get_proj_time:
        return path, dist[end], prev, visit_order, other_paths

    return path, dist[end], prev, visit_order, other_paths, proj, elapsed_time


def complete(SC, partial, ref_path, end, alpha):
    reversed_path = partial[1:][::-1]
    new_ref_path = Path(SC, reversed_path + list(ref_path.nodes))
    remainder, _, _, visited, _ = hstar(SC, new_ref_path, alpha)
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

    if pruning:
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


def k_rollout(SC, ref_path, alpha, max_depth, pruning, eps=1e-2, verbose=False, get_time=False):
    start_time = time.time()
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

    elapsed_time = time.time() - start_time

    if not get_time:
        return path, best_costs, best_paths, num_visited

    return path, best_costs, best_paths, num_visited, elapsed_time


def bhattacharya(SC, ref_path, eps=1e-6, others=False, verbose=False, get_time=False):
    start_time = time.time()
    start, end = ref_path[0], ref_path[-1]
    vec_dist = lambda x, y: np.linalg.norm(np.array(x)-np.array(y))
    heuristic = lambda u, v: vec_dist(SC.nodes[u], SC.nodes[v])
    ref_proj, zero_proj = tuple(ref_path.proj), tuple(np.zeros_like(ref_path.proj))

    pq = pqdict()
    pq[(start, 0)] = heuristic(start, end)

    prev = {(start,0):-1}
    dist = {start:{0:0}}
    visited = {start:set([0])}
    proj_dict = {start:{0:zero_proj}, end:{0:ref_proj}}

    def add_visited(node, proj_idx):
        if node in visited:
            visited[node].add(proj_idx)
        else:
            visited[node] = set([proj_idx])

    def get_proj_idx(node, node_proj):
        if node not in proj_dict:
            idx = 0
            proj_dict[node] = {idx:node_proj}
        elif node in proj_dict:
            idx = max_idx = -1
            for idx_i in proj_dict[node]:
                max_idx = max(max_idx, idx_i)
                if almost_equal(node_proj, proj_dict[node][idx_i]):
                    idx = idx_i
                    break
            if idx < 0:
                idx = max_idx + 1
                proj_dict[node][idx] = node_proj

        return idx

    num_visited = 0
    while True:
        (u, proj_u_idx), est_dist_u = pq.popitem()
        proj_u = proj_dict[u][proj_u_idx]
        dist_u = est_dist_u - heuristic(u, end)
        add_visited(u, proj_u_idx)

        if verbose and (u == end):
            proj_diff_u = vec_dist(proj_u, ref_proj)
            print(f"Found path with projection difference = {np.round(proj_diff_u, 10)} and length  = {np.round(dist[u][proj_u], 2)}")
        if (u == end) and almost_equal(ref_proj, proj_u, eps):
            break
        
        for v in SC.G.neighbors(u):
            uv_edge = Path(SC, [u,v])
            proj_v = np.array(proj_u) + uv_edge.proj
            dist_v_alt = dist_u + uv_edge.weight
            proj_v_idx = get_proj_idx(v, proj_v)  

            if v in visited and proj_v_idx in visited[v]:
                continue

            if v not in dist:
                dist[v] = {proj_v_idx:np.inf}
            elif proj_v_idx not in dist[v]:
                dist[v][proj_v_idx] = np.inf

            if dist_v_alt < dist[v][proj_v_idx]:
                prev[(v, proj_v_idx)] = (u, proj_u_idx)
                dist[v][proj_v_idx] = dist_v_alt
                pq[v, proj_v_idx] = dist_v_alt + heuristic(v, end)
        num_visited += 1


    def backtrace2(prev, start, end, ref_proj_idx, others):
        u, u_proj_idx = end, ref_proj_idx

        path = []
        while (u, u_proj_idx) != (start, 0):
            path = [u] + path
            (u, u_proj_idx) = prev[(u, u_proj_idx)]
        path = Path(SC, [u] + path)

        if not others:
            return path
        
        others_proj_idx = [x for x in proj_dict[end].keys() if x != ref_proj_idx]
        other_paths = [backtrace2(prev, start, end, x, False) for x in others_proj_idx]

        return path, other_paths

    elapsed_time = time.time() - start_time



    if get_time:
        return *backtrace2(prev, start, end, 0, others), num_visited, elapsed_time
    else:
        return backtrace2(prev, start, end, 0, others)
