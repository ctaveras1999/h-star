import sys
sys.path.insert(0, '../util')
sys.path.insert(0, '../')

from alg import bhattacharya, k_rollout
import numpy as np
from tqdm import tqdm
from sc import Path
def alpha_sweep(SC, ref_path, alphas, num_rollout, prune=False, eps=1e-3, verbose=False):
    inputs = [(SC, Path(SC, ref_path), alpha, num_rollout, prune, eps, verbose, True) for alpha in alphas]

    data = np.zeros((len(alphas), 5), dtype=object)
    for i, x in tqdm(enumerate(inputs), total=len(alphas), ncols=40):
        path, _, _, num_visited, elapsed = k_rollout(*x)
        proj_diff, path_len = path.proj_diff(x[1]), path.weight
        data[i,:] = path, proj_diff, path_len, num_visited, elapsed

    return data

def exp2_wrapper(x):
    i, SC, ref_path, alphas, num_steps, prune, eps, folder = x
    res = alpha_sweep(SC, ref_path, alphas, num_steps, prune, eps, False)
    fname = f"{folder}/" + "data" + ("_prune" if (prune and num_steps) else "") + (f"_steps{num_steps}_batch{i}")
    np.savez(fname, data=res)
    return res

def bhattacharya_wrapper(SC, ref_path, eps, others, label, folder):
    res = bhattacharya(SC, Path(SC, ref_path), eps, others, False, True)
    fname = f"{folder}/data_bhat_batch_{label}"
    np.savez(fname, data=res)
    return res