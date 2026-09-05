import numpy as np


def particle_swarm_optimization(f, bounds, pop_size=50, max_gens=200,
                                 c1=2.0, c2=2.0, w=0.7, seed=None):
    rng = np.random.default_rng(seed)
    (xmin, xmax), (ymin, ymax) = bounds
    lo = np.array([xmin, ymin])
    hi = np.array([xmax, ymax])
    span = hi - lo

    pos = lo + rng.random((pop_size, 2)) * span
    vel = (rng.random((pop_size, 2)) - 0.5) * span * 0.1

    fitness = np.array([f(p[0], p[1]) for p in pos])

    pbest_pos = pos.copy()
    pbest_val = fitness.copy()

    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    best_history = np.empty(max_gens)
    avg_history = np.empty(max_gens)

    for gen in range(max_gens):
        r1 = rng.random((pop_size, 2))
        r2 = rng.random((pop_size, 2))

        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))
        pos = pos + vel
        pos = np.clip(pos, lo, hi)

        fitness = np.array([f(p[0], p[1]) for p in pos])

        improved = fitness < pbest_val
        pbest_pos[improved] = pos[improved]
        pbest_val[improved] = fitness[improved]

        gen_best_idx = np.argmin(pbest_val)
        if pbest_val[gen_best_idx] < gbest_val:
            gbest_val = pbest_val[gen_best_idx]
            gbest_pos = pbest_pos[gen_best_idx].copy()

        best_history[gen] = gbest_val
        avg_history[gen] = fitness.mean()

    return {
        "best_x": gbest_pos[0],
        "best_y": gbest_pos[1],
        "best_val": gbest_val,
        "best_history": best_history,
        "avg_history": avg_history,
    }
