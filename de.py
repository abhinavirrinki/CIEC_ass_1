import numpy as np


def differential_evolution(f, bounds, pop_size=50, max_gens=200, CR=0.8, R=0.5,
                            seed=None, f_range=(-2.0, 2.0)):
    rng = np.random.default_rng(seed)
    (xmin, xmax), (ymin, ymax) = bounds
    lo = np.array([xmin, ymin])
    hi = np.array([xmax, ymax])

    pop = lo + rng.random((pop_size, 2)) * (hi - lo)
    fitness = np.array([f(ind[0], ind[1]) for ind in pop])

    best_history = np.empty(max_gens)
    avg_history = np.empty(max_gens)

    for gen in range(max_gens):
        F = rng.uniform(f_range[0], f_range[1])

        best_idx = np.argmin(fitness)
        x_best = pop[best_idx]

        new_pop = pop.copy()
        new_fitness = fitness.copy()

        for i in range(pop_size):
            candidates = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

            x_r1, x_r2, x_r3 = pop[r1], pop[r2], pop[r3]

            v = x_r1 + R * (x_best - x_r1) + F * (x_r2 - x_r3)

            u = pop[i].copy()
            j_rand = rng.integers(0, 2)
            for j in range(2):
                if rng.random() < CR or j == j_rand:
                    u[j] = v[j]

            u = np.clip(u, lo, hi)

            f_u = f(u[0], u[1])
            if f_u <= fitness[i]:
                new_pop[i] = u
                new_fitness[i] = f_u

        pop, fitness = new_pop, new_fitness
        best_history[gen] = fitness.min()
        avg_history[gen] = fitness.mean()

    best_idx = np.argmin(fitness)
    return {
        "best_x": pop[best_idx][0],
        "best_y": pop[best_idx][1],
        "best_val": fitness[best_idx],
        "best_history": best_history,
        "avg_history": avg_history,
    }
