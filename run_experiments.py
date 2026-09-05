import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from test_functions import PROBLEMS
from de import differential_evolution
from pso import particle_swarm_optimization

POP_SIZES = [20, 50, 100, 200]
GEN_COUNTS = [50, 200]
SEED_BASE = 42

OUT_DIR = "results"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def make_convergence_plot(best_hist, avg_hist, title, path):
    gens = np.arange(1, len(best_hist) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(gens, best_hist, label="Best value", linewidth=2)
    plt.plot(gens, avg_hist, label="Average value", linewidth=2, linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Objective function value")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def run_all():
    rows = []
    run_counter = 0

    for func_name, prob in PROBLEMS.items():
        f = prob["f"]
        bounds = prob["bounds"]
        known_val = prob["known_optimum_value"]

        for algo_name in ["DE", "PSO"]:
            for pop in POP_SIZES:
                for gens in GEN_COUNTS:
                    run_counter += 1
                    seed = SEED_BASE + run_counter

                    if algo_name == "DE":
                        res = differential_evolution(
                            f, bounds, pop_size=pop, max_gens=gens,
                            CR=0.8, R=0.5, seed=seed,
                        )
                    else:
                        res = particle_swarm_optimization(
                            f, bounds, pop_size=pop, max_gens=gens,
                            c1=2.0, c2=2.0, w=0.7, seed=seed,
                        )

                    title = f"{func_name} | {algo_name} | pop={pop}, gens={gens}"
                    fname = f"{func_name}_{algo_name}_pop{pop}_gen{gens}.png"
                    make_convergence_plot(
                        res["best_history"], res["avg_history"], title,
                        os.path.join(PLOT_DIR, fname),
                    )

                    error = abs(res["best_val"] - known_val)
                    rows.append({
                        "function": func_name,
                        "algorithm": algo_name,
                        "pop_size": pop,
                        "max_gens": gens,
                        "found_x": round(float(res["best_x"]), 5),
                        "found_y": round(float(res["best_y"]), 5),
                        "found_value": round(float(res["best_val"]), 5),
                        "known_optimum_value": known_val,
                        "abs_error": round(float(error), 6),
                    })
                    print(f"[{run_counter:2d}] {title:55s} "
                          f"-> value={res['best_val']:.4f} (error={error:.2e})")

    csv_path = os.path.join(OUT_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} plots to {PLOT_DIR}/")
    print(f"Saved summary table to {csv_path}")


if __name__ == "__main__":
    run_all()
