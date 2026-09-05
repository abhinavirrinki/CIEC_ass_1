import numpy as np


def eggholder(x, y):
    return -(y + 47.0) * np.sin(np.sqrt(np.abs(x / 2.0 + (y + 47.0)))) \
           - x * np.sin(np.sqrt(np.abs(x - (y + 47.0))))


def holder_table(x, y):
    return -np.abs(
        np.sin(x) * np.cos(y) * np.exp(np.abs(1.0 - np.sqrt(x ** 2 + y ** 2) / np.pi))
    )


PROBLEMS = {
    "eggholder": {
        "f": eggholder,
        "bounds": ((-512.0, 512.0), (-512.0, 512.0)),
        "known_optimum_value": -959.6407,
        "known_optimum_points": [(512.0, 404.2319)],
    },
    "holder_table": {
        "f": holder_table,
        "bounds": ((-10.0, 10.0), (-10.0, 10.0)),
        "known_optimum_value": -19.2085,
        "known_optimum_points": [
            (8.05502, 9.66459),
            (-8.05502, 9.66459),
            (8.05502, -9.66459),
            (-8.05502, -9.66459),
        ],
    },
}
