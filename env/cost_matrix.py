"""
Cost matrix factory for WarehouseEnv.

Modes (set via cost_matrix key in config):
  null / "random" : symmetric random, Uniform(0.5, 2.0)
  "hub"           : hub-and-spoke with tiered costs
  list[list]      : explicit N x N matrix from config
"""

import math
import numpy as np


def make_cost_matrix(cfg: dict, rng: np.random.Generator) -> np.ndarray:
    n    = cfg["env"]["n_warehouses"]
    mode = cfg.get("cost_matrix", None)

    if mode is None or mode == "random":
        return _random_cost_matrix(n, rng)

    if mode == "hub":
        hub_fraction = float(cfg.get("hub_fraction", 0.2))
        return _hub_cost_matrix(n, rng, hub_fraction)

    C = np.array(mode, dtype=np.float32)
    if C.shape != (n, n):
        raise ValueError(f"cost_matrix shape {C.shape} does not match n_warehouses={n}")
    return C


def _random_cost_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    c = rng.uniform(0.5, 2.0, size=(n, n)).astype(np.float32)
    c = (c + c.T) / 2.0
    np.fill_diagonal(c, 0.0)
    return c


def _hub_cost_matrix(n: int, rng: np.random.Generator, hub_fraction: float = 0.2) -> np.ndarray:
    """
    Hub-and-spoke cost matrix.
    The first ceil(hub_fraction * N) warehouses are hubs.

    Cost tiers (symmetric):
      hub to hub   : Uniform(0.1, 0.3)
      hub to spoke : Uniform(0.3, 0.8)
      spoke to spoke: Uniform(1.0, 2.0)
    """
    n_hubs  = max(1, math.ceil(hub_fraction * n))
    hub_set = set(range(n_hubs))

    c = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            i_hub = i in hub_set
            j_hub = j in hub_set
            if i_hub and j_hub:
                cost = float(rng.uniform(0.1, 0.3))
            elif i_hub or j_hub:
                cost = float(rng.uniform(0.3, 0.8))
            else:
                cost = float(rng.uniform(1.0, 2.0))
            c[i, j] = c[j, i] = cost
    return c


def get_hub_indices(n: int, hub_fraction: float = 0.2) -> list[int]:
    return list(range(max(1, math.ceil(hub_fraction * n))))
