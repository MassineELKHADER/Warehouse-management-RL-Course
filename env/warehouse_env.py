"""
WarehouseEnv: multi-warehouse inventory redistribution environment.

State  : dict with keys
    "inventory" : (N,) float32 -- current stock at each warehouse
    "demand"    : (N,) float32 -- demand for this step

Action : (N, N) float32 transport matrix T
    T[i,j] = units shipped from warehouse i to warehouse j
    Feasibility constraints are enforced inside step() via _project_action().

Reward : -(transport cost) - lambda*(unmet demand) - (overflow penalty)
         Optional external supplier adds a replenishment penalty when enabled.

Expects the full config dict (not cfg["env"]).
"""

import numpy as np
from env.demand_models import make_demand_model
from env.cost_matrix import make_cost_matrix


class WarehouseEnv:
    def __init__(self, cfg: dict, seed: int = 42):
        env_cfg = cfg["env"]

        self.n               = env_cfg["n_warehouses"]
        self.max_inventory   = env_cfg.get("max_inventory", 100)
        self.lam             = env_cfg.get("lambda_penalty", 2.0)
        self.lam_replenish   = env_cfg.get("lambda_replenishment", 1.0)
        self.episode_length  = env_cfg.get("episode_length", 50)

        ext_cfg = env_cfg.get("external_supplier", {})
        self._ext_enabled    = ext_cfg.get("enabled", False)
        self._ext_cost       = float(ext_cfg.get("cost_per_unit", 50.0))
        demand_mean          = env_cfg.get("demand_mean", 10.0)
        self._inventory_cap  = 10.0 * demand_mean

        self._rng = np.random.default_rng(seed)

        self._demand_model = make_demand_model(cfg)
        self.cost_matrix   = make_cost_matrix(cfg, self._rng)

        # normalization constants so each reward term is roughly in [0, 1]
        total_demand         = self.n * demand_mean
        max_cost_edge        = float(self.cost_matrix.max())
        self._norm_transport = max(max_cost_edge * self.n * (self.n - 1) * self.max_inventory, 1.0)
        self._norm_unmet     = max(total_demand, 1.0)
        self._norm_replenish = max(self._ext_cost * total_demand, 1.0)

        self._inventory:  np.ndarray = np.zeros(self.n, dtype=np.float32)
        self._demand:     np.ndarray = np.zeros(self.n, dtype=np.float32)
        self._step_count: int        = 0

    def reset(self) -> dict:
        if hasattr(self._demand_model, "reset"):
            self._demand_model.reset()
        self._inventory  = self._rng.uniform(0, self.max_inventory, size=self.n).astype(np.float32)
        self._demand     = self._demand_model.sample(self.n, self._rng)
        self._step_count = 0
        return self._get_state()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        T = self._project_action(action)

        transport_cost      = float(np.sum(self.cost_matrix * T))
        outgoing            = T.sum(axis=1)
        incoming            = T.sum(axis=0)
        inventory_post_ship = self._inventory - outgoing + incoming

        current_demand  = self._demand.copy()
        satisfied       = np.minimum(inventory_post_ship, current_demand)
        unmet           = np.maximum(current_demand - inventory_post_ship, 0.0)
        self._inventory = np.maximum(inventory_post_ship - satisfied, 0.0)

        replenishment_cost = 0.0
        if self._ext_enabled and unmet.sum() > 0:
            self._inventory   += unmet
            replenishment_cost = self._ext_cost * float(unmet.sum())

        overflow      = np.maximum(self._inventory - self._inventory_cap, 0.0)
        overflow_cost = float(np.sum(overflow))
        self._inventory = np.minimum(self._inventory, self._inventory_cap)

        self._demand     = self._demand_model.sample(self.n, self._rng)
        self._step_count += 1
        done = self._step_count >= self.episode_length

        reward = (
            - transport_cost       / self._norm_transport
            - self.lam        * float(np.sum(unmet)) / self._norm_unmet
            - self.lam_replenish * replenishment_cost / self._norm_replenish
            - overflow_cost        / self._norm_unmet
        )

        info = {
            "transport_cost":      transport_cost,
            "unmet_demand":        float(np.sum(unmet)),
            "demand_satisfaction": float(np.sum(satisfied) / (float(np.sum(current_demand)) + 1e-8)),
            "inventory":           self._inventory.copy(),
            "action":              T,
            "replenishment_cost":  replenishment_cost,
            "overflow_cost":       overflow_cost,
        }
        return self._get_state(), reward, done, info

    def sample_action(self) -> np.ndarray:
        T = np.zeros((self.n, self.n), dtype=np.float32)
        for i in range(self.n):
            if self._inventory[i] > 0:
                fracs    = self._rng.dirichlet(np.ones(self.n)) * self._rng.uniform(0, 1)
                fracs[i] = 0.0
                T[i]     = fracs * self._inventory[i]
        return T

    @property
    def state(self) -> dict:
        return self._get_state()

    @property
    def action_shape(self) -> tuple:
        return (self.n, self.n)

    @property
    def obs_dim(self) -> int:
        return 2 * self.n

    def _get_state(self) -> dict:
        return {
            "inventory": self._inventory.copy(),
            "demand":    self._demand.copy(),
        }

    def _project_action(self, action: np.ndarray) -> np.ndarray:
        """
        Project action into the feasible set:
          T[i,j] >= 0
          T[i,i]  = 0
          sum_j T[i,j] <= inventory[i]
        """
        T = np.array(action, dtype=np.float32)
        T = np.clip(T, 0.0, None)
        np.fill_diagonal(T, 0.0)
        row_sums = T.sum(axis=1)
        for i in range(self.n):
            if row_sums[i] > self._inventory[i]:
                T[i] *= self._inventory[i] / (row_sums[i] + 1e-8)
        return T

    def flat_obs(self, state: dict) -> np.ndarray:
        return np.concatenate([state["inventory"], state["demand"]])
