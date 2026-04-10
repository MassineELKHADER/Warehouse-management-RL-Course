"""
Discrete Warehouse Management RL Environment, Hub and Spoke Model
3 warehouses: W1 is a central hub (cheap imports, large capacity),
W2 and W3 are spokes (expensive imports, small capacity).
Transfers from hub are cheap, encouraging redistribution.
Actions at time t take effect at time t+1.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


# Action definitions
# Each action is a dict: {"imports": [i1,i2,i3], "transfers": [(src,dst,qty),...]}

def _build_action_list():
    actions = []

    # No-op
    actions.append({"imports": [0, 0, 0], "transfers": []})

    # Import to one warehouse (3 warehouses x 3 quantities)
    for wh in range(3):
        for qty in [2, 4, 6]:
            imp = [0, 0, 0]
            imp[wh] = qty
            actions.append({"imports": imp, "transfers": []})

    # Transfer between warehouses (6 directions x 2 quantities)
    for src in range(3):
        for dst in range(3):
            if src == dst:
                continue
            for qty in [2, 4]:
                actions.append({"imports": [0, 0, 0], "transfers": [(src, dst, qty)]})

    # Bulk import to all
    for qty in [2, 4]:
        actions.append({"imports": [qty, qty, qty], "transfers": []})

    # Import to hub + transfer to spoke combos
    actions.append({"imports": [4, 0, 0], "transfers": [(0, 1, 2)]})  
    actions.append({"imports": [4, 0, 0], "transfers": [(0, 2, 2)]})  
    actions.append({"imports": [6, 0, 0], "transfers": [(0, 1, 2), (0, 2, 2)]})  

    # Large hub import + single transfer
    actions.append({"imports": [6, 0, 0], "transfers": [(0, 1, 4)]})  
    actions.append({"imports": [6, 0, 0], "transfers": [(0, 2, 4)]})  
    actions.append({"imports": [8, 0, 0], "transfers": [(0, 1, 4), (0, 2, 4)]}) 

    # Hub large imports (hub has capacity 30)
    actions.append({"imports": [8, 0, 0], "transfers": []}) 
    actions.append({"imports": [10, 0, 0], "transfers": []})  

    # Transfer 6 from hub (hub can hold more)
    actions.append({"imports": [0, 0, 0], "transfers": [(0, 1, 6)]})  
    actions.append({"imports": [0, 0, 0], "transfers": [(0, 2, 6)]})

    return actions


ACTIONS = _build_action_list()
N_ACTIONS = len(ACTIONS)


# W1 = hub (index 0), W2/W3 = spokes (indices 1, 2)
DEFAULT_CAPACITIES = np.array([30, 15, 15], dtype=int)

# Import costs: cheap at hub, expensive at spokes
DEFAULT_IMPORT_COSTS = np.array([1.0, 5.0, 5.0])

# Holding costs: cheap at hub, expensive at spokes
DEFAULT_HOLDING_COSTS = np.array([0.3, 1.0, 1.0])

# Hub→spoke cheap, spoke→hub moderate, spoke→spoke expensive
DEFAULT_TRANSFER_COSTS = {
    (0, 1): 0.5, (0, 2): 0.5,   
    (1, 0): 1.5, (2, 0): 1.5,   
    (1, 2): 3.0, (2, 1): 3.0,   
}


# Demand model parameters
DEMAND_PARAMS = {
    0: {"base": 8, "amplitude": 3, "phase": 0},   # W1 peak summer
    1: {"base": 5, "amplitude": 2, "phase": 6},   # W2 peak winter
    2: {"base": 3, "amplitude": 1, "phase": 3},   # W3 peak autumn
}


def demand_mean(warehouse, month):
    p = DEMAND_PARAMS[warehouse]
    return max(1.0, p["base"] + p["amplitude"] * np.sin(2 * np.pi * (month - p["phase"]) / 12))


# Environment
class WarehouseEnv:
    """
    State: (stock_w1, stock_w2, stock_w3, month)
    - stock_wi in [0, capacity_i]
    - month in [0, 11]

    Reward (hub-and-spoke):
        - alpha * unmet_demand              (penalty, per unit)
        - import_cost[i] * imports[i]       (per-warehouse import cost)
        - transfer_cost[src,dst] * qty      (directional transfer cost)
        - holding_cost[i] * stock[i]        (per-warehouse holding cost)
        - penalty for infeasible action parts
    """

    def __init__(
        self,
        capacities = None,
        n_months = 12,
        alpha = 10.0,
        import_costs = None,
        holding_costs = None,
        transfer_costs = None,
        infeasible_penalty = 1.0,
        seed = None,
    ):
        self.capacities = capacities if capacities is not None else DEFAULT_CAPACITIES.copy()
        self.n_months = n_months
        self.alpha = alpha
        self.import_costs = import_costs if import_costs is not None else DEFAULT_IMPORT_COSTS.copy()
        self.holding_costs = holding_costs if holding_costs is not None else DEFAULT_HOLDING_COSTS.copy()
        self.transfer_costs = transfer_costs if transfer_costs is not None else DEFAULT_TRANSFER_COSTS.copy()
        self.infeasible_penalty = infeasible_penalty

        self.rng = np.random.default_rng(seed)

        # State
        self.stocks = np.zeros(3, dtype=int)
        self.month = 0
        self.done = False

    @property
    def capacity(self):
        """Max capacity across all warehouses (for backward compat)."""
        return int(np.max(self.capacities))


    def reset(self, initial_stocks = None):
        if initial_stocks is not None:
            self.stocks = np.array(initial_stocks, dtype=int)
        else:
            # Start each warehouse at half capacity
            self.stocks = (self.capacities // 2).astype(int)
        self.month = 0
        self.done = False
        return self._get_state()

    def step(self, action_idx):
        assert not self.done, "Episode is done, call reset()"
        assert 0 <= action_idx < N_ACTIONS

        action = ACTIONS[action_idx]
        imports = np.array(action["imports"], dtype=int)
        transfers = action["transfers"]

        infeasible_count = 0

        # Apply imports (clamped by per-warehouse capacity)
        new_stocks = self.stocks.copy()
        actual_imports = np.zeros(3, dtype=int)
        for i in range(3):
            space = self.capacities[i] - new_stocks[i]
            actual_import = min(imports[i], space)
            if actual_import < imports[i]:
                infeasible_count += imports[i] - actual_import
            new_stocks[i] += actual_import
            actual_imports[i] = actual_import

        # Apply transfers (clamped by source stock and dest capacity)
        total_transferred = 0
        transfer_cost_total = 0.0
        per_transfer_details = []
        for src, dst, qty in transfers:
            available = new_stocks[src]
            space = self.capacities[dst] - new_stocks[dst]
            actual = min(qty, available, space)
            if actual < qty:
                infeasible_count += qty - actual
            new_stocks[src] -= actual
            new_stocks[dst] += actual
            total_transferred += actual
            cost_per_unit = self.transfer_costs.get((src, dst), 1.0)
            transfer_cost_total += cost_per_unit * actual
            per_transfer_details.append((src, dst, actual))

        # Sample demand
        demands = np.array([
            self.rng.poisson(demand_mean(i, self.month)) for i in range(3)
        ], dtype=int)

        # Fulfill demand
        fulfilled = np.minimum(demands, new_stocks)
        unmet = demands - fulfilled
        new_stocks -= fulfilled

        # Compute reward
        import_cost_total = np.sum(self.import_costs * actual_imports)
        holding_cost_total = np.sum(self.holding_costs * new_stocks)

        reward = (
            - self.alpha * np.sum(unmet)
            - import_cost_total
            - transfer_cost_total
            - holding_cost_total
            - self.infeasible_penalty * infeasible_count
        )

        self.stocks = new_stocks
        self.month += 1
        self.done = self.month >= self.n_months

        info = {
            "demands": demands,
            "fulfilled": fulfilled,
            "unmet": unmet,
            "actual_imports": actual_imports,
            "total_transferred": total_transferred,
            "transfer_details": per_transfer_details,
            "infeasible_count": infeasible_count,
            "stocks": self.stocks.copy(),
        }

        return self._get_state(), reward, self.done, info

    def _get_state(self):
        return (int(self.stocks[0]), int(self.stocks[1]), int(self.stocks[2]), self.month)


    def get_transition_reward(
        self, state, action_idx, demand_sample
    ):
        """
        Deterministic transition given a specific demand realization.
        Used by Value Iteration to enumerate over demand scenarios.
        Returns (next_state, reward, info).
        """
        s1, s2, s3, month = state
        stocks = np.array([s1, s2, s3], dtype=int)

        action = ACTIONS[action_idx]
        imports = np.array(action["imports"], dtype=int)
        transfers = action["transfers"]

        infeasible_count = 0
        new_stocks = stocks.copy()
        actual_imports = np.zeros(3, dtype=int)

        for i in range(3):
            space = self.capacities[i] - new_stocks[i]
            actual_import = min(imports[i], space)
            if actual_import < imports[i]:
                infeasible_count += imports[i] - actual_import
            new_stocks[i] += actual_import
            actual_imports[i] = actual_import

        total_transferred = 0
        transfer_cost_total = 0.0

        for src, dst, qty in transfers:
            available = new_stocks[src]
            space = self.capacities[dst] - new_stocks[dst]
            actual = min(qty, available, space)
            if actual < qty:
                infeasible_count += qty - actual
            new_stocks[src] -= actual
            new_stocks[dst] += actual
            total_transferred += actual
            cost_per_unit = self.transfer_costs.get((src, dst), 1.0)
            transfer_cost_total += cost_per_unit * actual

        demands = demand_sample
        fulfilled = np.minimum(demands, new_stocks)
        unmet = demands - fulfilled
        new_stocks -= fulfilled
        new_stocks = np.clip(new_stocks, 0, self.capacities)

        import_cost_total = np.sum(self.import_costs * actual_imports)
        holding_cost_total = np.sum(self.holding_costs * new_stocks)

        reward = (
            - self.alpha * np.sum(unmet)
            - import_cost_total
            - transfer_cost_total
            - holding_cost_total
            - self.infeasible_penalty * infeasible_count
        )

        next_month = month + 1
        done = next_month >= self.n_months
        next_state = (int(new_stocks[0]), int(new_stocks[1]), int(new_stocks[2]), next_month)

        info = {
            "unmet": unmet,
            "actual_imports": actual_imports,
            "total_transferred": total_transferred,
            "infeasible_count": infeasible_count,
        }

        return next_state, reward, info


    def describe_action(self, action_idx):
        a = ACTIONS[action_idx]
        parts = []
        for i, imp in enumerate(a["imports"]):
            if imp > 0:
                parts.append(f"import {imp} to W{i+1}")
        for src, dst, qty in a["transfers"]:
            parts.append(f"transfer {qty} from W{src+1} to W{dst+1}")
        return ", ".join(parts) if parts else "no-op"
