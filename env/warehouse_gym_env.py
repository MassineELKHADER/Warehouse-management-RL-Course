"""
Gymnasium wrapper around WarehouseEnv for use with Stable Baselines 3.

Observation: Box(0, 1, shape=(2*N+1,)) -- [inventory/max, demand/scale, step/episode_len]
Action:      Box(-1, 1, shape=(N*N,))  -- raw flat transport matrix
             _project_action() in WarehouseEnv enforces all constraints.
"""

import numpy as np
import gymnasium as gym

from env.warehouse_env import WarehouseEnv


class WarehouseGymEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, seed: int = 42):
        super().__init__()
        self._env  = WarehouseEnv(cfg, seed=seed)
        self._cfg  = cfg
        N           = self._env.n
        demand_mean = cfg["env"].get("demand_mean", 10.0)
        self._demand_scale = demand_mean * 3.0

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2 * N + 1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(N * N,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        state = self._env.reset()
        return self._obs(state), {}

    def step(self, action: np.ndarray):
        # clip(a, 0, 1) * max_inventory so that a<=0 maps to zero shipping
        T_scaled = np.clip(action, 0.0, 1.0) * self._env.max_inventory
        T = T_scaled.reshape(self._env.n, self._env.n)
        next_state, reward, done, info = self._env.step(T)
        obs = self._obs(next_state)
        return obs, float(reward), done, False, info

    def render(self):
        pass

    @property
    def n(self) -> int:
        return self._env.n

    @property
    def cost_matrix(self) -> np.ndarray:
        return self._env.cost_matrix

    @property
    def max_inventory(self) -> float:
        return self._env.max_inventory

    def _obs(self, state: dict) -> np.ndarray:
        inv = np.asarray(state["inventory"], dtype=np.float32) / self._env.max_inventory
        dem = np.asarray(state["demand"],    dtype=np.float32) / self._demand_scale
        t   = np.array([self._env._step_count / self._env.episode_length], dtype=np.float32)
        return np.concatenate([inv, dem, t])
