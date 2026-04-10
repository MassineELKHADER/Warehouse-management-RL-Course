"""
Demand models for the warehouse environment.
"""

import numpy as np


class PoissonDemand:
    def __init__(self, mean: float = 10.0):
        self.mean = mean

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.poisson(lam=self.mean, size=n).astype(np.float32)


class GaussianDemand:
    def __init__(self, mean: float = 10.0, std: float = 3.0):
        self.mean = mean
        self.std  = std

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        d = rng.normal(loc=self.mean, scale=self.std, size=n)
        return np.clip(d, 0, None).astype(np.float32)


class SeasonalDemand:
    """Sinusoidal demand with Poisson noise."""

    def __init__(self, mean: float = 10.0, amplitude: float = 5.0, period: int = 20):
        self.mean      = mean
        self.amplitude = amplitude
        self.period    = period
        self._t        = 0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        seasonal = self.mean + self.amplitude * np.sin(2 * np.pi * self._t / self.period)
        self._t += 1
        lam = max(seasonal, 1.0)
        return rng.poisson(lam=lam, size=n).astype(np.float32)

    def reset(self) -> None:
        self._t = 0


def make_demand_model(cfg: dict):
    model_type = cfg.get("env", {}).get("demand_model", "poisson")
    mean       = cfg.get("env", {}).get("demand_mean", 10.0)

    if model_type == "poisson":
        return PoissonDemand(mean=mean)
    elif model_type == "gaussian":
        return GaussianDemand(mean=mean, std=cfg.get("env", {}).get("demand_std", 3.0))
    elif model_type == "seasonal":
        return SeasonalDemand(mean=mean)
    else:
        raise ValueError(f"Unknown demand model: {model_type}")
