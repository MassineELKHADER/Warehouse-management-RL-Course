# Warehouse RL

Two complementary approaches to multi-warehouse inventory redistribution with reinforcement learning.

---

## Discrete setting

A 3-warehouse hub-and-spoke model with a fixed action set (imports + transfers).
Solved with value iteration, Q-learning, and DQN.

**Run:**
Open `discrete_warehouse_rl.ipynb` and run all cells.

The environment is in `discrete_warehouse_env.py`.

**Dependencies:** numpy, matplotlib, scipy, torch, pandas, tqdm

---

## Continuous setting

N warehouses, continuous action space: an N x N transport matrix.
Trained with PPO (and optionally SAC) via Stable Baselines 3.

**Install:**
```
pip install -r requirements.txt
```

Copy `.env` and fill in your WandB credentials:
```
WANDB_API_KEY=...
WANDB_PROJECT=warehouse-rl
WANDB_ENTITY=your_entity
WANDB_MODE=online
```

**Train:**
```
python train_sb3.py --algo ppo --config small --seed 42
python train_sb3.py --algo ppo --config medium --seed 42 --no-wandb
python train_sb3.py --algo sac --config small --seed 42
```

Available configs: `small` (N=4), `medium` (N=9), `large` (N=16).

Checkpoints are saved to `outputs/`.

**Environment:**
- `env/warehouse_env.py` -- core environment
- `env/warehouse_gym_env.py` -- Gymnasium wrapper for SB3
- `env/cost_matrix.py` -- hub-and-spoke cost matrix
- `env/demand_models.py` -- Poisson / Gaussian / Seasonal demand

**Visualization:**
`visualizations/shipping_graph_viz.py` draws the learned shipping policy as a directed graph. Called automatically during training if WandB is enabled.
