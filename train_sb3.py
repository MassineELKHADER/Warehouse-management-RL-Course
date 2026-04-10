"""
SB3 training script for the warehouse redistribution environment.

Usage:
    python train_sb3.py --algo ppo --config small --seed 42
    python train_sb3.py --algo sac --config small --seed 42
    python train_sb3.py --algo ppo --config medium --seed 42 --no-wandb
    python train_sb3.py --algo ppo --config small --seed 42 --timesteps 200000
"""

import argparse
import os

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config(name: str) -> dict:
    with open(os.path.join("configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    if name != "default":
        with open(os.path.join("configs", f"{name}.yaml")) as f:
            override = yaml.safe_load(f)
        for key, val in override.items():
            if key == "_base_":
                continue
            if isinstance(val, dict) and key in cfg:
                cfg[key].update(val)
            else:
                cfg[key] = val
    return cfg


class WandbMetricsCallback:
    """Logs episode metrics to WandB every eval_every episodes."""

    def __init__(self, wandb_run, env, eval_every: int = 50, n_eval_eps: int = 3):
        self._run        = wandb_run
        self._env        = env
        self._eval_every = eval_every
        self._n_eval_eps = n_eval_eps
        self._ep_count   = 0

    def on_episode_end(self, model, episode: int = 0):
        rewards, costs, sats, inv_stds = [], [], [], []
        last_T = None

        for _ in range(self._n_eval_eps):
            obs, _ = self._env.reset()
            done   = False
            ep_reward, ep_costs, ep_sats, ep_invs = 0.0, [], [], []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self._env.step(action)
                done = terminated or truncated
                ep_reward     += reward
                ep_costs.append(info["transport_cost"])
                ep_sats.append(info["demand_satisfaction"])
                ep_invs.append(info["inventory"])
                last_T = action.reshape(self._env.n, self._env.n)
            rewards.append(ep_reward)
            costs.append(np.sum(ep_costs))
            sats.append(np.mean(ep_sats))
            inv_stds.append(float(np.std(np.stack(ep_invs).mean(axis=0))))

        metrics = {
            "eval/episode_reward":      float(np.mean(rewards)),
            "eval/transport_cost":      float(np.mean(costs)),
            "eval/demand_satisfaction": float(np.mean(sats)),
            "eval/inventory_std":       float(np.mean(inv_stds)),
        }
        self._run.log(metrics, step=episode)

        if last_T is not None:
            try:
                from visualizations.shipping_graph_viz import plot_shipping_graph
                import wandb
                fig = plot_shipping_graph(
                    last_T,
                    self._env.cost_matrix,
                    hub_fraction=0.2,
                    title=f"Shipping policy (ep {episode})",
                )
                self._run.log({"eval/shipping_graph": wandb.Image(fig)}, step=episode)
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass


from stable_baselines3.common.callbacks import BaseCallback


class _EpisodeCallback(BaseCallback):
    """Calls WandbMetricsCallback every eval_every episodes."""

    def __init__(self, metrics_cb, episode_length: int, eval_every: int, n_envs: int):
        super().__init__()
        self._mcb            = metrics_cb
        self._episode_length = episode_length
        self._eval_every     = eval_every
        self._n_envs         = n_envs
        self._eval_interval  = eval_every * episode_length

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_interval == 0:
            ep = self.n_calls // self._episode_length
            self._mcb.on_episode_end(self.model, episode=ep)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",       default="ppo",   choices=["ppo", "sac"])
    parser.add_argument("--config",     default="small", choices=["default", "small", "medium", "large"])
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--timesteps",  type=int, default=None,
                        help="Total env timesteps (default: n_episodes * episode_length)")
    parser.add_argument("--no-wandb",   action="store_true")
    parser.add_argument("--n-envs",     type=int, default=8,
                        help="Number of parallel envs")
    parser.add_argument("--lambda-penalty", type=float, default=None,
                        help="Override lambda_penalty from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.lambda_penalty is not None:
        cfg["env"]["lambda_penalty"] = args.lambda_penalty
    episode_length = cfg["env"]["episode_length"]
    n_episodes     = cfg["training"]["n_episodes"]
    eval_every     = cfg["training"]["eval_every"]
    total_steps    = args.timesteps or (n_episodes * episode_length)

    np.random.seed(args.seed)

    from env.warehouse_gym_env import WarehouseGymEnv
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    env = make_vec_env(
        lambda: WarehouseGymEnv(cfg, seed=args.seed),
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
    )
    # normalize rewards only; obs is already in [0,1]
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = WarehouseGymEnv(cfg, seed=args.seed + 1000)

    lam      = cfg["env"]["lambda_penalty"]
    run_name = f"sb3_{args.algo}_{args.config}_lam{lam}_s{args.seed}"
    print(f"Training [{run_name}] for {total_steps} timesteps")

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "warehouse-rl"),
            entity=os.getenv("WANDB_ENTITY", None),
            name=run_name,
            group=f"sb3_{args.algo}_lambda_sweep",
            tags=["sb3", args.algo, args.config, f"seed_{args.seed}", f"lambda_{cfg['env']['lambda_penalty']}"],
            config={
                "algo":           args.algo,
                "scenario":       args.config,
                "seed":           args.seed,
                "n_warehouses":   cfg["env"]["n_warehouses"],
                "max_inventory":  cfg["env"]["max_inventory"],
                "episode_length": episode_length,
                "total_steps":    total_steps,
                "lambda_penalty": cfg["env"]["lambda_penalty"],
            },
        )

    # clip_range decays linearly from 0.2 to 0.0 over training
    def linear_schedule(initial_value):
        def func(progress_remaining):
            return progress_remaining * initial_value
        return func

    if args.algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=episode_length,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=linear_schedule(0.2),
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [128, 128], "log_std_init": -1.0},
            seed=args.seed,
            device="cpu",
            verbose=1,
        )
    elif args.algo == "sac":
        from stable_baselines3 import SAC
        model = SAC(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            learning_starts=episode_length * 10,
            policy_kwargs={"net_arch": [128, 128]},
            seed=args.seed,
            device="cpu",
            verbose=1,
        )

    callbacks = []
    if wandb_run:
        mcb = WandbMetricsCallback(
            wandb_run, eval_env,
            eval_every=eval_every,
            n_eval_eps=10,
        )
        callbacks.append(_EpisodeCallback(mcb, episode_length, eval_every, args.n_envs))

    model.learn(total_timesteps=total_steps, callback=callbacks or None)

    os.makedirs("outputs", exist_ok=True)
    ckpt_path = f"outputs/{run_name}.zip"
    model.save(ckpt_path)
    print(f"Saved: {ckpt_path}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
