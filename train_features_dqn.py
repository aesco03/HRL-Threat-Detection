import ray
import yaml
from ray.rllib.algorithms.dqn import DQNConfig
from envs.precomputed_features_env import PrecomputedFeaturesEnv

if __name__ == "__main__":
    ray.init()
    with open("./configs/feature_groups.yaml", "r") as fh:
        cfg_fg = yaml.safe_load(fh)
    env_cfg = {
        "parquet_path": "./data/processed/labeled/cleaned_pdfmalware2022.parquet",
        "label_col": cfg_fg.get("label_col", "label"),
        "meta_cols": cfg_fg.get("meta_cols", []),
        "pdf_cols": cfg_fg.get("pdf_cols", []),
        "text_cols": cfg_fg.get("text_cols", []),
        "image_cols": cfg_fg.get("image_cols", []),
        "group_dim": int(cfg_fg.get("group_dim", 16)),
        "max_steps": int(cfg_fg.get("max_steps", 20)),
        "step_cost": float(cfg_fg.get("step_cost", 0.01)),
        "pos_fraction": cfg_fg.get("pos_fraction", 0.5),
    }
    config = (
        DQNConfig()
        .framework("torch")
        .environment(PrecomputedFeaturesEnv, env_config=env_cfg)
        .training(double_q=True, dueling=True)
    )
    algo = config.build()
    for i in range(20):
        result = algo.train()
        print(i, result.get("episode_reward_mean"))
