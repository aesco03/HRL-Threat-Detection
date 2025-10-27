 import ray
 from ray.rllib.algorithms.ppo import PPOConfig
 from environments.base_file_analysis_env import FileAnalysisEnv
## FLAT DQN or DDQN RL ALGO
 if __name__ == "__main__":
     ray.init()
     config = (
         PPOConfig()
         .framework("torch")
         .environment(
             FileAnalysisEnv,
             env_config={
                 "dataset_path": "./data/files/",
                 "labels_csv": "./data/labels.csv",
                 "feature_dim": 64,
                "metadata_dim": 16,
                "max_steps": 20,
                "step_cost": 0.01,
                "pos_fraction": 0.5,
            },
        )
    )
     algo = config.build()
     for i in range(20):
         result = algo.train()
         print(i, result.get("episode_reward_mean"))
