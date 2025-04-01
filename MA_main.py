import os
from ray import ray, tune
from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from pettingzoo.classic import pistonball_v6 

class PistonballMultiAgentEnv(MultiAgentEnv):
    def __init__(self):
        super().__init__(self,config=None)
        

config = (
    PPOConfig()
    .environment("Pendulum-v1")
    # Specify a simple tune hyperparameter sweep.
    .training(
        lr=tune.grid_search([0.0005, 0.0001]),
        gamma=tune.grid_search([0.99, 0.95]),
    )
    .framework(framework="torch",)
    .env_runners(num_env_runners=5,
                 num_envs_per_env_runner=2,
                 num_cpus_per_env_runner=2,
                 num_gpus_per_env_runner=0
                 )
    .learners(num_learners=4,
              num_gpus_per_learner=0.5,
              num_cpus_per_learner=1,)
)

ray.init(
    num_cpus=16,
    num_gpus=2,
    runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "1,2"}},
)
results_dir = os.path.abspath("./ray_results")
# Create a Tuner instance to manage the trials.
tuner = tune.Tuner(
    config.algo_class,
    param_space=config,

    run_config=tune.RunConfig(
        stop={"env_runners/episode_return_mean": -1100.0},
        storage_path=f"file://{results_dir}",
    ),
)
# Run the Tuner and capture the results.
results = tuner.fit()

