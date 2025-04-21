import os
from ray import ray, tune
from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.rl_modules.classes.tiny_atari_cnn_rlm import TinyAtariCNN
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
def env_creator(args):
    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=900,
    )
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.frame_stack_v1(env, 3)
    return env


if __name__ == "__main__":
    ray.init(
        num_cpus=16,
        num_gpus=2,
        runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "1,3"}},
    )
    results_dir = os.path.abspath("./ray_results")
    config = (
        PPOConfig()
        .environment(
            env= "pistonball_v6",
            env_config={},
            clip_actions=True,
        )
        .framework("torch")
        .env_runners(num_env_runners=5,
                    num_envs_per_env_runner=2,
                    num_cpus_per_env_runner=2,
                    num_gpus_per_env_runner=0,
                    rollout_fragment_length=200,
                    )
        .learners(num_learners=2,
                num_gpus_per_learner=1,
                num_cpus_per_learner=1,
                )
        .rl_module(
                # Plug-in our custom RLModule class.
                rl_module_spec=RLModuleSpec(
                    module_class=TinyAtariCNN,
                    # Feel free to specify your own `model_config` settings below.
                    # The `model_config` defined here will be available inside your
                    # custom RLModule class through the `self.model_config`
                    # property.
                    model_config={
                        "conv_filters": [
                            # num filters, kernel wxh, stride wxh, padding type
                            [16, 4, 2, "same"],
                            [32, 4, 2, "same"],
                            [256, 11, 1, "valid"],
                        ],
                    },
                ),
            )
        # Specify a simple tune hyperparameter sweep.
        .training(
            lr=tune.grid_search([0.0005, 0.0001]),
            gamma=tune.grid_search([0.99, 0.95]),       
        )
    )
    register_env("pistonball_v6", lambda config: ParallelPettingZooEnv(env_creator(config)))    
    # Create a Tuner instance to manage the trials.
    tuner = tune.Tuner(
        config.algo_class,
        param_space=config,
        run_config=tune.RunConfig(
            stop={"timesteps_total":50000000},
            storage_path=f"file://{results_dir}",
            verbose=2,
        ),
    )
    # Run the Tuner and capture the results.
    results = tuner.fit()

