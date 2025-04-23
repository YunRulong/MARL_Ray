import os
import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
from pettingzoo.butterfly import pistonball_v6
from ray.train import CheckpointConfig


class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.BatchNorm2d(32),# 添加层归一化
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.BatchNorm2d(64),# 添加层归一化
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.BatchNorm2d(64),# 添加层归一化
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.LayerNorm(512),  # 添加层归一化
            nn.Dropout(0.1),    # 防止过拟合
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


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
        num_cpus=32,
        num_gpus=2,
        runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "1,3"}},
    )

    env_name = "pistonball_v6"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    config = (
        PPOConfig()
        .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            ) 
        .environment(env=env_name, clip_actions=True)
        .env_runners(
            num_env_runners=12,
            num_envs_per_env_runner=2,
            num_cpus_per_env_runner=2,
            num_gpus_per_env_runner=0
            )
        .resources(
            num_gpus=2,
            num_gpus_per_learner_worker=1,  # 每个 learner worker 的 GPU
            num_cpus_per_learner_worker=1,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")     
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.2,
            grad_clip=0.5,
            vf_clip_param=10.0,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            num_sgd_iter=10,
            )               
    )
    results_dir = os.path.abspath("./ray_results")
    tuner = tune.Tuner(
        config.algo_class,
        param_space=config,        
        run_config=tune.RunConfig(
            checkpoint_config=CheckpointConfig(checkpoint_frequency=10),
            stop={"timesteps_total": 50000000},            
            storage_path=f"file://{results_dir}",
        ),        
    )
    results = tuner.fit()
