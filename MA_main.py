import os
from ray import ray, tune
from ray.rllib.algorithms.ppo import PPOConfig
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from torch import nn
import torch

class ContinuousPolicyHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 均值输出层
        self.mu_head = nn.Linear(input_dim, 1)
        # 对数标准差层（可训练参数）
        self.log_std = nn.Parameter(torch.zeros(1))-0.5  # 初始化

    def forward(self, x):
         # 确保x和log_std在同一设备
        log_std = self.log_std.expand_as(x[:, :1]).to(x.device)  # 动态对齐设备
        return torch.cat([
            self.mu_head(x), 
            log_std  # 广播标准差到批量维度
        ], dim=1)

class CNNRLModuleV2(TorchRLModule, nn.Module):
    """符合新 API 标准的 CNN 模型"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 梯度监控相关属性
        self.gradient_info = {}        
        # 注意：RLModule 初始化时不立即构建网络，需在 setup() 中完成

    def setup(self):
        """核心网络构建方法"""
        # 从配置中获取参数（示例，可根据需要扩展）
        conv_filters = self.model_config.get("conv_filters", [
            (32, [8,8], 4),
            (64, [4,4], 2),
            (64, [3,3], 1)
        ])
        # 自动计算输入通道
        obs_shape = self.observation_space.shape
        in_channels = obs_shape[-1] if len(obs_shape) == 3 else 3
        
        # 构建卷积层
        conv_layers = []
        for out_channels, kernel, stride in conv_filters:
            conv_layers += [
                nn.Conv2d(in_channels, out_channels, kernel, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            in_channels = out_channels
            
        # 全连接部分
        self.cnn = nn.Sequential(*conv_layers)
        
        # 自动计算展平后的尺寸
        with torch.no_grad():
            test_input = torch.randn(1, *obs_shape).permute(0, 3, 1, 2)  # NHWC -> NCHW
            #print(f"测试输入形状: {test_input.shape}") 
            conv_out = self.cnn(test_input)
            #print(f"卷积输出形状: {conv_out.shape}")  # 验证输出形状
            #self._features_dim = conv_out.contiguous().view(1,-1).shape[1]
            self._features_dim = int(torch.prod(torch.tensor(conv_out.shape[1:])))
            #print(f"输出维度{self._features_dim},计算维度{conv_out.size(1) * conv_out.size(2) * conv_out.size(3)}")
            
        # 后续网络
        self.fc = nn.Sequential(
            nn.Linear(self._features_dim, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        
        # 输出头
        self.policy_head = ContinuousPolicyHead(512)
        #self.policy_head = nn.Linear(512, 1)
        self.value_head = nn.Linear(512, 1)

    def _common_forward(self, batch):
        """共享前向逻辑"""
        # 调整输入维度 (NHWC -> NCHW)
        x = batch["obs"].permute(0, 3, 1, 2).float()
        
        # 特征提取
        conv_features = self.cnn(x)
        flattened = conv_features.contiguous().view(conv_features.size(0), -1)
        fc_out = self.fc(flattened)

        # logits = self.policy_head(fc_out)
        # return {
        #     "logits": logits,
        #     "state_out": torch.zeros(fc_out.size(0), 0, device=fc_out.device),
        #     "actions": logits,
        #     "vf_preds": self.value_head(fc_out).squeeze(-1)
        # }
        # 获取动作分布参数
        action_params = self.policy_head(fc_out)  # 形状 [B,2]
        
        # 拆分均值和标准差
        action_mu = action_params[:, :1]
        action_log_std = action_params[:, 1:]
        
        # 采样动作
        actions = action_mu + torch.randn_like(action_mu) * action_log_std.exp()
        return {
        "action_dist_inputs": action_params,  # 必须包含此字段
        "logits": action_mu,                 # 兼容性字段
        "actions": actions,                  # 实际执行的动作
        "vf_preds": self.value_head(fc_out).squeeze(-1),
        "state_out": torch.zeros(fc_out.size(0), 0, device=fc_out.device)
    }

    def _forward_inference(self, batch, **kwargs):
        return self._common_forward(batch)

    def _forward_exploration(self, batch, **kwargs):
        return self._common_forward(batch)

    def _forward_train(self, batch, **kwargs):
        return self._common_forward(batch)

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
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.frame_stack_v1(env, 3)
    return env


if __name__ == "__main__":
    ray.init(
        num_cpus=16,
        num_gpus=1,
        runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "1"}},
        configure_logging=True, 
    )
    results_dir = os.path.abspath("./ray_results")
    register_env("pistonball_v6", lambda config: ParallelPettingZooEnv(env_creator(config)))   
    config = (
        PPOConfig()
        .environment(
            env= "pistonball_v6",
            env_config={"continuous": True},
            clip_actions=True,
        )
        .multi_agent(
            policies={"p0"},
            # All agents map to the exact same policy.
            policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
        )
        .framework("torch")
        .env_runners(num_env_runners=2,
                    num_envs_per_env_runner=1,
                    num_cpus_per_env_runner=4,
                    num_gpus_per_env_runner=0,
                    rollout_fragment_length=50,
                    sample_timeout_s=300,
                    )
        .learners(num_learners=1,
                num_gpus_per_learner=1,
                num_cpus_per_learner=1,
                )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={"p0": RLModuleSpec(
                    module_class=CNNRLModuleV2,
                    model_config={
                        "conv_filters": [
                            (32, [8,8], 4),
                            (64, [4,4], 2),
                            (64, [3,3], 1)
                        ],
                    },
                )},
            ),
        ) 
        .debugging(log_level="ERROR")       
        # Specify a simple tune hyperparameter sweep.
        .training(
            # lr=tune.grid_search([0.0005, 0.0001]),
            # gamma=tune.grid_search([0.99, 0.95]), 
            lr=0.0001,
            gamma=0.99,        
        )
    )
     
    # Create a Tuner instance to manage the trials.
    tuner = tune.Tuner(
        config.algo_class,
        param_space=config,
        run_config=tune.RunConfig(
            stop={"num_env_steps_sampled_lifetime":50000000},
            storage_path=f"file://{results_dir}",
            verbose=2,
        ),
    )
    # Run the Tuner and capture the results.
    results = tuner.fit()
    # env = env_creator({})
    
    # obs, _ = env.reset()
    # print("Observation space:", env.observation_space)
    # print("Action space:", env.action_space)
    # print("Initial observation:", obs)
