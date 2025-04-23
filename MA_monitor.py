import os
import ray
import csv
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
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np
from datetime import datetime
from ray.rllib.utils.annotations import (OldAPIStack,)
class GradientMonitorCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.count = 0#保存计数
        self.gradient_hooks = []
        self.grad_continer = []
        self.grad_dir = "/home/data/yunrulong/MARL/grad_data"  # 存储目录  
        os.makedirs(self.grad_dir, exist_ok=True)

    def grad_hook(self,grad, container):
        container.append(grad.clone().cpu().numpy())
        #print(f"捕获到梯度: shape={grad.shape}, mean={grad.mean()}")  
    @OldAPIStack
    def on_learn_on_batch(self, *, policy, train_batch, result, **kwargs):
        # train_batch: dict_keys(['obs', 'new_obs', 'actions', 'rewards', 
        # 'terminateds', 'truncateds', 'eps_id', 'unroll_id', 'agent_index', 't', 'vf_preds', 
        # 'action_dist_inputs', 'action_logp', 'values_bootstrapped', 'advantages', 'value_targets', 'infos'])
        #清除之前的钩子
        for hook in self.gradient_hooks:
            hook.remove()
        self.gradient_hooks.clear()
        self.count += 1
        # 仅处理 PyTorch 策略
        if policy.framework == "torch":
            model = policy.model
            # 为每个参数注册梯度钩子
            for param in model.parameters():
                if param.requires_grad and self.count %10 == 0:
                    hook = param.register_hook(lambda g: self.grad_hook(g, self.grad_continer))                    
                    self.gradient_hooks.append(hook)               
            # 保存梯度数据
            if len(self.grad_continer) > 16000:                
                try:   
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 格式: 年月日_时分秒_微秒      # 取第一个样本的时间步
                    file_name = f"grad_timestamp_{timestamp}.csv"
                    file_path = os.path.join(self.grad_dir, file_name)
                    print("file_path:",file_path)
                    print("gradients:",len(self.grad_continer))
                    with open(file_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)                                             
                        # 写入列标题
                        writer.writerow([
                            'Layer', 'Shape', 
                            'Mean', 'Max', 'Min',
                            'Norm', 'Std', 'NonZero%' 
                        ])                        
                        # 写入各层梯度统计
                        for i, grad in enumerate(self.grad_continer):
                            grad_array = grad.ravel()                            
                            # 计算统计指标
                            stats = [
                                i,  # Layer ID
                                str(grad.shape),  # Shape
                                np.mean(grad_array),  # Mean
                                np.max(grad_array),  # Max
                                np.min(grad_array),  # Min
                                np.linalg.norm(grad_array),  # Norm
                                np.std(grad_array),  # Std
                                (np.count_nonzero(grad_array) / grad_array.size)  # NonZero%
                           ]                            
                            # 格式化为科学计数法（保留4位小数）
                            formatted_stats = [
                                stats[0], 
                                stats[1],
                                "{:.4e}".format(stats[2]),
                                "{:.4e}".format(stats[3]),
                                "{:.4e}".format(stats[4]),
                                "{:.4e}".format(stats[5]),
                                "{:.4e}".format(stats[6]),
                                "{:.2%}".format(stats[7])
                            ]                            
                            writer.writerow(formatted_stats)                        
                    print(f"CSV文件已保存到 {file_path}")   
                    self.grad_continer.clear()  # 清空容器以便下次使用
                except Exception as e:
                    print(f"保存CSV失败: {str(e)}")
                    if os.path.exists(file_path):
                        os.remove(file_path)  # 删除可能损坏的文件        

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
        max_cycles=1000,
    )
    
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.frame_stack_v1(env, 3)
    return env

if __name__ == "__main__":
    ray.init(
        num_cpus=32,
        num_gpus=2,
        runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "0,3"}},
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
            num_env_runners=15,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=2,
            num_gpus_per_env_runner=0
            )
        .resources(
            num_gpus=2,
            num_gpus_per_learner_worker=1,  # 每个 learner worker 的 GPU
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")    
        .callbacks(GradientMonitorCallback) 
        .training(
            train_batch_size=512,
            lr=0.0001,
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
