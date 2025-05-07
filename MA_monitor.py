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
from ray.tune import CheckpointConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np
from datetime import datetime
from ray.rllib.utils.annotations import (OldAPIStack)
        
class GradientMonitorCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.count = 0#保存计数
        self.save_length = 16000#保存频率
        self.gradient_hooks = []
        self.grad_continer = []
        self.grad_dir = "/home/data/yunrulong/MARL/grad_data"  # 存储目录 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.grad_save_dir = os.path.join(self.grad_dir,timestamp)  # 存储目录 
        os.makedirs(self.grad_save_dir, exist_ok=True)  # 创建目录
        self.train_data_file = "train_data.csv"  # 文件名       
        
    @OldAPIStack
    def on_algorithm_init(self, *, algorithm, metrics_logger = None, **kwargs,):
        if os.environ.get("RESTORE_MODE") == "1":
            return  # 恢复时跳过初始化
        self.header = [
            "iteration", 
            "timesteps_total",
            "num_env_steps_sampled",
            "num_env_steps_trained",
            "episode_reward_max",
            "episode_reward_min",
            "episode_reward_mean", 
            "total_loss",
            "policy_loss",
            "vf_loss",
            "entropy",
            "cur_kl_coeff",
            "vf_explained_var",            
        ]  
        new_dir = algorithm.logdir
        new_dir = self._extract_and_combine_paths(new_dir)
        self.save_dir = os.path.join('/home/data/yunrulong/MARL/ray_results', new_dir) 
        self.train_data_path = os.path.join(self.save_dir, self.train_data_file)
        if not os.path.exists(self.train_data_path):
            with open(self.train_data_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # 写入列标题
                writer.writerow(self.header) 
        # 仅在训练开始时创建 CSV 文件     
    @OldAPIStack
    def on_train_result(self, *, algorithm, metrics_logger = None, result, **kwargs):
        # result: dict_keys(['custom_metrics', 'episode_media', 'info', 'env_runners', 'num_healthy_workers', 'num_in_flight_async_sample_reqs', 
        # 'num_remote_worker_restarts', 'num_agent_steps_sampled', 'num_agent_steps_trained', 'num_env_steps_sampled', 'num_env_steps_trained', 
        # 'num_env_steps_sampled_this_iter', 'num_env_steps_trained_this_iter', 'num_env_steps_sampled_throughput_per_sec', 'num_env_steps_trained_throughput_per_sec', 
        # 'timesteps_total', 'num_env_steps_sampled_lifetime', 'num_agent_steps_sampled_lifetime', 'num_steps_trained_this_iter', 'agent_timesteps_total', 
        # 'timers', 'counters', 'done', 'training_iteration', 'trial_id', 'date', 'timestamp', 'time_this_iter_s', 'time_total_s', 'pid', 'hostname', 'node_ip',
        #  'config', 'time_since_restore', 'iterations_since_restore', 'perf'])
        # result: {'learner': {'default_policy': {'custom_metrics': {}, 'learner_stats': {'cur_kl_coeff': 0.2, 'cur_lr': 1e-05, 'total_loss': 0.40663162417709825,
        #  'policy_loss': 0.0037339560501277445, 'vf_loss': 2.2746329824253917, 'vf_explained_var': 3.5456456243991853e-05, 'kl': 0.007701946182351094, 
        # 'entropy': 1.6730096450448035, 'entropy_coeff': 0.1}, 'model': {}, 'num_grad_updates_lifetime': 2000.5, 'diff_num_grad_updates_vs_sampler_policy': 399.5}}, 
        # 'num_env_steps_sampled': 1536, 'num_env_steps_trained': 1536, 'num_agent_steps_sampled': 30720, 'num_agent_steps_trained': 30720}
        # print("result:",result.keys())
        # print("result:",result['info'])         
        # 基础训练指标
        base_metrics = {
            "iteration": result["training_iteration"],
            "timesteps_total": result["timesteps_total"],
            "num_env_steps_sampled": result["info"].get("num_env_steps_sampled", 0),
            "num_env_steps_trained": result["info"].get("num_env_steps_trained", 0),
            'episode_reward_max' : result["env_runners"].get("episode_reward_max", 0),
            'episode_reward_min' : result["env_runners"].get("episode_reward_min", 0),
            'episode_reward_mean' : result["env_runners"].get("episode_reward_mean", 0),
        }

        policy_metrics = {}
        learner_data = result["info"].get("learner", {})
        for policy_id, policy_info in learner_data.items():
            if isinstance(policy_info, dict):
                stats = policy_info.get("learner_stats", {})
                print(f"策略 {policy_id} 的统计信息: {stats}")
                policy_metrics.update({
                    "total_loss": stats.get("total_loss", 0.0),
                    "policy_loss": stats.get("policy_loss", 0.0),
                    "vf_loss": stats.get("vf_loss", 0.0),
                    "entropy": stats.get("entropy", 0.0),
                    "cur_kl_coeff": stats.get("cur_kl_coeff", 0.0),
                    "vf_explained_var": stats.get("vf_explained_var", 0.0),
                })
        
        # 合并指标字典
        valid_metrics = {**base_metrics, **policy_metrics}        
        # 按固定顺序生成数据行（自动填充缺失值）
        row_data = [valid_metrics.get(key, 0.0) for key in self.header]  # 关键修改点：顺序控制
        # 追加写入数据
        with open(self.train_data_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

    def _extract_and_combine_paths(self,original_path):
        # 分割路径为组件列表
        path_parts = original_path.split(os.sep)        
        # 定位目标组件（索引从后往前计算）
        target_part1 = None
        target_part2 = None        
        # 逆向遍历查找关键组件
        for part in reversed(path_parts):
            if part.startswith("PPO_") and "_00000_0_" in part:
                target_part2 = part
            elif part.startswith("PPO_") and not any(c.isalpha() for c in part[4:]):
                target_part1 = part
                break  # 找到第一个PPO时间戳后停止        
        # 安全校验
        if not target_part1 or not target_part2:
            raise ValueError("路径结构不符合预期格式")        
        # 拼接最终路径
        return os.path.join(target_part1, target_part2)

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
            if len(self.grad_continer) > self.save_length:                
                try:   
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 格式: 年月日_时分秒_微秒      # 取第一个样本的时间步
                    file_name = f"grad_timestamp_{timestamp}.csv"
                    file_path = os.path.join(self.grad_save_dir, file_name)
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
                                i%12,  # Layer ID
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
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
            )
        # self.model = nn.Sequential(
        #     nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
        #     nn.BatchNorm2d(32),# 添加层归一化
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
        #     nn.BatchNorm2d(64),# 添加层归一化
        #     nn.LeakyReLU(0.1),
        #     nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
        #     nn.BatchNorm2d(64),# 添加层归一化
        #     nn.LeakyReLU(0.1),
        #     nn.Flatten(),
        #     nn.Linear(3136, 512),
        #     nn.LayerNorm(512),  # 添加层归一化
        #     nn.Dropout(0.1),    # 防止过拟合
        #     nn.LeakyReLU(0.1),
        #     )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)
        for name, param in self.model.named_parameters():
                print(f"层名: {name},\t\t\t 形状: {param.shape}")

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
            lr=0.00001,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            vf_clip_param=10.0,
            #entropy_coeff=0.1,#4.28中午
            entropy_coeff=0.01,#4.28下午
            vf_loss_coeff=0.25,
            num_epochs=10,
            model={"custom_model": "CNNModelV2"},
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
