from stable_baselines3 import HerReplayBuffer, SAC
from sb3_contrib import ARS, QRDQN, TQC, TRPO, RecurrentPPO
import gymnasium
import panda_gym
import numpy as np
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from gymnasium import spaces
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    unwrap_vec_normalize,
)

from model import TQCEnvSwitchWrapper
from utils import get_push_env


env_id = "PandaPush-v3"
log_dir = './panda_push_v3_tensorboard/'

# env for train
env1 = get_push_env()
env2 = get_push_env(1.0, 0.001,2.0)
env3 = get_push_env(1.0, 0.001,3.0)
env4 = get_push_env(1.0, 0.001,4.0)
# env for test
env5 = get_push_env(1.0, 0.001,10)

train_env = DummyVecEnv([env1,env2,env3,env4])
test_env = DummyVecEnv([env5,env5,env5,env5])

# SAC train model
model = TQCEnvSwitchWrapper(policy = "MultiInputPolicy",
                        env = train_env,
                        batch_size=2048,
                        gamma=0.95,
                        learning_rate=1e-4,
                        train_freq=64,
                        gradient_steps=64,
                        tau=0.05,
                        replay_buffer_class=HerReplayBuffer,
                        replay_buffer_kwargs=dict(
                            n_sampled_goal=4,
                            goal_selection_strategy="future",
                        ),
                        policy_kwargs=dict(
                            net_arch=[512, 512, 512],
                            n_critics=2,
                        ),
                        learning_starts = 1000,
                        verbose=1,
                        tensorboard_log=log_dir)

model.learn(total_timesteps=2_000_000,progress_bar=True)
train_env.close()

# save_model
model_name = "TQC-PandaPush-v3"
model.save(model_name)
# train_env.save(f"{model_name}_vec_normalize.pkl")

# switch environment
model.reset_env(test_env)

# SAC test model
model.learn(total_timesteps=2_000_000,progress_bar=True)
test_env.close()