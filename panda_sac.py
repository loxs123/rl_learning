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
from stable_baselines3.common.evaluation import evaluate_policy

from model import SACEnvSwitchWrapper
from utils import get_push_env, get_pick_and_place_env, get_push_dense_env, get_push_joints_env, get_push_joints_dense_env, get_reach_env, get_slide_env, get_stack_env
import argparse

import numpy

def make_env(name,lateral_friction=1.0,spinning_friction=0.001,mass=1.0):
    print(name)
    if name == "PandaPush-v3":
        print("This is PandaPush-v3 Env, Welcome!")
        return get_push_env(lateral_friction, spinning_friction, mass)
    elif name == "PandaPickAndPlace-v3":
        print("This is PandaPickAndPlace-v3 Env, Welcome!")
        return get_pick_and_place_env(lateral_friction, spinning_friction, mass)
    elif name == "PandaPushDense-v3":
        print("This is PandaPushDense-v3 Env, Welcome!")
        return get_push_dense_env(lateral_friction, spinning_friction, mass)
    elif name == "PandaPushJoints-v3":
        print("This is PandaPushJoints-v3 Env, Welcome!")
        return get_push_joints_env(lateral_friction, spinning_friction, mass)
    elif name == "PandaPushJointsDense-v3":
        print("This is PandaPushJointsDense-v3 Env, Welcome!")
        return get_push_joints_dense_env(lateral_friction, spinning_friction, mass)
    elif name == "PandaReach-v3":
        print("This is PandaReach-v3 Env, Welcome!")
        return get_reach_env(lateral_friction, spinning_friction, mass)
    elif name == "PandaSlide-v3":
        print("This is PandaSlide-v3 Env, Welcome!")
        return get_slide_env(lateral_friction, spinning_friction, mass)
    elif name == "PandaStack-v3":
        print("This is PandaStack-v3 Env, Welcome!")
        return get_stack_env(lateral_friction, spinning_friction, mass)
    else:
        print("Unkown Environment!! ")
    

def train(args):
    env_id = args.domain_name
    # log_dir = './panda_push_v3_tensorboard/'
    log_dir = './' + args.domain_name + '_tensorboard/'

    if args.test_mass == 1: # friction experiment
        if args.test_lateral_friction == 1.0: # spinning_friction
            env_int = numpy.random.uniform(low=args.random_float[0], high=args.random_float[1], size=4)
            print("Random Spinning Friciton Values:", env_int)
            # env for train
            env1 = make_env(env_id, 1.0, env_int[0], 1.0)
            env2 = make_env(env_id, 1.0, env_int[1], 1.0)
            env3 = make_env(env_id, 1.0, env_int[2], 1.0)
            env4 = make_env(env_id, 1.0, env_int[3], 1.0)
        else:
            env_int = numpy.random.uniform(low=1.0, high=4.0, size=4)
            print("Random Lateral Friciton Values:", env_int)
            # env for train
            env1 = make_env(env_id, env_int[0], 0.001, 1.0)
            env2 = make_env(env_id, env_int[1], 0.001, 1.0)
            env3 = make_env(env_id, env_int[2], 0.001, 1.0)
            env4 = make_env(env_id, env_int[3], 0.001, 1.0)
    else:
        env_int = numpy.random.randint(low=args.random_int[0], high=args.random_int[1], size=4, dtype='l')
        print("Random Mass Values:", env_int)
        # env for train
        env1 = make_env(env_id, 1.0, 0.001, env_int[0])
        env2 = make_env(env_id, 1.0, 0.001, env_int[1])
        env3 = make_env(env_id, 1.0, 0.001, env_int[2])
        env4 = make_env(env_id, 1.0, 0.001, env_int[3])
    # env for test
    env5 = make_env(env_id, args.test_lateral_friction, args.test_spinning_friction, args.test_mass)
    train_env = DummyVecEnv([env1,env2,env3,env4])
    test_env = DummyVecEnv([env5,env5,env5,env5])
    
    # SAC train model
    model = SACEnvSwitchWrapper(policy = "MultiInputPolicy",
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

    model.learn(total_timesteps=args.time_step,progress_bar=True)
    train_env.close()

    # save_model
    if args.test_mass == 1: # friction experiment
        if args.test_lateral_friction == 1.0: # spinning_friction
            # save_model
            model_name = "SAC-" + str(args.domain_name + "-model-spinning-friction" + str(args.test_spinning_friction))
            model.save(model_name)
            model.save_replay_buffer('SAC-' + str(args.domain_name) + '-buffer' + "-model-spinning-friction" + str(args.test_spinning_friction))
        else:
            # save_model
            model_name = "SAC-" + str(args.domain_name + "-model-lateral-friction" + str(args.test_lateral_friction))
            model.save(model_name)
            model.save_replay_buffer('SAC-' + str(args.domain_name) + '-buffer' + "-model-lateral-friction" + str(args.test_lateral_friction))
    else:
        # save_model
        model_name = "SAC-" + str(args.domain_name + "-model-mass" + str(args.test_mass))
        model.save(model_name)
        model.save_replay_buffer('SAC-' + str(args.domain_name) + '-buffer' + "-model-mass" + str(args.test_mass))
    # train_env.save(f"{model_name}_vec_normalize.pkl")

    # switch environment
    model2 = SACEnvSwitchWrapper.load(model_name, env=test_env)
    model2.eval_env = True

    model2.learn(total_timesteps=args.time_step,progress_bar=True)

    test_env.close()


def test(args):
    # env for train
    env1 = make_env()
    env2 = make_env(env_id, 1.0, 0.001,2.0)
    env3 = make_env(env_id, 1.0, 0.001,3.0)
    env4 = make_env(env_id, 1.0, 0.001,4.0)
    # env for test
    env5 = make_env(env_id, 1.0, 0.001,10)

    train_env = DummyVecEnv([env1,env2,env3,env4])
    test_env = DummyVecEnv([env5,env5,env5,env5])

    model = SACEnvSwitchWrapper.load('SAC-PandaPush-v3',env=train_env)
    # model.env = train_env
    # train_mean_reward, train_std_reward = evaluate_policy(model, train_env, 100)
    test_mean_reward, test_std_reward = evaluate_policy(model, test_env, 100)

    # print(f"Train Mean reward = {train_mean_reward:.2f} +/- {train_std_reward:.2f}")
    print(f"Test Mean reward = {test_mean_reward:.2f} +/- {test_std_reward:.2f}")

    train_env.close()
    test_env.close()


def retrain(args):
    # env for test
    env5 = make_env(env_id, 1.0, 0.001,50)
    test_env = DummyVecEnv([env5,env5,env5,env5])

    model = SACEnvSwitchWrapper.load('SAC-PandaPush-v3',env=test_env)
    model.eval_env = True

    model.learn(total_timesteps=args.time_step,progress_bar=True)

    test_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', default='PandaPush-v3')
    parser.add_argument('--random_int', default=[1, 5], nargs='+', type=int)
    parser.add_argument('--random_float', default=[0.001, 0.01], nargs='+', type=float)
    parser.add_argument('--test_mass', default=1.0, type=int)
    parser.add_argument('--time_step', default=800000, type=int)
    parser.add_argument('--test_spinning_friction', default=0.001, type=float)
    parser.add_argument('--test_lateral_friction', default=1.0, type=float)
    args = parser.parse_args()

    train(args)