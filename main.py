# imports
import pickle
import time
import torch
from envs.modified_envs import *
from envs.configs import *
from click import progressbar
from torch import nn
import gymnasium as gym
import argparse
import numpy as np
import pandas as pd
import os
import random
from stable_baselines3.common.env_util import make_vec_env
from utilities.shield_gen import Shield
from gymnasium import spaces, register
#from utilities.ppo import PPO
import highway_env
from stable_baselines3 import PPO
#import ray
from gymnasium.wrappers import FlattenObservation
import warnings
from stable_baselines3.common.env_util import DummyVecEnv
#from envs.stable3_env import *
from utilities.algo_configs import PPOConfig, ShieldConfig,ShieldCallbackConfig
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
from stable_baselines3.common.logger import configure, CSVOutputFormat
from algorithms.shieldPPO_sb3 import *
from envs.setup import *
from stable_baselines3.common.callbacks import EvalCallback



def save_args_to_file(args, filename="args.txt"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")

def setup_device(use_gpu):
    device = torch.device('cpu')
    if use_gpu and (torch.cuda.is_available()):
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
    print(f"Device is set up to {device}")
    return device

def setup_seed(env, seed):
    torch.manual_seed(seed)
    # CUDA seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.action_space.seed(seed)
    print(f"Seed is set to {seed}")

def get_discounted_costs_episode_samples(episode, discount_factor_cost):
    """
    given episode and discount_factor_costs, returns the episode samples with discounted cost
    """
    episode_samples = []
    # reverse iterating through the episode - from end to beginning
    d_cost = 0
    for step in zip(reversed(episode)):
        state, action, cost, is_terminal = step[0]
        if is_terminal:
            d_cost = 0
        d_cost = cost + (discount_factor_cost * d_cost)
        d_cost_tensor = torch.tensor(d_cost)
        episode_samples.insert(0, (state, action, d_cost_tensor))
    return episode_samples


def prepare_shield_input(state,action,cost):
    input_features = torch.cat((state.view(-1),action.float()))  # Concat state and action
    return input_features


class AgentFactory:
    @staticmethod
    def create_agent(agent_type: str, config: dict, env, shield, shield_callback_config,device):
        agents = {
            "PPO": lambda: PPO(
                device = device,
                policy=config['ppo_config'].policy,
                env=env,
                learning_rate=config['ppo_config'].learning_rate,
                n_steps=config['ppo_config'].n_steps,
                n_epochs=config['ppo_config'].n_epochs,
                gamma=config['ppo_config'].gamma,
                clip_range=config['ppo_config'].clip_range,
                verbose=config['ppo_config'].verbose
            ),
        "ShieldPPO": lambda:ShieldMaskablePPO(device=device, ppo_config=config['ppo_config'],
                                              env=env, shield=shield, shield_callback_config=shield_callback_config)

        }

        return agents.get(agent_type, lambda: None)()


def get_arguments(arguments=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--select_action_algo", default="PPO", help="algorithm to use when selecting action:  PPO | Random | ShieldPPO | RuleBasedShieldPPO (REQUIRED)")
    parser.add_argument("--env", default="highway-fast-v0",
                        help="names of the environment to train on")
    parser.add_argument("--max_ep_len", type=int, default=500,
                        help="max timesteps in one episode. In cartpole-v0 env it should be 200.")
    parser.add_argument("--max_training_timesteps", type=int, default=int(1e6),
                        help="break training loop after 'max_training_timesteps' timesteps.")

    # running setups
    parser.add_argument("--render", type=bool, default=False,
                        help="render environment. default is False.")
    parser.add_argument("--use_gpu", type=bool, default=False,
                        help="If true, use gpu instead of cpu as device for PPO and ShieldPPO (Shield+MaskablePPO) algorithms.")
    parser.add_argument("--cpu", type=int, default=4,
                        help="Number of cpus")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Number of envs")
    parser.add_argument("--seed", type=int, default=0,
                        help="defines random seed. default is 0.")
    parser.add_argument("--observation_type", type=str, default="Kinematics",
                        choices=["Kinematics", "OccupancyGrid", "TimeToCollision"],
                        help="Defines the observation type of the highway environment.")

    # logs & frequencies
    parser.add_argument("--save_model_freq", type=int, default=int(5000),
                        help="save trained model every save_model_freq timestpes. default is 5000.")
    parser.add_argument("--base_path", type=str, default="models/",
                        help="base path for saving logs and more")
    parser.add_argument("--record_trajectory_length", type=int, default=200,
                        help="Record trajectory length")
    parser.add_argument("--save_buffer_pickle_n", type=float, default=5000,
                        help="save buffer pickle after n steps")
    parser.add_argument("--shield_save_stats_freq", type=int, default=1000,
                        help="save shield stats every 'shield_save_stats_freq' timesteps ")

    # agents training
    parser.add_argument("--verbose", type=int, default=1,
                        help="ppo verbose argument for stable3 PPO implementation. determines the amount of info printed to console during training process. valid values are 0,1,2.")
    parser.add_argument("--total_timesteps", type=int, default=1e6,
                        help="ppo total_timesteps argument for stable3 PPO implementation. determines the total amount of training timesteps for ppo agent.")


    # PPO arguments
    parser.add_argument("--ppo_n_steps", type=int, default=100,
                        help="ppo n_steps argument for stable3 PPO implementation. it sets the buffer size of the PPO Algorithm")
    parser.add_argument("--ppo_n_epochs", type=int, default=10,
                        help="ppo n_epochs argument for stable3 PPO implementation")
    parser.add_argument("--ppo_gamma", type=float, default=0.9,
                        help="gamma argument for stable3 PPO implementation")
    parser.add_argument("--ppo_learning_rate", type=float, default=0.0001,
                        help="learning rate argument for stable3 PPO implementation")
    parser.add_argument("--ppo_clip_range", type=float, default=0.1,
                        help="ppo_clip_range argument stable3 PPO implementation")
    parser.add_argument("--ppo_policy", type=str, default="MlpPolicy",
                        help="ppo_policy argument stable3 PPO implementation")

    # Shield arguments
    parser.add_argument("--shield_K_epochs", type=int, default=10,
                        help="update Shield for K epochs")
    parser.add_argument("--shield_discount_factor_cost", type=float, default=0.9,
                        help="discount factor for shield, while calculating loss function")
    parser.add_argument("--shield_unsafe_tresh", type=float, default=0.9, help="Unsafe treshold for the Shield network")
    parser.add_argument("--shield_update_timestep", type=float, default=100,
                        help="Update the shield network each 'shield_update_episode' time steps")
    parser.add_argument("--shield_batch_size", type=int, default=1026,
                        help="The number of states to sample from shield buffer while updating Shield")
    parser.add_argument("--shield_buffer_size", type=int, default=int(1e6),
                        help="maximum amount of samples in shield buffer (prioritizied experience replay buffer)")
    parser.add_argument("--shield_lr", type=float, default=0.004,
                        help="shield optimizer learning rate")
    parser.add_argument("--shield_minimum_buffer_samples", type=int, default=50000,
                        help="start training the Shield only when the buffer has minimum 'minimum_buffer_samples' samples")
    parser.add_argument("--shield_sub_batch_size", type=int, default=1440,
                        help="Shield sub batch size")


    # Shield Convergence args
    parser.add_argument("--shield_convergence_threshold", type=float, default=0.01,
                        help="Minimum improvement in loss for considering convergence.")
    parser.add_argument("--shield_convergence_episodes_interval", type=int, default=100,
                        help="Number of episodes to check for convergence.")

    # cost arguments - cartpole
    parser.add_argument("--safe_limit_x", type=float, default=0.4,
                        help="safety distance from x-treshold (defining cost)")
    parser.add_argument("--safe_limit_theta", type=float, default=0.03,
                        help="safety distance from theta tresh (defining cost)")
    return parser.parse_args(arguments)



def get_input_size(env, observation_type):
    if env == 'highway-fast-v0' and observation_type == "Kinematics":
        # + 1 for concatenated action
        input_size= 71
    elif env == 'highway-fast-v0' and observation_type== "TimeToCollision":
        input_size = 91
    else:
        print(f"Not supported input size for env {env}")
    return input_size

def create_ppo_config(env, args):
    """Create PPO training configuration from arguments"""
    return {
        'ppo_config': PPOConfig(
            total_timesteps=args.total_timesteps,
            env=env,
            policy=args.ppo_policy,
            learning_rate=args.ppo_learning_rate,
            n_steps=args.ppo_n_steps,
            n_epochs=args.ppo_n_epochs,
            gamma=args.ppo_gamma,
            clip_range=args.ppo_clip_range,
            verbose=args.verbose
        )
    }


def main(arguments=None):
    args = get_arguments(arguments)
    save_args_to_file(args, args.base_path + "/arguments.txt")

    device = setup_device(args.use_gpu)

    shield_config = ShieldConfig(
        input_size=get_input_size(args.env, args.observation_type),
        num_layers=5,
        hidden_dim=32,
        activation=nn.ReLU(),
        unsafe_tresh=args.shield_unsafe_tresh,
        buffer_size= args.shield_buffer_size,
        batch_size=args.shield_batch_size,
        sub_batch_size=args.shield_sub_batch_size,
        learning_rate=args.shield_lr
    )
    shield = Shield(shield_config, device=device)
    is_masked_env = True if args.select_action_algo == "ShieldPPO" else False
    env, eval_env = setup_environment(args,shield, is_masked_env=is_masked_env)

    setup_seed(env, args.seed)
    # Create training configuration
    config = create_ppo_config(env, args)
    log_dir = os.path.join(args.base_path, "logs")
    eval_callback = EvalCallback(eval_env, log_path= log_dir, eval_freq=50,
                                 deterministic=True, render=False)
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "progress.csv")
    new_logger = configure(log_dir, ["csv", "stdout"])
    # define callback to collect for each episode (cumulative rewards, cumulative costs, episode length)

    # relevant only for ShieldPPO agent
    shield_callback_config = ShieldCallbackConfig(update_freq=args.shield_update_timestep,
                         save_freq=args.shield_save_stats_freq,
                         output_dir=log_dir)

    # Create and train agent
    agent = AgentFactory.create_agent(
        agent_type=args.select_action_algo,
        config=config,
        env=env,
        device=device,
        shield=shield,
        shield_callback_config = shield_callback_config)
    agent.set_logger(new_logger)

    agent.learn(
        total_timesteps=args.max_training_timesteps,
        progress_bar=True,
        log_interval=1,
        callback = eval_callback
    )

if __name__ == '__main__':
    main()