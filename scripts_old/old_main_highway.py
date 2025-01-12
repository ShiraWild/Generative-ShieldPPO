import pickle
import time
import torch
import torch.nn as nn
import gymnasium as gym
import argparse
import numpy as np
import pandas as pd
import os
import random
from highway_env.envs import HighwayEnv
from stable_baselines3.common.env_util import make_vec_env
from utilities.shield_gen import Shield
from gymnasium import spaces, register
#from utilities.ppo import PPO
import highway_env
from stable_baselines3 import PPO
#import ray
from typing import NamedTuple
from gymnasium.wrappers import FlattenObservation
import warnings
from stable_baselines3.common.env_util import DummyVecEnv
#from envs.stable3_env import *
from algorithms.ppo_stable3 import CustomPPO, PPOConfig
from algorithms.ShieldPPO import ShieldPPO, ShieldConfig



################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

def highway_env_creator(config_env: dict):
    env = HighwayEnv(config_env)
    cfg = env.unwrapped.default_config()
    cfg.update(config_env)
    env.unwrapped.config = cfg
    #env = FlattenObservation(env)
    return env

"""
        learning_rate: Union[float, Schedule] = 3e-4, #lr_actor, lr_critic (not separated)
        n_steps: int = 2048, # equivalent to update_timesteps
        n_epochs: int = 10, # equivalent to k_epochs
        gamma: float = 0.99, # equivalent to gamma 
        clip_range: Union[float, Schedule] = 0.2, # equivalent to eps_clip 
    )
"""



"""

# create end points for the environments
def register_env(env):
    entry = 'envs.modified_envs:' + env[:-len('-v0')]  # Update the entry_point
    register(
        id=env,
        entry_point=entry,
    )
"""

learning_rate: float
n_steps: int
n_epochs: int
gamma: float
clip_range: float
env: gym.Env
total_timesteps: int



def get_valid_actions(env):
    return list(range(env.action_space.n))  # returns a vector of values 0/1 which indicate which actions are valid


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

def train(arguments=None):
    parser = argparse.ArgumentParser()

    # General training arguments
    # TBD - add GenShieldPPO (however we'll name it)
    parser.add_argument("--select_action_algo", default="PPO",
                        help="algorithm to use when selecting action:  PPO | Random | ShieldPPO | RuleBasedShieldPPO (REQUIRED)")
    parser.add_argument("--env", default="highway-fast-v0",
                        help="names of the environment to train on")
    parser.add_argument("--max_ep_len", type=int, default=500,
                        help="max timesteps in one episode. In cartpole-v0 env it should be 200.")
    parser.add_argument("--max_training_timesteps", type=int, default=int(1e6),
                        help="break training loop after 'max_training_timesteps' timesteps.")

    # running setups
    parser.add_argument("--render", type=bool, default=False,
                        help="render environment. default is False.")
    parser.add_argument("--cpu", type=int, default=4,
                        help="Number of cpus")
    parser.add_argument("--seed", type=int, default=0,
                        help="defines random seed. default is 0.")
    parser.add_argument("--highway_observation_type", type=str, default="Kinematics",
                        choices=["Kinematics", "OccupancyGrid", "TimeToCollision"],
                        help="Defines the observation type of the highway environment.")

    # logs & frequencies
    parser.add_argument("--log_freq", type=int, default=5,
                        help="save logs every log_freq episodes. default is 5.")
    parser.add_argument("--save_model_freq", type=int, default=int(5000),
                        help="save trained model every save_model_freq timestpes. default is 5000.")
    parser.add_argument("--base_path", type=str, default="models/",
                        help="base path for saving logs and more")
    parser.add_argument("--record_trajectory_length", type=int, default=200,
                        help="Record trajectory length")
    parser.add_argument("--save_buffer_pickle_n", type=float, default=5000,
                        help="save buffer pickle after n steps")

    # PPO arguments

    parser.add_argument("--ppo_n_steps", type=int, default=1000,
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
    parser.add_argument("--shield_K_epochs", type=int, default=30,
                        help="update Shield for K epochs")
    parser.add_argument("--shield_discount_factor_cost", type=float, default=0.6,
                        help="discount factor for shield, while calculating loss function")
    parser.add_argument("--shield_unsafe_tresh", type=float, default=0.5, help="Unsafe treshold for the Shield network")
    parser.add_argument("--shield_update_timestep", type=float, default=50,
                        help="Update the shield network each 'shield_update_episode' time steps")
    parser.add_argument("--shield_batch_size", type=int, default=50000,
                        help="The number of states to sample from shield buffer while updating Shield")
    parser.add_argument("--shield_buffer_size", type=int, default=int(1e6),
                        help="maximum amount of samples in shield buffer (prioritizied experience replay buffer)")
    parser.add_argument("--shield_lr", type=float, default=1e-3,
                        help="shield optimizer learning rate")
    parser.add_argument("--shield_minimum_buffer_samples", type=int, default=50000,
                        help="start training the Shield only when the buffer has minimum 'minimum_buffer_samples' samples")
    parser.add_argument("--shield_sub_batch_size", type=int, default=500,
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

    # parse the arguments

    args = parser.parse_args(arguments)
    print(args)
    # General training arguments

    if args.env == "CartPoleWithCost-v0":
        env = gym.make(args.env, safe_limit_x=args.safe_limit_x, safe_limit_theta=args.safe_limit_theta)
    elif args.env == "highway-fast-v0" and args.highway_observation_type == "Kinematics":
        # env_spec = gym.envs.registry.get(args.env)
        # env = gym.make(env_spec)
        env_config = {
            "observation": {
                "normalize": False,
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],

            },
            "duration": 50,
            "vehicles_count": 20,
            "vehicles_density": 2,
        }

        env = highway_env_creator(env_config)
        # # env = FlattenObservation(env)
        # cfg = env.unwrapped.default_config()
        # cfg.update(config)
        # env.unwrapped.config= cfg
        # env.reset()
    elif args.env == "highway-fast-v0" and args.highway_observation_type == "OccupancyGrid":
        env = gym.make(args.env)
        config = {
            "vehicles_count": 10,
            "observation": {
                "normalize":False,
                "vehicles_count": 5,
                "type": "OccupancyGrid",
                "features": ["x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-50, 50],
                    "y": [-50, 50],
                    "vx": [-10, 10],
                    "vy": [-10, 10]
                },
                "grid_size": [[-20, 20], [-20, 20]],
                "grid_step": [5, 5],
                "absolute": False
            }
        }
        cfg = env.unwrapped.default_config()
        cfg.update(config)
        env.unwrapped.config= cfg
        env.reset()
    elif args.env == "highway-fast-v0" and args.highway_observation_type == "TimeToCollision":
        env = gym.make(args.env)
        config = {"observation": {"normalize":False, "type": "TimeToCollision", "horizon": 10}}
        cfg = env.unwrapped.default_config()
        cfg.update(config)
        env.unwrapped.config= cfg
        env.reset()
    else:
        raise ValueError(
            f"Unsupported environment: {args.env}. Please choose a valid environment (e.g., 'CartPoleWithCost-v0' or 'highway-fast-v0').")
    # env set up

    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n


    base_path = args.base_path + f"/observation_type={env_config['observation']['type']}"

    # create log paths

    if not os.path.exists(base_path):
        print(f"Given base path directory '{base_path}' did not exist. Creating it.. ")
        os.makedirs(base_path)

    save_model_path = f"./{base_path}/model.pth"
    save_args_path = f"./{base_path}/commandline_args.txt"
    save_stats_path = f"./{base_path}/stats.log"
    os.makedirs(base_path + "/Videos", exist_ok=True)

    # Define random seed
    random_seed = args.seed

    # save arguments to text file

    save_args_path = base_path + "/commandline_args.txt"  # You can customize this path
    with open(save_args_path, 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    if random_seed:
        print("random seed is set to ", random_seed)
        torch.manual_seed(random_seed)
        # CUDA seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        env.action_space.seed(random_seed)

    """
    # define agent according to given argument
    if agent == "PPO":
        ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space, action_std)

    elif agent == "ShieldPPO":
        ppo_agent = ShieldPPO(state_dim=state_dim, action_dim=action_dim, lr_actor=lr_actor, lr_critic=lr_critic,
                              gamma=gamma, eps_clip=eps_clip, k_epochs_ppo=K_epochs, k_epochs_shield=K_epochs_shield,
                              k_epochs_gen=K_epochs_gen,
                              has_continuous_action_space=has_continuous_action_space, lr_shield=lr_shield,
                              lr_gen=lr_gen, latent_dim=latent_dim, discount_factor_cost=discount_factor_cost,
                              action_std_init=action_std, masking_threshold=shield_masking_tresh,
                              unsafe_tresh=unsafe_tresh, use_gen_v2 = use_gen_v2,shield_buffer_size = shield_buffer_size, shield_batch_size = shield_batch_size,  param_ranges=param_ranges)

    else:
        print("Accepting one of the following agents as input - PPO, ShieldPPO, RuleBasedShieldPPO")
    """

    # Define agents

    if args.env == 'highway-fast-v0' and args.highway_observation_type == "Kinematics":
        # + 1 for concatenated action
        input_size= 71
    elif args.env == 'highway-fast-v0' and args.highway_observation_type == "OccupancyGrid":
        observation_config = env.unwrapped.config['observation']
        features_channels = len(observation_config['features'])
        grid_size, grid_step = observation_config["grid_size"], observation_config["grid_step"]
        x_span, y_span = grid_size[0][1] - grid_size[0][0], grid_size[1][1] - grid_size[1][0]
        num_x_cells = x_span // grid_step[0]
        num_y_cells = y_span // grid_step[1]
        grid_dim = num_x_cells * num_y_cells
        input_size = grid_dim *  features_channels + 1 # + 1 for concatenated action
    elif args.env == 'highway-fast-v0' and args.highway_observation_type == "TimeToCollision":
        input_size = 91
    else: # cartpole
        input_size = 5

    #ppo_agent = PPO(state_dim, action_dim, args.ppo_lr_actor, args.ppo_lr_critic, args.ppo_gamma, args.ppo_K_epochs, args.ppo_eps_clip, False)

    # counters
    # logs
    cost_log = []  # List to log (time_step, cost) at each step
    reward_log = []  # List to log (time_step, reward) at each step

    time_step = 0
    costs_per_episode = []
    rewards_per_episode = []
    episode_cnt = 0
    episodes_len = []

    ppo_update_log = pd.DataFrame(columns=['update_timestep', 'loss'])
    if args.select_action_algo == "PPO":
        ppo_config = PPOConfig(policy=args.ppo_policy, learning_rate=args.ppo_learning_rate, n_steps=args.ppo_n_steps, n_epochs=args.ppo_n_epochs, gamma=args.ppo_gamma, clip_range=args.ppo_clip_range)
        agent = CustomPPO(ppo_config=ppo_config,env=env, log_path = args.base_path)
        agent.policy.to(device)
    elif args.select_action_algo == "ShieldPPO":
        ppo_config = PPOConfig(policy=args.ppo_policy, learning_rate=args.ppo_learning_rate, n_steps=args.ppo_n_steps, n_epochs=args.ppo_n_epochs, gamma=args.ppo_gamma, clip_range=args.ppo_clip_range)
        shield_config = ShieldConfig(input_size=input_size, num_layers=5, hidden_dim=32, activation=nn.MSELoss())
        agent = ShieldPPO(env, ppo_config, shield_config, args.base_path, device)

    else:
       print("No supported select action algorithm")
    total_training_timesteps = 0
    ppo_update_loss = {"entropy_loss": [], "pg_loss": []}

    while time_step < args.max_training_timesteps:
        episode_cnt += 1
        if time_step != 0:
            episodes_len.append((episode_cnt, t + 1))
        # print(f"Starting episode {episode + 1}...")
        state, info = env.reset()
        done = False
        ep_cumulative_cost = 0
        ep_cumulative_reward = 0
        for t in range(1, args.max_ep_len + 1):
            prev_state = state.copy()
            start_time = time.time()
            valid_actions = get_valid_actions(env)
            if args.select_action_algo == "PPO":
                flatten_state = torch.tensor(state.flatten()).unsqueeze(0).to(device)
                action, log_prob, value = agent.select_action(flatten_state)
            elif args.select_action_algo == "Random":
                action = random.choice(valid_actions)
            elif args.select_action_algo == "ShieldPPO":
                flatten_state = torch.tensor(state.flatten()).unsqueeze(0).to(device)
                action = agent.select_action(flatten_state)
            else:
                raise ValueError(
                    f"Unsupported action selection algorithm: {args.select_action_algo}. Please choose a valid algorithm (e.g., 'PPO' or 'Random').")

            if args.env == 'highway-fast-v0':
                new_state, reward, done, truncated, info = env.step(action)
                state = new_state
                cost = info['crashed']
                episode_start = np.array([True] if t == 1 else [False])
                agent.add_to_buffer(prev_state, action.cpu().numpy(), reward, episode_start, value.detach(), log_prob.detach())

            else:  # other envs
                state, reward, done, info = env.step(action)
                cost = info['cost']
            # add feedback from environment to PPO agent (the rest is added from ppo.select_action method
            """
            if args.select_action_algo == 'PPO':
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(done)
            """

            # Log the cost and reward at this time step

            ep_cumulative_cost += cost
            ep_cumulative_reward += reward

            time_step += 1
            if time_step % 500 == 0:
                print(f"time_step is now {time_step}")

            if done:
                break

        if agent.rollout_buffer.full:
            print(f"Updating agent in time_step {time_step}")
            entropy_loss, pg_loss = agent.update()
            ppo_update_loss["entropy_loss"].append(entropy_loss)
            ppo_update_loss["pg_loss"].append(pg_loss)



        # print(f"Episode {episode_cnt} length is {t+1}, finished due to {done} condition;")
        costs_per_episode.append((episode_cnt + 1, ep_cumulative_cost))
        rewards_per_episode.append((episode_cnt + 1, ep_cumulative_reward))

        # print(f"Finished episode {episode} after {episode_len} time steps")
        """
        # relevant if the update is in the end of each episode
        if shield_net.buffer.get_buffer_len() >= shield_minimum_buffer_samples:
            shield_loss = shield_net.update(num_update=shield_num_updates)
            # Log the shield loss for this update, along with the current time step
            # int(f"Log loss for episode {episode_cnt} = {shield_loss}")
            shield_loss_log.append((time_step, shield_loss))  # Store (time_step, loss)
            shield_num_updates += 1
            print(
                f"Num Update {shield_num_updates}. Average loss for updating Shield after episode {episode_cnt + 1} is: {shield_loss}")
        """
        if episode_cnt % args.log_freq == 0:
            # print(f"Saving logs to {base_path} in the end of episode {episode_cnt+1}.")
            cost_df = pd.DataFrame(costs_per_episode, columns=['Episode', 'Cumulative Cost'])
            reward_cumulative_df = pd.DataFrame(rewards_per_episode, columns=['Episode', 'Cumulative Reward'])

            episodes_len_df = pd.DataFrame(episodes_len, columns=['Episode', 'Episode Length'])

            # save DataFrames to CSV files
            costs_csv_path = base_path + "/costs_log.csv"
            cumulative_rewards_csv_path = base_path + "/ep_cumulative_rewards_log.csv"

            episodes_len_path = base_path + "/episodes_len.csv"

            cost_df.to_csv(costs_csv_path, index=False)
            reward_cumulative_df.to_csv(cumulative_rewards_csv_path, index=False)

    """
    # not relevant in the Stable-Baselines3 implementation — these are saved automatically
        if args.select_action_algo == 'PPO':
        # save ppo loss
        ppo_update_log.to_csv(f"{base_path}/ppo_update_log.csv", index=False)
    """
    ppo_loss_df = pd.DataFrame(ppo_update_loss)
    ppo_loss_df.to_csv(args.base_path + "/ppo_loss_values.csv", index=False)

    # shield_losses_df = pd.DataFrame(shield_loss_log, columns=['Update Time Step', 'Loss'])
    # shield_losses_df.to_csv(losses_csv_path, index=False)


if __name__ == '__main__':
    train()



