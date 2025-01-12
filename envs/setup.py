from envs.modified_envs import ShieldedEnv
from utilities.shield_gen import Shield
from envs.configs import EnvConfig
import gymnasium as gym
from highway_env.envs import HighwayEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.wrappers import ActionMasker
from algorithms.shieldPPO_sb3 import mask_fn


def highway_env_creator(env_id, num_envs, config_env: dict, shield: Shield, is_masked_env):
    def make_env():
        env = HighwayEnv(config=config_env.copy())
        env = ShieldedEnv(env, shield)
        if is_masked_env:
            env = ActionMasker(env, mask_fn)
        return env

    return make_vec_env(make_env, n_envs=num_envs)

def setup_environment(args, shield: Shield, is_masked_env):
    """Setup and return the environment based on arguments"""
    if args.env == "highway-fast-v0":
        # Get the config based on the observation type passed in the args
        env_config = EnvConfig.get_config(args.observation_type)
        # Create the environment using the modified env creator function
        env = highway_env_creator(args.env, args.num_envs, env_config, shield,is_masked_env)
        eval_env = highway_env_creator(args.env, args.num_envs, env_config, shield,is_masked_env)

    else:  # In case of supporting more envs in the future
        env = gym.make(args.env)

    return env, eval_env
