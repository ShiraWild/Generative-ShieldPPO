# register_env.py
from gymnasium.envs.registration import register

# Register the environment globally (only once)
def register_env():
    register(
        id='highway-fast-v0',
        entry_point='highway_env.envs:HighwayEnv',  # Update this path to match your environment location
    )

# Call register_env() to register the environment
register_env()