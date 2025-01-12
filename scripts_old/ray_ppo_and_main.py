# ppo_ray.py
from ray.rllib.algorithms.ppo import PPOConfig
import warnings

warnings.filterwarnings("ignore")

def train_ppo():
    config = (
        PPOConfig()
        .environment(env="highway-fast")  # Use the pre-built environment
        .framework("torch")  # Use PyTorch (or "tf" for TensorFlow)
        .env_runners(num_env_runners=4)
  # Specify the number of workers for parallelization
        .training(train_batch_size=4000)  # Training batch size for PPO
    )

    algo = config.build()  # Build the PPO algorithm from the configuration
    return algo  # Return the PPO algorithm for use in the main loop


# main.py
import gymnasium as gym
import ray
from draft_ray_ppo import train_ppo  # Assuming you define the PPO training function in ppo_ray.py
import highway_env
from gymnasium import register
from ray_register_env import register_env


register_env()

def run_rl_loop():
    env = gym.make("highway-fast-v0")  # Create the environment

    # Initialize the PPO algorithm
    ray.init(ignore_reinit_error=True)
    ppo_algorithm = train_ppo()  # Returns the PPO algorithm built from ppo_ray.py

    for step in range(100):  # Run for 100 steps
        obs, info = env.reset()  # Reset the environment at the start of each episode

        done = False
        total_reward = 0

        while not done:
            action = ppo_algorithm.compute_action(obs)  # Select action using PPO model
            obs, reward, done, truncated, info = env.step(action)  # Take a step in the environment

            total_reward += reward

            # After every 10 steps, train the PPO model to improve its policy
            if step % 10 == 0:
                ppo_algorithm.train()

            print(f"Step: {step}, Total Reward: {total_reward}")

        print(f"Episode {step} finished with reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    run_rl_loop()
