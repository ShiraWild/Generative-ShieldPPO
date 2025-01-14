import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str, default="models/",
                    help="base path for saved dataframes and logs")
parser.add_argument("--window_size", type=int, default=10,
                    help="Window size for moving average (number of episodes)")
args = parser.parse_args()

def plot_with_moving_average(base_path, reward_file, cost_file, window_size):
    """
    Plots cumulative rewards and costs over episodes with moving average applied.

    Parameters:
        base_path (str): Base path for saving plots.
        reward_file (str): Path to the CSV file containing cumulative rewards.
        cost_file (str): Path to the CSV file containing cumulative costs.
        window_size (int): Window size for moving average (number of episodes).
    """
    plots_base_path = base_path + "/plots"
    # create 'plots' dir under base_path, if not exists
    os.makedirs(plots_base_path, exist_ok=True)
    # Load the data
    try:
        reward_df = pd.read_csv(reward_file)
        cost_df = pd.read_csv(cost_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Check for required columns
    for df, y_label in zip([reward_df, cost_df], ['Cumulative Reward', 'Cumulative Cost']):
        if 'Episode' not in df.columns or y_label not in df.columns:
            print(f"Error: Required columns 'Episode' and '{y_label}' not found in the file.")
            return

    # Calculate moving averages
    reward_df['Smoothed Reward'] = reward_df['Cumulative Reward'].rolling(window=window_size).mean()
    cost_df['Smoothed Cost'] = cost_df['Cumulative Cost'].rolling(window=window_size).mean()

    # Plot cumulative rewards
    plt.figure(figsize=(12, 6))
    plt.plot(reward_df['Episode'], reward_df['Cumulative Reward'], alpha=0.5, label='Raw Reward')
    plt.plot(reward_df['Episode'], reward_df['Smoothed Reward'], label=f'Smoothed Reward (Window={window_size})', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward over Episodes')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{plots_base_path}/Cumulative_Reward_Plot.png")
    print(f"Saved reward plot to {plots_base_path}/Cumulative_Reward_Plot.png")

    # Plot cumulative costs
    plt.figure(figsize=(12, 6))
    plt.plot(cost_df['Episode'], cost_df['Cumulative Cost'], alpha=0.5, label='Raw Cost', color='orange')
    plt.plot(cost_df['Episode'], cost_df['Smoothed Cost'], label=f'Smoothed Cost (Window={window_size})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Cost')
    plt.title('Cumulative Cost over Episodes')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{plots_base_path}/Cumulative_Cost_Plot.png")
    print(f"Saved cost plot to {plots_base_path}/Cumulative_Cost_Plot.png")

# Example usage
plot_with_moving_average(
    base_path=args.base_path,
    reward_file=args.base_path + '/ep_cumulative_rewards_log.csv',
    cost_file=args.base_path + '/costs_log.csv',
    window_size=args.window_size
)
