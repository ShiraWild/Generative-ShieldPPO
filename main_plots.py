import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_progress_with_smoothing(csv_path, output_path, smoothing_window=50):
    """
    Reads a progress.csv file, extracts relevant columns, and saves three plots to the output path.
    Adds smoothed lines to show trends in the data.

    Args:
        csv_path (str): Path to the input progress.csv file.
        output_path (str): Path to save the output plots.
        smoothing_window (int): Window size for rolling average smoothing.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Extract relevant columns
    columns_to_extract = [
        "eval/mean_reward",
        "time/total_timesteps",
        "eval/mean_ep_length",
        "train/loss"
    ]
    # Filter DataFrame for the required columns
    df = df[columns_to_extract]

    # Ensure the columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)  # Remove rows with missing values

    # Compute smoothed values using rolling average
    smoothed_df = df.rolling(window=smoothing_window, min_periods=1).mean()

    # Plot 1: Train Loss over Time (Total Timesteps)
    plt.figure(figsize=(10, 6))
    plt.plot(df["time/total_timesteps"], df["train/loss"], label="Train Loss (Raw)", color="red", alpha=0.6)
    plt.plot(smoothed_df["time/total_timesteps"], smoothed_df["train/loss"], label="Train Loss (Smoothed)",
             color="darkred")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Train Loss")
    plt.title("PPO Train Loss over Total Timesteps")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}/ppo_train_loss_over_time.png")
    plt.close()

    # Plot 2: Mean Rewards over Total Timesteps
    plt.figure(figsize=(10, 6))
    plt.plot(df["time/total_timesteps"], df["eval/mean_reward"], label="Mean Reward (Raw)", color="blue", alpha=0.6)
    plt.plot(smoothed_df["time/total_timesteps"], smoothed_df["eval/mean_reward"], label="Mean Reward (Smoothed)",
             color="darkblue")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward over Total Timesteps")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}/mean_reward_over_timesteps.png")
    plt.close()

    # Plot 3: Mean Episode Length over Total Timesteps
    plt.figure(figsize=(10, 6))
    plt.plot(df["time/total_timesteps"], df["eval/mean_ep_length"], label="Mean Episode Length (Raw)", color="green",
             alpha=0.6)
    plt.plot(smoothed_df["time/total_timesteps"], smoothed_df["eval/mean_ep_length"],
             label="Mean Episode Length (Smoothed)", color="darkgreen")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Mean Episode Length")
    plt.title("Mean Episode Length over Total Timesteps")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}/mean_ep_length_over_timesteps.png")
    plt.close()

    print(f"Plots saved to {output_path}")
import pandas as pd
import matplotlib.pyplot as plt

def plot_shield_loss(csv_path, output_path, smoothing_window=50):
    """
    Reads a CSV file with `update_timestep` and `loss_value` columns,
    plots shield loss over time with raw and smoothed lines, and saves the plot.

    Args:
        csv_path (str): Path to the input CSV file.
        output_path (str): Path to save the output plot.
        smoothing_window (int): Window size for rolling average smoothing.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Ensure the columns are numeric
    df = df[["update_timestep", "loss_value"]].apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)  # Remove rows with missing values

    # Compute smoothed values using rolling average
    smoothed_df = df.rolling(window=smoothing_window, min_periods=1).mean()

    # Plot Shield Loss over Time
    plt.figure(figsize=(10, 6))
    plt.plot(df["update_timestep"], df["loss_value"], label="Loss (Raw)", color="orange", alpha=0.6)
    plt.plot(smoothed_df["update_timestep"], smoothed_df["loss_value"], label="Loss (Smoothed)", color="darkorange")
    plt.xlabel("Update Timestep")
    plt.ylabel("Loss Value")
    plt.title("Shield Loss over Time")
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_file = f"{output_path}/shield_loss_over_time.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")



def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Generate plots from SB3 PPO progress.csv with optional smoothing.")
    parser.add_argument("--progress_csv_path", type=str, required=True, help="Path to the progress.csv file.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the plots.")
    parser.add_argument("--smoothing_window", type=int, default=50, help="Window size for smoothing (default: 50).")
    parser.add_argument("--shield_loss_path", type=str, required=True, help="path to csv shield loss file")
    args = parser.parse_args()
    # Call the plotting function with parsed arguments
    #plot_progress_with_smoothing(args.csv_path, args.output_path, args.smoothing_window)
    plot_shield_loss(args.shield_loss_path, args.output_path, args.smoothing_window)

if __name__ == "__main__":
    main()
