from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import gymnasium as gym
import pandas as pd


def mask_fn(env: gym.Env):
    num_actions = env.action_space.n
    current_state = env.shield.state
    # Initialize shield_predictions if not already set
    if current_state is None:
        print("Warning: State is not initialized. Returning an unmasked action mask.")
        return np.ones(num_actions, dtype=int)

    # Create state-action pairs for all possible actions
    states = np.repeat(current_state, num_actions, axis=0)
    actions = np.arange(num_actions)

    # Get shield predictions for all state-action pairs
    with torch.no_grad():
        shield_predictions = env.shield.predict_batch(
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32)
        )
    env.shield_predictions = shield_predictions

    # Create mask based on shield predictions
    action_mask = (shield_predictions.squeeze() >= env.shield.unsafe_tresh).cpu().numpy().astype(int)
    return action_mask

class ShieldCallback(BaseCallback):
    def __init__(self, device, shield, update_freq, save_freq, output_dir, verbose):
        super().__init__(verbose)
        self.verbose = verbose
        self.shield = shield
        self.update_freq = update_freq
        self.save_freq = save_freq
        self.output_dir = output_dir

        self.loss_log = []  # To store (update_timestep, loss_value)
        self.last_save_timestep = 0  # To track the last save point
        self.device = device
        self.predictions_costs = []  # To store list of tuples [ (states, shield_predictions, costs) , ... ]
        self.predictions_costs_df = pd.DataFrame(columns=["state", "shield_pred", "cost"])

    def _on_step(self):
        # Get the latest transition
        if self.locals.get('new_obs') is not None:
            states = self.locals['obs_tensor']
            actions = self.locals['actions']
            infos = self.locals['infos']
            self.shield.set_state(self.locals.get('new_obs'))

            # Assuming collision information is in info dict
            costs = np.array([info["crashed"] for info in infos])

            # Add to shield buffer
            self.shield.buffer.push(
                torch.tensor(states, dtype=torch.float32),
                actions,
                torch.tensor(costs, dtype=torch.float32)
            )
            """
            if shield_predictions:
                self.predictions_costs.append((states, shield_predictions, costs))
            """
        # Update shield periodically
        if self.num_timesteps % self.update_freq == 0:
            loss = self.shield.update()
            self.loss_log.append((self.num_timesteps, loss))
            if self.verbose > 0:
                print(f"Progress update from timestep {self.num_timesteps}: Updated shield, loss is {loss}")

        # Save log periodically
        if self.num_timesteps - self.last_save_timestep >= self.save_freq:
            """
            self._save_predictions_costs_log()
            """
            self._save_loss_log()
            self.last_save_timestep = self.num_timesteps
            if self.verbose > 0:
                print(f"Progress update from timestep {self.num_timesteps}: Saved shield_loss.csv. ")

        return True

    def _on_training_end(self):
        # Save remaining logs at the end of training
        self._save_loss_log()

    def _save_loss_log(self):
        if self.loss_log:
            df = pd.DataFrame(self.loss_log, columns=["update_timestep", "loss_value"])
            output_file = self.output_dir + "/shield_loss.csv"
            df.to_csv(output_file, index=False)
            if self.verbose > 0:
                print(f"Loss log saved to {output_file}")

    def _save_predictions_costs_log(self):
        if self.predictions_costs:
            states, shield_pred, costs = ([i_env[0] for i_env in self.predictions_costs],
                                          [i_env[1] for i_env in self.predictions_costs],
                                          [i_env[2] for i_env in self.predictions_costs])
            for state, action, cost in zip(states, shield_pred, costs):
                self.predictions_costs_df = pd.concat([self.predictions_costs_df, pd.DataFrame({
                    "state": [state],
                    "shield_pred": [shield_pred],
                    "cost": [cost]
                })], ignore_index=True)
                output_file = self.output_dir + '/predictions_costs_log.csv'
                self.predictions_costs_df.to_csv(output_file, index=False)
                print(f"Predictions and costs log saved to {output_file}")

class ShieldMaskablePPO(MaskablePPO):
    def __init__(self, ppo_config, env, shield, shield_callback_config, device):
        super().__init__(
            verbose=ppo_config.verbose,
            env = ppo_config.env,
            policy=ppo_config.policy,
            learning_rate=ppo_config.learning_rate,
            n_steps=ppo_config.n_steps,
            n_epochs=ppo_config.n_epochs,
            gamma=ppo_config.gamma,
            clip_range=ppo_config.clip_range,
            device = device
        )
        self.shield = shield
        self.device = device
        self.shield_callback = ShieldCallback(device=self.device, shield=self.shield, update_freq = shield_callback_config.update_freq,
                                              save_freq= shield_callback_config.save_freq,
                                              output_dir = shield_callback_config.output_dir, verbose=ppo_config.verbose)

    def learn(self, total_timesteps, progress_bar, log_interval, callback):
        # Call the parent class's learn method

        return super().learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            log_interval=log_interval,
            callback=[self.shield_callback, callback]
        )