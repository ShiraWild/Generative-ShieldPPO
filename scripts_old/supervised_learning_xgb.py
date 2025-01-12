# imports

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Subset
import random
import torch.nn.functional as F
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import time
import argparse

from xgboost import XGBRFRegressor


def myExpLoss (logits, labels):
    return  (((2.0 * labels.float() - 1.0) * logits).exp()).mean()

# set device
print("============================================================================================")
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

# arguments
parser = argparse.ArgumentParser(description='Supervised Learning with Optuna')
parser.add_argument('--base_output_path', type=str, required=True, help='Base path to save the output file')
parser.add_argument('--trajectories_pkl_path', type=str, required=True, help='Path to trajectories pkl file')
parser.add_argument('--n_trials', type=int, required=True, default = 50, help='Number of trials for Optuna optimization. Default is 50')

args = parser.parse_args()
save_output_path = os.path.join(args.base_output_path, f"{args.n_trials}_trials_{int(time.time())}.csv")
os.makedirs(os.path.dirname(save_output_path), exist_ok=True)

# seeding
random_seed = 0
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


def exponential_loss(y_pred, y_true):
    return torch.mean(torch.exp(torch.abs(y_pred - y_true)) - 1)

try:
    with open(args.trajectories_pkl_path, 'rb') as file:
        trajectories = pickle.load(file)
        print(f"Successfully loaded  'samples_path' with {len(trajectories)} samples.")
except Exception as e:
    print(f"An error occurred: {e}")


class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.data[idx][0]
        action = self.data[idx][1]
        cost = self.data[idx][2]
        state_input = state.view(-1)
        x = torch.cat((state_input, action.float()), dim=0)
        return x, cost.float()

# supervised learning - split to train and test
train_size = int(0.8 * len(trajectories))
test_size = len(trajectories) - train_size
train_dataset, test_dataset = random_split(TrajectoryDataset(trajectories), [train_size, test_size])



trial_results = []


def objective(trial):
    start_time = time.time()

    lr_shield = trial.suggest_loguniform('lr_shield', 1e-5, 1e-1)  # Extended range: from 0.00001 to 0.1
    # Convert mini_batch_size to a range instead of categorical for broader exploration

    shield_model = XGBRFRegressor()
    #criterion = exponential_loss
    criterion = nn.L1Loss()
    x_train, y_train = [x[0] for x in train_dataset], [x[1] for x in train_dataset]

    x_test, y_test = [x[0] for x in test_dataset], [x[1] for x in test_dataset]
    shield_model.fit(np.array(x_train), np.array(y_train))
    y_pred = shield_model.predict(np.array(x_train))
    mae = mean_absolute_error(y_pred, y_train)
    trial_csv_path = os.path.join(args.base_output_path, f"trial_{trial.number}.csv")
    trial_df = pd.DataFrame({
        'y_pred': [y.item() for y in y_pred],
        'y_test': [y.item() for y in y_train]
    })
    trial_df.to_csv(trial_csv_path, index=False)
    print(f"Trial {trial.number} predictions saved to {trial_csv_path}")

    return mae
# List to store trial results


# Run the study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=args.n_trials)