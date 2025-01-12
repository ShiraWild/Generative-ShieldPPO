import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, cost):
        # Iterate through each sample in the batch (based on num_envs)
        # and add each sample individually to the buffer.
        for state, action, cost in zip(state, action, cost):
            self.buffer.append((state, action, cost))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def get_buffer_len(self):
        return len(self.buffer)


class TrajectoryDataset(Dataset):
    def __init__(self, samples, prepare_input_fn):
        self.samples = samples
        self.prepare_input_fn = prepare_input_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, action, cost = self.samples[idx]
        input_features = self.prepare_input_fn(state, action)
        return input_features, cost



class Shield(nn.Module):
    def __init__(self,config, device):
        super(Shield, self).__init__()
        # Network architecture
        layers = []
        layers.append(nn.Linear(config.input_size, config.hidden_dim))
        for _ in range(config.num_layers - 1):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(config.activation)
        layers.append(nn.Linear(config.hidden_dim, 1))
        layers.append(nn.Sigmoid())
        self.device = device
        self.model = nn.Sequential(*layers).to(self.device)
        self.unsafe_tresh = config.unsafe_tresh
        # Training components
        self.buffer = ReplayBuffer(config.buffer_size)
        self.batch_size = config.batch_size
        self.sub_batch_size = config.sub_batch_size
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.state = None

    def prepare_shield_input(self, state, action):
        """Concatenate state and action for shield input"""
        action_tensor = torch.tensor([action], dtype=torch.float32)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        return torch.cat((state_tensor.flatten().to(self.device), action_tensor.flatten().to(self.device))).to(self.device)

    def forward(self, x):
        return self.model(x.to(self.device))

    def set_state(self, new_state):
        self.state = new_state

    def predict_batch(self, states, actions):
        """Predict safety scores for multiple state-action pairs"""
        batch_inputs = torch.stack(list(map(self.prepare_shield_input, states, actions)))
        with torch.no_grad():
            return self.forward(batch_inputs)

    def update(self):
        batch_size = self.batch_size
        sub_batch_size = self.sub_batch_size
        if self.buffer.get_buffer_len() < self.batch_size:
            batch_size = self.buffer.get_buffer_len()
            sub_batch_size = batch_size // 2

        batch_samples = self.buffer.sample(batch_size)
        dataset = TrajectoryDataset(batch_samples, self.prepare_shield_input)
        data_loader = DataLoader(dataset, batch_size=sub_batch_size, shuffle=True)

        total_loss = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.float().to(self.device), targets.float().view(-1, 1).to(self.device)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(data_loader)