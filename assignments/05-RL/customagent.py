import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Docstring
    """

    def __init__(self, observation_space, action_space):
        """
        Docstring
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(observation_space, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc7 = nn.Linear(32, action_space)

    def forward(self, x):
        """
        Docstring
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        # x = torch.relu(self.fc5(x))
        # x = torch.relu(self.fc6(x))

        x = self.fc7(x)
        return x


class Agent:
    """
    Docstring
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        """
        Docstring
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.q_network = QNetwork(observation_space.shape[0], action_space.n).float()
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=0.001, weight_decay=1e-5
        )
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.97
        self.epsilon = 0.01
        self.last_action = None

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Docstring
        """
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.q_network(
                    torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                )
            action = int(torch.argmax(q_values).item())
        self.last_action = action
        return action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Docstring
        """
        action = self.last_action
        q_values = self.q_network(
            torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        )
        target_q_values = q_values.clone().detach()

        if terminated or truncated:
            target_q_values[0, action] = reward
        else:
            with torch.no_grad():
                next_q_values = torch.max(
                    self.q_network(
                        torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    )
                )
            target_q_values[0, action] = reward + self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
