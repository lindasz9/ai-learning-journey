# Reinforcement Learning Algorithms in Python

## Algorithms

📘 **Off-Policy Algorithms**

* Q-Learning
* Deep Q-Networks (DQN)

📙 **On-Policy Algorithms**

* SARSA
* Policy Gradient Methods
* Actor-Critic Methods

---

## 📘 Off-Policy Algorithms

### 🔹 Q-Learning

```python
import numpy as np
import gym

env = gym.make("FrozenLake-v1", is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))

gamma = 0.95
alpha = 0.8
epsilon = 0.1
episodes = 2000

for _ in range(episodes):
    state = env.reset()[0]
    done = False
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        next_state, reward, done, _, _ = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        state = next_state

print("Trained Q-table:\n", Q)
```

### 🔹 Deep Q-Network (DQN)

```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

env = gym.make("CartPole-v1")
model = DQN(env.observation_space.shape[0], env.action_space.n)
target_model = DQN(env.observation_space.shape[0], env.action_space.n)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=1e-3)
replay_buffer = deque(maxlen=10000)
gamma = 0.99
epsilon = 0.1

for episode in range(500):
    state = env.reset()[0]
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = model(torch.FloatTensor(state)).argmax().item()
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state

        if len(replay_buffer) >= 64:
            batch = random.sample(replay_buffer, 64)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            q_values = model(states).gather(1, actions).squeeze()
            next_q_values = target_model(next_states).max(1)[0]
            expected_q = rewards + gamma * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())
```

---

## 📙 On-Policy Algorithms

### 🔹 SARSA

```python
import numpy as np
import gym

env = gym.make("FrozenLake-v1", is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))

gamma = 0.95
alpha = 0.8
epsilon = 0.1
episodes = 2000

for _ in range(episodes):
    state = env.reset()[0]
    action = np.random.choice(env.action_space.n) if np.random.rand() < epsilon else np.argmax(Q[state])
    done = False
    while not done:
        next_state, reward, done, _, _ = env.step(action)
        next_action = np.random.choice(env.action_space.n) if np.random.rand() < epsilon else np.argmax(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
        state, action = next_state, next_action

print("Trained Q-table:\n", Q)
```

### 🔹 Policy Gradient (REINFORCE)

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.fc(x)

env = gym.make("CartPole-v1")
policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

episodes = 500
for _ in range(episodes):
    state = env.reset()[0]
    log_probs = []
    rewards = []
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state)
        probs = policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))
        state, reward, done, _, _ = env.step(action.item())
        rewards.append(reward)

    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    loss = -torch.stack(log_probs) * (returns - returns.mean())
    loss = loss.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 🔹 Actor-Critic (A2C)

```python
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Linear(obs_dim, 128)
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.shared(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for episode in range(500):
    state = env.reset()[0]
    log_probs, values, rewards = [], [], []
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state)
        probs, value = model(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_state, reward, done, _, _ = env.step(action.item())

        log_probs.append(dist.log_prob(action))
        values.append(value)
        rewards.append(reward)

        state = next_state

    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    values = torch.cat(values).squeeze()

    advantage = returns - values
    actor_loss = -torch.stack(log_probs) * advantage.detach()
    critic_loss = advantage.pow(2)

    loss = actor_loss.sum() + critic_loss.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
