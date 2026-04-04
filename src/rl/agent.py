"""PPO agent for the ride-hailing dispatch environment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

from src.config import get_config


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int] | None = None):
        super().__init__()
        cfg = get_config()["rl"]
        hs = hidden_sizes or cfg["hidden_sizes"]
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hs:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, act_dim)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return Categorical(logits=logits), value


class PPOBuffer:
    """Stores trajectories for a PPO update."""

    def __init__(self):
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []
        self.dones: list[bool] = []

    def store(self, obs, action, reward, value, log_prob, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def compute_gae(self, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_val = self.values[t + 1] if t + 1 < n else 0.0
            mask = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * next_val * mask - self.values[t]
            last_gae = delta + gamma * lam * mask * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(self.values[:n], dtype=np.float32)
        return torch.tensor(advantages), torch.tensor(returns)


class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int):
        cfg = get_config()["rl"]
        self.net = PolicyNetwork(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])
        self.gamma = cfg["gamma"]
        self.lam = cfg["gae_lambda"]
        self.clip_eps = cfg["clip_epsilon"]
        self.epochs = cfg["epochs_per_update"]
        self.buffer = PPOBuffer()

    def select_action(self, obs: np.ndarray) -> tuple[int, float, float]:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist, value = self.net(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def update(self, entropy_coeff: float = 0.01) -> dict[str, float]:
        advantages, returns = self.buffer.compute_gae(self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32)
        act_t = torch.tensor(self.buffer.actions, dtype=torch.long)
        old_lp = torch.tensor(self.buffer.log_probs, dtype=torch.float32)

        total_loss = 0.0
        for _ in range(self.epochs):
            dist, values = self.net(obs_t)
            new_lp = dist.log_prob(act_t)
            ratio = torch.exp(new_lp - old_lp)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()

        self.buffer.clear()
        return {"loss": total_loss / self.epochs}

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, weights_only=True))
