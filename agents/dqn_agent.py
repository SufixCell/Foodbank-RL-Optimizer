import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))
    def forward(self, x):
        z = self.feature(x)
        v = self.value(z)
        a = self.advantage(z)
        return v + a - a.mean(dim=1, keepdim=True)

class DQNAgent:
    def __init__(self, obs_dim, action_dim, lr=5e-4, gamma=0.97, device=None, tau=0.025):
        self.obs_dim = obs_dim; self.action_dim = action_dim; self.gamma = gamma; self.tau = tau
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.q = QNetwork(obs_dim, action_dim).to(self.device)
        self.target = QNetwork(obs_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.opt = optim.AdamW(self.q.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()
    def act(self, state, epsilon=0.0):
        if np.random.random() < epsilon:
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.q(s), dim=1).item())
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        q_values = self.q(states).gather(1, actions)
        with torch.no_grad():
            next_actions = torch.argmax(self.q(next_states), dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions)
            target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = self.loss_fn(q_values, target)
        self.opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step(); return float(loss.item())
    def sync_target(self):
        with torch.no_grad():
            for tp, qp in zip(self.target.parameters(), self.q.parameters()):
                tp.data.mul_(1.0 - self.tau); tp.data.add_(self.tau * qp.data)
    def hard_sync_target(self):
        self.target.load_state_dict(self.q.state_dict())
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model_state_dict': self.q.state_dict(), 'target_state_dict': self.target.state_dict(), 'optimizer_state_dict': self.opt.state_dict(), 'obs_dim': self.obs_dim, 'action_dim': self.action_dim, 'gamma': self.gamma, 'tau': self.tau}, path)
    def load(self, path):
        try: ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError: ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt['model_state_dict'])
        self.target.load_state_dict(ckpt.get('target_state_dict', ckpt['model_state_dict']))
        if 'optimizer_state_dict' in ckpt:
            try: self.opt.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception: pass
        self.q.to(self.device); self.target.to(self.device); self.target.eval()
