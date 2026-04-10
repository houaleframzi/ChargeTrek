"""dagger_dqn_agent.py
Standard DQN agent with **DAgger** support for the ChargeTrek environment.
• Observation : RGBA grid (96×101×4) ➔ tensor (4,101,96)
• Action space: {0=charge, 1=discharge, 2=idle}
• Replay buffer : LMDB‑backed (DiskReplayBuffer)

Key DAgger behaviour
--------------------
1. **Data collection**: push `(state, expert_action)` into the buffer (reward optional).
2. **Supervised update**: sample a mini‑batch from that buffer and do a plain
   cross‑entropy step so the network imitates the expert labels.
3. RL updates (`optimise`) are still available but *optional*.
"""

import random, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from utils.disk_replay_buffer import DiskReplayBuffer

# ─────────────────── Hyper‑parameters ───────────────────────── #
LR           = 1e-4
BATCH_SIZE   = 256
MEM_CAPACITY = 1_000_000
MAP_SIZE = (MEM_CAPACITY *1e-5 * 5) * 1e9        # ring‑buffer capacity on disk
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────── Network (DQN) ──────────────────────────── #
class DQNNet(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4,  32, kernel_size=8, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 256, kernel_size=21, stride=1, padding=0), 
            nn.ReLU(),  
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 101, 96)
            feat_size = self.features(dummy).shape[-1]
        self.fc = nn.Sequential(nn.Linear(feat_size, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        return self.fc(self.features(x))   # (B, A)

# ─────────────────── Agent wrapper ──────────────────────────── #
class DQNAgent:
    def __init__(self, env, buffer_path=None):
        self.env     = env
        self.actions = env.action_space.n

        self.net = DQNNet(self.actions).to(DEVICE)
        self.tgt = DQNNet(self.actions).to(DEVICE)
        self.tgt.load_state_dict(self.net.state_dict()); self.tgt.eval()

        self.opt = optim.Adam(self.net.parameters(), lr=LR)
        if buffer_path is not None:
            
            self.mem = DiskReplayBuffer(path=buffer_path, capacity=MEM_CAPACITY, map_size = int(MAP_SIZE), mode='dagger')

        self.steps = 0

    # ------------------------ Helpers ----------------------- #
    @staticmethod
    def prep(obs):
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs.transpose(2,1,0)).float() / 255.0
        elif torch.is_tensor(obs):
            return obs.float()
        raise TypeError(f"Unsupported obs type {type(obs)}")

    # ------------------------ Act --------------------------- #
    def act(self, state, validation=False):
        self.steps += 1
        self.net.eval()  # Ensure deterministic behavior
        with torch.no_grad():
            q = self.net(state.unsqueeze(0).to(DEVICE))  # (1, A)
        return int(q.argmax(1))

    # ----------------------- DAgger push -------------------- #
    def store_dagger(self, state, expert_action):
        """Save (state, expert_action) to disk buffer."""
        self.mem.push(state, expert_action)  # reward/next/done placeholders

    # --------------------- Supervised step ------------------ #
    def dagger_update(self):
        if len(self.mem) < BATCH_SIZE:
            return None
        batch = self.mem.sample(BATCH_SIZE)
        states = torch.stack([self.prep(o).to(DEVICE) for o in batch.state])
        labels = torch.tensor(batch.action, device=DEVICE)  # expert actions were stored as `action`
        self.opt.zero_grad()
        logits = self.net(states)                # (B, A)
        loss   = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        self.opt.step()
        return float(loss.item())
    def dagger_eval_loss(self):
        """
        Like dagger_update, but only computes & returns the loss on a sampled batch—
        no zero_grad, backward, or optimizer.step. Useful for validation/test.
        """
        if len(self.mem) < BATCH_SIZE:
            return None

        # sample a batch
        batch = self.mem.sample(BATCH_SIZE)
        # prepare inputs
        states = torch.stack([self.prep(o).to(DEVICE) for o in batch.state])
        labels = torch.tensor(batch.action, device=DEVICE)

        # switch to eval (e.g. disable dropout/batchnorm updates)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(states)               # (B, A)
            loss   = nn.CrossEntropyLoss()(logits, labels)

        # back to train mode if you’ll train elsewhere
        self.net.train()

        return float(loss.item())


