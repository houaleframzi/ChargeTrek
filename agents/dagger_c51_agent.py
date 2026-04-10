"""dagger_c51_agent.py
Categorical DQN (C51) agent for the ChargeTrek environment.
- Observation : RGBA grid (96×101×4)   -> tensor (4,101,96)
- Action space: {0=charge, 1=discharge, 2=idle}
"""

import random, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
from tqdm import trange
from utils.disk_replay_buffer import DiskReplayBuffer

# ---------------- Hyper‑parameters ---------------------------------------- #
N_ATOMS      = 51
V_MIN, V_MAX = -5, 5          # value support range ($/kWh)
GAMMA        = 0.99
LR           = 1e-4
BATCH_SIZE   = 256
MEM_CAPACITY = 1_000_000


DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

# ---------------- Replay buffer ------------------------------------------- #
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.mem = deque(maxlen=capacity)
    def push(self, *args):
        self.mem.append(Transition(*args))
    def sample(self, n: int):
        batch = random.sample(self.mem, n)
        return Transition(*zip(*batch))
    def __len__(self): return len(self.mem)

# ---------------- C51 Network --------------------------------------------- #
class C51Net(nn.Module):
    """Categorical DQN network with 3 conv layers + 2 FC layers.
       Designed for input size (4, 101, 96)."""
    def __init__(self, n_actions: int):
        super().__init__()
        self.n_atoms = N_ATOMS
        self.n_actions = n_actions

        self.features = nn.Sequential(
            nn.Conv2d(4,  32, kernel_size=8, stride=2, padding=1), nn.ReLU(),   # Output ≈ (32, 48, 47)
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),   # Output ≈ (32, 24, 24)
            nn.Conv2d(32, 256, kernel_size=21, stride=1, padding=0), nn.ReLU(), # Output ≈ (256, 4, 4)
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 101, 96)
            feat_size = self.features(dummy_input).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(feat_size, 512), nn.ReLU(),
            nn.Linear(512, n_actions * N_ATOMS)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(self.features(x))
        return logits.view(-1, self.n_actions, N_ATOMS)


# ---------------- Full Agent wrapper -------------------------------------- #
class C51Agent:
    def __init__(self, env, buffer_path = 'G:/dqn_buffer'):
        self.env = env
        self.actions = env.action_space.n

        self.net  = C51Net(self.actions).to(DEVICE)
        self.tgt  = C51Net(self.actions).to(DEVICE)
        self.tgt.load_state_dict(self.net.state_dict()); self.tgt.eval()

        self.opt  = optim.Adam(self.net.parameters(), lr=LR)
        #self.mem  = ReplayBuffer(MEM_CAPACITY)
        if buffer_path is not None:
            self.mem  = DiskReplayBuffer(path = buffer_path, capacity=MEM_CAPACITY, mode = 'dagger')
            
        self.support = torch.linspace(V_MIN, V_MAX, N_ATOMS).to(DEVICE)
        self.dz      = (V_MAX - V_MIN)/(N_ATOMS-1)
        self.steps   = 0

    @staticmethod
    def prep(obs):
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs.transpose(2,1,0)).float() / 255.0
        elif torch.is_tensor(obs):
            return obs.float()
        else:
            raise TypeError(f"Unsupported obs type {type(obs)}")



    def act(self, state: torch.Tensor, validation = False) -> int:
        self.steps += 1
        with torch.no_grad():
            logits = self.net(state.unsqueeze(0).to(DEVICE))
            probs  = torch.softmax(logits, dim=2)
            q      = (probs * self.support).sum(2)
        return int(q.argmax(1))

    def project(self, next_dist, r, d):
        d = d.float()
        B = r.size(0)
        proj = torch.zeros(B, N_ATOMS, device=DEVICE)

        for b in range(B):
            for j in range(N_ATOMS):
                tz = torch.clamp(r[b] + GAMMA * (1 - d[b]) * self.support[j], V_MIN, V_MAX)
                bj = (tz - V_MIN) / self.dz
                l  = int(torch.floor(bj))
                u  = int(torch.ceil(bj))

                val = next_dist[b, j].item()
                if l == u:
                    proj[b, l] += val
                else:
                    wl = (u - bj).item()
                    wu = (bj - l).item()
                    proj[b, l] += val * wl
                    proj[b, u] += val * wu
        return proj


       # ----------------------- DAgger push -------------------- #
    def store_dagger(self, state, expert_action):
        """Save (state, expert_action) to disk buffer."""
        self.mem.push(state.cpu(), expert_action)  # reward/next/done placeholders

    def supervised_update(self, states, expert_actions):
        """
        DAgger supervised learning step.
        states: Tensor of shape (batch_size, 4, 101, 96)
        expert_actions: Tensor of shape (batch_size,), containing optimal actions
        """
        self.opt.zero_grad()
        logits = self.net(states)
        # Calculate cross-entropy loss over actions
        action_logits = logits.mean(dim=2)  # Reduce atom dimension for classification
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(action_logits, expert_actions)
        loss.backward()
        self.opt.step()
        return loss.item()

