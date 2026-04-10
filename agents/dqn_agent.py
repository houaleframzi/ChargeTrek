"""
dqn_agent.py
Standard DQN agent for the ChargeTrek environment.
- Observation : RGBA grid (96×101×4)   -> tensor (4,101,96)
- Action space: {0=charge, 1=discharge, 2=idle}
"""

import random, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
from tqdm import trange
from utils.disk_replay_buffer import DiskReplayBuffer

# ---------------- Hyper-parameters ---------------------------------------- #
GAMMA        = 0.99
LR           = 1e-4
BATCH_SIZE   = 256
MEM_CAPACITY = 1_000_000
TARGET_SYNC  = 500
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.02, 1_000_000
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

# ---------------- DQN Network --------------------------------------------- #
class DQNNet(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4,  32, kernel_size=8, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 256, kernel_size=21, stride=1, padding=0), nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 101, 96)
            feat_size = self.features(dummy_input).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(feat_size, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))

# ---------------- Full Agent wrapper -------------------------------------- #
class DQNAgent:
    def __init__(self, env, buffer_path = 'G:/dqn_buffer'):
        self.env = env
        self.actions = env.action_space.n

        self.net  = DQNNet(self.actions).to(DEVICE)
        self.tgt  = DQNNet(self.actions).to(DEVICE)
        self.tgt.load_state_dict(self.net.state_dict()); self.tgt.eval()

        self.opt  = optim.Adam(self.net.parameters(), lr=LR)
        #self.mem  = ReplayBuffer(MEM_CAPACITY)
        if buffer_path is not None:
            self.mem  = DiskReplayBuffer(path = buffer_path,capacity=MEM_CAPACITY)
        
        self.steps   = 0

    @staticmethod
    def prep(obs):
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs.transpose(2,1,0)).float() / 255.0
        elif torch.is_tensor(obs):
            return obs.float()
        else:
            raise TypeError(f"Unsupported obs type {type(obs)}")

    def ε(self):
        return max(EPS_END, EPS_START - self.steps/EPS_DECAY)

    def act(self, state: torch.Tensor, validation = False) -> int:
        self.steps += 1
        if not validation and random.random() < self.ε():
            return random.randrange(self.actions)
        with torch.no_grad():
            q_values = self.net(state.unsqueeze(0).to(DEVICE))
        return int(q_values.argmax(1))

    def optimise(self):
        if len(self.mem) < BATCH_SIZE: return
        batch = self.mem.sample(BATCH_SIZE)
        s  = torch.stack([self.prep(o).to(DEVICE) for o in batch.state])
        ns = torch.stack([self.prep(o).to(DEVICE) for o in batch.next_state])
        a  = torch.tensor(batch.action, device=DEVICE).unsqueeze(1)
        r  = torch.tensor(batch.reward, device=DEVICE).unsqueeze(1)
        d  = torch.tensor(batch.done,   device=DEVICE, dtype=torch.float32).unsqueeze(1)

        q_values      = self.net(s).gather(1, a)
        with torch.no_grad():
            next_q = self.tgt(ns).max(1, keepdim=True)[0]
            target = r + (1 - d) * GAMMA * next_q

        loss = nn.MSELoss()(q_values, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.item())

    def train(self, total_steps=300_000):
        state = self.prep(self.env.reset()[0])
        for t in trange(total_steps, desc="DQN train"):
            a = self.act(state)
            ns, r, done, _, _ = self.env.step(a)
            ns_t = self.prep(ns)
            self.mem.push(state.cpu(), a, r, ns_t.cpu(), done)
            state = ns_t if not done else self.prep(self.env.reset()[0])
            self.optimise()
            if t % TARGET_SYNC == 0:
                self.tgt.load_state_dict(self.net.state_dict())
