"""c51_agent.py
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
V_MIN, V_MAX = -10, 10          # value support range ($/kWh)
GAMMA        = 0.99
LR           = 1e-4
BATCH_SIZE   = 256
MEM_CAPACITY = 1_000_000
TARGET_SYNC  = 500
MAP_SIZE = (MEM_CAPACITY *1e-5 * 10) * 1e9        # ring‑buffer capacity on disk
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

# ---------------- C51 Network --------------------------------------------- #
class C51Net(nn.Module):
    """Categorical DQN network with 3 conv layers + 2 FC layers.
       Designed for input size (4, 101, 96)."""
    def __init__(self, n_actions: int):
        super().__init__()
        self.n_atoms = N_ATOMS
        self.n_actions = n_actions

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
    def __init__(self, env, buffer_path = None, vmin=V_MIN, vmax=V_MAX):
        self.env = env
        self.actions = env.action_space.n

        self.net  = C51Net(self.actions).to(DEVICE)
        self.tgt  = C51Net(self.actions).to(DEVICE)
        self.tgt.load_state_dict(self.net.state_dict()); self.tgt.eval()

        self.opt  = optim.Adam(self.net.parameters(), lr=LR)
        self.vmin = vmin
        self.vmax = vmax
        #self.mem  = ReplayBuffer(MEM_CAPACITY)
        if buffer_path is not None:
            self.mem  = DiskReplayBuffer(path = buffer_path,capacity=MEM_CAPACITY, map_size = int(MAP_SIZE))

        self.support = torch.linspace(self.vmin, self.vmax, N_ATOMS).to(DEVICE)
        self.dz      = (self.vmax - self.vmin)/(N_ATOMS-1)
        self.steps   = 0
    
    @staticmethod
    def prep(obs):
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs.transpose(2,1,0)).float() / 255.0
        elif torch.is_tensor(obs):
            return obs.float()
        else:
            raise TypeError(f"Unsupported obs type {type(obs)}")
    

    def ε(self): return max(EPS_END, EPS_START - self.steps/EPS_DECAY)

    def act(self, state: torch.Tensor, validation = False) -> int:
        self.steps += 1
        self.net.eval()  # Ensure deterministic behavior
        if not validation:
            if random.random() < self.ε(): return random.randrange(self.actions)
        with torch.no_grad():
            logits = self.net(state.unsqueeze(0).to(DEVICE))
            probs  = torch.softmax(logits, dim=2)
            q      = (probs * self.support).sum(2)
        return int(q.argmax(1))
    """
    def project(self, next_dist, r, d):
        d = d.float()
        B = r.size(0)
        proj = torch.zeros(B, N_ATOMS, device=DEVICE)

        for b in range(B):
            for j in range(N_ATOMS):
                tz = torch.clamp(r[b] + GAMMA * (1 - d[b]) * self.support[j], self.vmin, self.vmax)
                bj = (tz - self.vmin) / self.dz
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
        """
    


    def project(self, next_dist, r, d):
        """
        Vectorized distribution projection step for C51.
        next_dist: [B, N_ATOMS]
        r: [B, 1]
        d: [B, 1] (float)
        """
        batch_size = r.size(0)
        support = self.support.unsqueeze(0)  # [1, N_ATOMS]
        delta_z = self.dz

        # Compute projected supports (tz)
        tz = r + (1 - d) * GAMMA * support  # [B, N_ATOMS]
        tz = tz.clamp(self.vmin, self.vmax)

        # Compute projection bin positions
        b = (tz - self.vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l = l.clamp(0, N_ATOMS - 1)
        u = u.clamp(0, N_ATOMS - 1)

        proj_dist = torch.zeros(batch_size, N_ATOMS, device=DEVICE)

        offset = torch.linspace(0, (batch_size - 1) * N_ATOMS, batch_size, device=DEVICE).long().unsqueeze(1)

        next_dist = next_dist.clamp(min=1e-8)  # Prevent log(0) or NaNs

        # Distribute probability mass to l and u
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1),
            (next_dist * (u.float() - b)).view(-1).float()
        )

        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1),
            (next_dist * (b - l.float())).view(-1).float()
        )

        return proj_dist

        
    def optimise(self):
        if len(self.mem) < BATCH_SIZE: return
        batch = self.mem.sample(BATCH_SIZE)
        s  = torch.stack([self.prep(o).to(DEVICE) for o in batch.state])
        ns = torch.stack([self.prep(o).to(DEVICE) for o in batch.next_state])
        a  = torch.tensor(batch.action, device=DEVICE).unsqueeze(1)
        r  = torch.tensor(batch.reward, device=DEVICE).unsqueeze(1)
        d  = torch.tensor(batch.done,   device=DEVICE, dtype=torch.float32).unsqueeze(1)

        logits = self.net(s)
        probs  = torch.softmax(logits, dim=2)
        dist   = probs.gather(1, a.unsqueeze(-1).expand(-1, -1, N_ATOMS)).squeeze(1)

        with torch.no_grad():
            next_logits = self.net(ns)
            next_probs  = torch.softmax(next_logits, dim=2)
            q_next      = (next_probs * self.support).sum(2)
            na          = q_next.argmax(1)
            tgt_logits  = self.tgt(ns)
            tgt_probs   = torch.softmax(tgt_logits, dim=2)
            next_dist   = tgt_probs[range(BATCH_SIZE), na]
            target_dist = self.project(next_dist, r, d)

        loss = -(target_dist * torch.log(dist + 1e-8)).sum(1).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.item())
        
    
    
    def train(self, total_steps=300_000):
        state = self.prep(self.env.reset()[0])
        for t in trange(total_steps, desc="C51 train"):
            a = self.act(state)
            ns, r, done, _, _ = self.env.step(a)
            ns_t = self.prep(ns)
            self.mem.push(state.cpu(), a, r, ns_t.cpu(), done)
            state = ns_t if not done else self.prep(self.env.reset()[0])
            self.optimise()
            if t % TARGET_SYNC == 0:
                self.tgt.load_state_dict(self.net.state_dict())
