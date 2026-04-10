import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from utils.price_loader import load_caiso_prices, create_rgba_grid
from envs.chargetrek_env import ChargeTrekEnv
from agents.c51_agent import C51Agent
from agents.dagger_dqn_agent import DQNAgent
import gymnasium as gym
import torch
import copy 


#Choose day 0 --> 135

option = {"day":0,"user": [16, 24, 90, 16 * 4]}
# Load data and environment
price_df = load_caiso_prices("data", node_name="SMD4_ASR-APND LMP")
grid     = create_rgba_grid(price_df, steps=96, soc_levels=101, day=0)
#print(grid.shape)  # Should be (96, 101, 4)
env = ChargeTrekEnv(grid, price_df)

# ----- 2. create agent & load checkpoint -----------------------------
strategy = 'dagger'  # 'optimal', 'stepwise','dagger', 'dump' or 'ib_c51'



if strategy == 'dagger':
    agent = DQNAgent(env, buffer_path = None)
    device = next(agent.net.parameters()).device  
    ckpt = torch.load("checkpoints/dagger_dqn_latest.pt", map_location=device,weights_only=False)
    agent.net.load_state_dict(ckpt["policy"])   # policy weights only
    agent.net.eval()   
    obs, _ = env.reset(options=option)
    strategy = 'dqn'

elif strategy == 'ib_c51':
    agent = C51Agent(env, buffer_path = None)
    device = next(agent.net.parameters()).device
    ckpt = torch.load("checkpoints/ib_c51.pt", map_location=device,weights_only=False)
    agent.net.load_state_dict(ckpt["policy"])   # policy weights only
    agent.net.eval()   
    obs, _ = env.reset(options=option)
    strategy = 'dqn'

else:

    agent = C51Agent(env, buffer_path = None)
    device = next(agent.net.parameters()).device
    obs, _ = env.reset(options=option)


# Precompute optimal actions
path, optimal_actions, cost = env.full_replay_benchmark(key='real', render=False, strategy=strategy, agent = agent) #strategy = optimal for magic solution or stepwise or dqn
current_optimal_step = [0]
print('total cost', cost)
obs, _ = env.reset(options=option) # reset because the env has been used by reply benchmark when selecting dqn strategy, this will avoid showing the final results directly
reward_total = list()
# Setup figure
plt.ion()
fig, ax = plt.subplots(figsize=(25, 25))

def redraw(reward,reward_avg):
    ax.clear()
    img = env.render(return_image=True)
    ax.imshow(img, aspect='auto')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(np.arange(img.shape[1]))
    ax.set_xticklabels(np.arange(img.shape[1]))
    ax.set_yticks(np.arange(img.shape[0]))
    ax.set_yticklabels(np.arange(img.shape[0] - 1, -1, -1))
    ax.set_title(f"SoC: {env.current_soc:.1f}%, Time: {(env.current_time/4 + env.arrival_time)}, Money: {env.money:.2f}, Reward: {reward:.2f}, Total reward: {reward_avg:.2f}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("SoC Level")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    
    

def on_key(event):
    if event.key in ['0', '1', '2']:
        action = int(event.key)
    elif event.key == 'm':
        if current_optimal_step[0] < len(optimal_actions):
            action = optimal_actions[current_optimal_step[0]]
            print(f"🔁 Optimal step {current_optimal_step[0]}: Action {action}")
            current_optimal_step[0] += 1
        else:
            print("✅ Optimal trajectory completed.")
            return
    elif event.key == 'q':
        print("❌ Quit simulation.")
        plt.close()
        return
    else:
        return

    obs, reward, done, _, info = env.step(action)
    reward_total.append(reward)
    reward_avg = np.sum(reward_total)
    redraw(reward,reward_avg)

    if done:
        print(f"✅ Simulation ended. Final money: {env.money:.2f}, reward: {reward:.2f}, Total reward: {reward_avg:.2f}")
        print(f"Expected final: {path[-1]}, actual: ({env.current_time}, {int(env.current_soc)})")
        redraw(reward,reward_avg)
        plt.pause(0.1)
        plt.close()

redraw(0,0)
fig.canvas.mpl_connect('key_press_event', on_key)

import tkinter as tk
tk.mainloop()
