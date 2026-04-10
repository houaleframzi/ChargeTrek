"""
train_c51_burst.py
──────────────────
* Collect episodes in blocks (≈ 38k transitions)
* Then run optimiser steps as one learning burst
* Log and checkpoint once per block
"""

# ───────────────────────── Imports ──────────────────────────────
import os, csv, shutil, tempfile, torch
from pathlib import Path
import numpy as np
from tqdm import tqdm as base_tqdm
from utils.price_loader import load_caiso_prices, create_rgba_grid
from envs.chargetrek_env import ChargeTrekEnv
from agents.c51_agent     import C51Agent, DEVICE
#import matplotlib.pyplot as plt
from utils.charge_trek_multigraph import build_charge_trek_multigraph
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import copy
import  functools
import os, shutil
from utils.disk_replay_buffer import Transition, safe_copy_lmdb
import random
import json
tqdm= functools.partial(base_tqdm, disable=False)



# Paths
FAST_BUFFER_PATH  = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "c51_buffer"
PERM_BUFFER_PATH  = Path("buffers/c51_buffer")
print("Using fast buffer path:", FAST_BUFFER_PATH)
print("Using permanent buffer path:", PERM_BUFFER_PATH)

# Restore previous buffer from project dir if it exists
if not FAST_BUFFER_PATH.exists():
    print("🚚 No buffer on SSD, creating path")
    FAST_BUFFER_PATH.mkdir(parents=True, exist_ok=True)
    print('fast buffer path created')
    if PERM_BUFFER_PATH.exists():
        print(f"📥 Copying buffer from {PERM_BUFFER_PATH} to SSD {FAST_BUFFER_PATH}...")
        shutil.copytree(PERM_BUFFER_PATH, FAST_BUFFER_PATH, dirs_exist_ok=True)




# ───────────────────────── Hyper-params ─────────────────────────
COLLECT_EPISODES      = 25    # episodes per data-collection phase
OPT_STEPS_PER_BURST   = 100      # optimiser steps per burst
MAX_UPDATES_SYNC_INTERVAL   = 500    # target net sync interval (in update steps)
TEST_COLLECT_EPISODES = 25
#BATCH_SIZE            =      # (used inside c51_agent.py)
TOTAL_STEPS           = 2_000_000
PRINT_EVERY_BLOCK     = 1
CHECK_BLOCK_INTERVAL  = 1
BUFFER_SYNC_INTERVAL  = 10
V_MIN, V_MAX = -15, 15          # value support range ($/kWh)

# ───────── Logging / checkpoint folders ─────────────────────────
LOG_DIR, CKPT_DIR = Path("logs"), Path("checkpoints/c51_burst")
LOG_DIR.mkdir(parents=True,exist_ok=True)
CKPT_DIR.mkdir(parents=True,exist_ok=True)
ckpt_path = CKPT_DIR / "latest.pt"
bkp_path  = CKPT_DIR / "prev_latest.pt"

def atomic_save(obj, path: Path):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        torch.save(obj, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)

def safe_list(x):
    """Convert NumPy arrays/scalars or Python lists to a JSON-safe Python list."""
    return np.array(x).tolist()


# ───────── Build environment & agent ────────────────────────────

# Create the list
data = list(range(1, 129))

# Shuffle
np.random.seed(42)  # reproducibility
np.random.shuffle(data)

# Split index
split_idx = int(len(data) * 0.8)

train = data[:split_idx]
test  = data[split_idx:]

price_df = load_caiso_prices("data/", node_name="SMD4_ASR-APND LMP")
grid     = create_rgba_grid(price_df, steps=96, soc_levels=101, day=0)
#print(grid.shape)  # Should be (96, 101, 4)
env      = ChargeTrekEnv(grid, price_df, dayslist=train)
validation_env = ChargeTrekEnv(grid, price_df, dayslist=train)
test_env = ChargeTrekEnv(grid, price_df,dayslist=test)
hybrid_test_env = ChargeTrekEnv(grid, price_df,dayslist=test)
hybrid_env = ChargeTrekEnv(grid, price_df, dayslist=train)
agent    = C51Agent(env, buffer_path = str(FAST_BUFFER_PATH), vmin=V_MIN, vmax=V_MAX)  # Disk buffer, mode="dagger" inside agent
validation_agent = C51Agent(validation_env, buffer_path = None, vmin=V_MIN, vmax=V_MAX)
test_agent = C51Agent(test_env, buffer_path = None, vmin=V_MIN, vmax=V_MAX)
hybrid_test_agent = C51Agent(hybrid_test_env, buffer_path = None, vmin=V_MIN, vmax=V_MAX)
hybrid_agent = C51Agent(hybrid_env, buffer_path = None, vmin=V_MIN, vmax=V_MAX)



# ───────── Resume from checkpoint if available ─────────────────
start_block = 0
if ckpt_path.exists():
    print("🔄 Resuming from", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    agent.net.load_state_dict(ckpt["policy"])
    agent.tgt.load_state_dict(ckpt["target"])
    agent.opt.load_state_dict(ckpt["optimizer"])
    #agent.mem.mem.clear()
    #agent.mem.mem.extend(ckpt["buffer"])
    agent.steps = ckpt["steps"]
    start_block = ckpt["episode"] + 1
else:
    print("🚀 Starting fresh training")


validation_agent.net.load_state_dict(agent.net.state_dict())
test_agent.net.load_state_dict(agent.net.state_dict())
hybrid_test_agent.net.load_state_dict(agent.net.state_dict())
hybrid_agent.net.load_state_dict(agent.net.state_dict())

# ───────── CSV logger (append mode) ─────────────────────────────
log_path = LOG_DIR / "train_log.csv"
new_file = not log_path.exists()
with log_path.open("a", newline="") as flog:
    logger = csv.writer(flog)
    if new_file:
        logger.writerow(["block_idx", "steps", "avg_reward",
                         "avg_loss_burst", "avg_arrival_rate","epsilon", "avg_gap", "avg_optimal_cost","std_soc_gap",
                          "avg_stepwise_cost", "avg_rl_cost", "avg_agent_soc", "avg_req_soc", "avg_dump_charger_cost",
                          "avg_validation_reward", "avg_validation_arrival_rate", "avg_validation_agent_cost", "avg_validation_agent_soc", "avg_validation_soc_gap","std_validation_soc_gap",
                          "avg_test_reward", "avg_test_arrival_rate", "avg_test_agent_cost", "avg_test_agent_soc", "avg_test_req_soc", "avg_test_soc_gap", "avg_test_optimal_cost","std_test_soc_gap",
                            "avg_test_stepwise_cost", "avg_test_dump_charger_cost",
                            "avg_hybrid_test_reward", "avg_hybrid_test_arrival_rate", "avg_hybrid_test_agent_cost", "avg_hybrid_test_agent_soc", "avg_hybrid_test_soc_gap",
                            "avg_hybrid_reward", "avg_hybrid_arrival_rate", "avg_hybrid_agent_cost", "avg_hybrid_agent_soc", "avg_hybrid_soc_gap",
                            
                            "raw_episode_list","raw_steps_list",
                            "raw_rewards_block","raw_losses_burst","raw_arrival_rate","raw_optimal_costs_block","raw_stepwise_costs_block","raw_agent_costs_block","raw_agent_soc_block","raw_req_soc_block","raw_soc_gap_block","raw_dump_charger_cost_block",
                            "raw_validation_rewards_block","raw_validation_arrival_rate","raw_validation_agent_costs_block","raw_validation_agent_soc_block","raw_validation_soc_gap_block",
                            "raw_test_rewards_block","raw_test_arrival_rate","raw_test_optimal_costs_block","raw_test_stepwise_costs_block","raw_test_agent_costs_block","raw_test_agent_soc_block","raw_test_req_soc_block","raw_test_soc_gap_block","raw_test_dump_charger_cost_block",
                            "raw_hybrid_rewards_block","raw_hybrid_arrival_rate","raw_hybrid_agent_costs_block","raw_hybrid_agent_soc_block","raw_hybrid_soc_gap_block",
                            "raw_hybrid_test_rewards_block","raw_hybrid_test_arrival_rate","raw_hybrid_test_agent_costs_block","raw_hybrid_test_agent_soc_block","raw_hybrid_test_soc_gap_block"])

    # ─────── Main loop: one “block” = episodes + updates ──────
    global_step = agent.steps
    updates_since_sync = 0
    #fig, ax = plt.subplots(figsize=(25, 25))
    for block in range(start_block, 1_000_000):

        # =========== 1. DATA COLLECTION ========================
        rewards_block = []
        optimal_costs_block = []
        stepwise_costs_block = []
        agent_costs_block = []
        agent_soc_block = []
        req_soc_block = []
        arrival_rate = []
        soc_gap_block = []
        dump_charger_cost_block = []
        #-------------------------------------------------------------
        validation_rewards_block = []
        validation_arrival_rate = []
        validation_agent_costs_block = []
        validation_agent_soc_block = []
        validation_soc_gap_block = []
        #----------------------------------------------------------------------
        test_rewards_block = []
        test_optimal_costs_block = []
        test_stepwise_costs_block = []
        test_agent_costs_block = []
        test_agent_soc_block = []
        test_req_soc_block = []
        test_arrival_rate = []
        test_soc_gap_block = []
        test_dump_charger_cost_block = []
        #----------------------------------------------------------------------
        hybrid_test_rewards_block = []
        hybrid_test_arrival_rate = []
        hybrid_test_agent_costs_block = []
        hybrid_test_agent_soc_block = []
        hybrid_test_soc_gap_block = []

        #----------------------------------------------------------------------
        hybrid_rewards_block = []
        hybrid_arrival_rate = []
        hybrid_agent_costs_block = []
        hybrid_agent_soc_block = []
        hybrid_soc_gap_block = []

        episode_buffer = []

        episode_list = []
        steps_list = []
        episode = 0
        USER_PROFILES = env.users_pool
        for _ in tqdm(range(COLLECT_EPISODES), desc=f"C51 Collecting block {block}"):

            user = USER_PROFILES[_ % len(USER_PROFILES)].copy()
            user[0], user[-1] = user[0] + random.choice([-0.25, 0.25]), user[-1] + random.choice([-1, 1])
            obs, _ = env.reset(options={"user": user}); state = agent.prep(obs)
            
        
            done, ep_reward = False, 0.0
            #graph = build_charge_trek_multigraph(env.price_df, day=env.day, arrival_time= env.arrival_time)
            validation_obs, _ = validation_env.reset(options={"day": agent.env.day, "user": agent.env.profile})# reset validation env with same day and user profile
            validation_state = validation_agent.prep(validation_obs)
            validation_done, validation_ep_reward = False, 0.0

            hybrid_obs, _ = hybrid_env.reset(options={"day": env.day, "user": env.profile})# reset  env with same day and user profile
            hybrid_state = hybrid_agent.prep(hybrid_obs)
            hybrid_done, hybrid_ep_reward = False, 0.0
            
            
            #print("Graph nodes:", graph.nodes())
            #print( "env pos", env.agent_pos)
            #print("arrival time", env.arrival_time)
            episode += 1
            with ThreadPoolExecutor() as executor:
                f1 = executor.submit(env.helper_replay_benchmark, env.graph)
                f2 = executor.submit(env.helper_replay_benchmark, env.graph, strategy='stepwise')
                f3 = executor.submit(env.dump_charger)

                _, _, optimal_costs_episode = f1.result()
                _, _, stepwise_costs_episode = f2.result()
                _, _, dump_charger_cost = f3.result()
                
            """
            _, _, optimal_costs_episode = env.helper_replay_benchmark(graph, start_node=env.agent_pos, strategy='optimal')
            _, _, stepwise_costs_episode = env.helper_replay_benchmark(graph, start_node=env.agent_pos, strategy='stepwise')
            _, _, dump_charger_cost = env.dump_charger()
            """

            check = True
            found_list_of_actions = False
            hybrid_action_idx = 0
            hybrid_actions = []
                
            if  optimal_costs_episode == np.inf:
                print("day:", env.day, "dump charger cost:", dump_charger_cost, "optimal cost:", optimal_costs_episode, "profile", env.profile)
            #print("Dump charger cost:", dump_charger_cost)
            
            
            while not done and not hybrid_done and global_step < TOTAL_STEPS:


                act = agent.act(state)
                nxt, r, done, _, data = env.step(act)
                has_arrived = data['reached_goal']
            
                #agent.mem.push(obs, act, r, nxt, done)
                episode_buffer.append(Transition(obs, act, r, nxt, done))
                obs, state = (nxt, agent.prep(nxt)) if not done else (None, None)
                ep_reward += r
                global_step += 1
#---------------------------------------------------------
                validation_act = validation_agent.act(validation_state, validation=True)# set validation to True to avoid using epsilon-greedy
                validation_nxt, validation_r, validation_done, _, validation_data = validation_agent.env.step(validation_act)
                validation_has_arrived = validation_data['reached_goal']
                validation_obs, validation_state = (validation_nxt, validation_agent.prep(validation_nxt)) if not validation_done else (None, None)
                validation_ep_reward += validation_r
                
            
                """
                plt.ion()
                ax.clear()
                img = env.render(return_image=True)
                ax.imshow(img, aspect='auto')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.set_xticks(np.arange(img.shape[1]))
                ax.set_xticklabels(np.arange(img.shape[1]))
                ax.set_yticks(np.arange(img.shape[0]))
                ax.set_yticklabels(np.arange(img.shape[0] - 1, -1, -1))
                ax.set_title(f"SoC: {env.current_soc:.1f}%, Time: {(env.current_time/4 + env.arrival_time)}, Money: {env.money:.2f}, Reward: {r:.2f}, episode reward: {ep_reward:.2f}")
                ax.set_xlabel("Time Step")
                ax.set_ylabel("SoC Level")
                plt.tight_layout()
                plt.draw()
                plt.pause(0.0001)
                """
                # -- Hybrid fallback agent --
                if check:
                    # Try normal agent policy
                    hybrid_act = hybrid_agent.act(hybrid_state, validation=True)
                    feas = hybrid_env.smart_feasibility_check(hybrid_act)
                    if feas == 0:
                        check = False  # Enter fallback mode

                if not check:
                    if not found_list_of_actions:
                        # Compute fallback plan once
                        _, hybrid_actions, _ = hybrid_env.helper_replay_benchmark(
                            strategy='stepwise',
                            start_node=hybrid_env.agent_pos,
                            graph=hybrid_env.graph
                        )
                        found_list_of_actions = True
                        hybrid_action_idx = 0

                    # Use fallback plan
                    if hybrid_action_idx < len(hybrid_actions):
                        hybrid_act = hybrid_actions[hybrid_action_idx]
                        hybrid_action_idx += 1
                    else:
                        # Backup safety: if we run out of fallback actions
                        if hybrid_env.current_soc < hybrid_env.goal_soc:
                            hybrid_act = 0
                        elif hybrid_env.current_soc > hybrid_env.goal_soc:
                            hybrid_act = 1
                        else:
                            hybrid_act = 2
                
                # Step hybrid agent
                hybrid_nxt, hybrid_r, hybrid_done, _, hybrid_data = hybrid_env.step(hybrid_act)
                hybrid_has_arrived = hybrid_data['reached_goal']
                hybrid_obs, hybrid_state = (hybrid_nxt, hybrid_agent.prep(hybrid_nxt)) if not hybrid_done else (None, None)
                hybrid_ep_reward += hybrid_r


            


           
            
            #---------------------------------------------------------
            rewards_block.append(ep_reward)
            arrival_rate.append(1 if has_arrived else 0)
            agent_costs_block.append(env.money)
            agent_soc_block.append(env.current_soc)
            req_soc_block.append(env.goal_soc)
            soc_gap_block.append(env.current_soc - env.goal_soc)
            optimal_costs_block.append(optimal_costs_episode)
            stepwise_costs_block.append(stepwise_costs_episode)
            dump_charger_cost_block.append(dump_charger_cost)

            # Validation episode
            validation_rewards_block.append(validation_ep_reward)
            validation_arrival_rate.append(1 if validation_has_arrived else 0)
            validation_agent_costs_block.append(validation_agent.env.money)
            validation_agent_soc_block.append(validation_agent.env.current_soc)
            validation_soc_gap_block.append(validation_agent.env.current_soc - validation_agent.env.goal_soc)

            # Hybrid  episode
            
            hybrid_rewards_block.append(hybrid_ep_reward)
            hybrid_arrival_rate.append(1 if hybrid_has_arrived else 0) 
            hybrid_agent_costs_block.append(hybrid_agent.env.money)
            hybrid_agent_soc_block.append(hybrid_agent.env.current_soc)
            hybrid_soc_gap_block.append(hybrid_agent.env.current_soc - hybrid_agent.env.goal_soc)

            episode_list.append(episode)
            steps_list.append(global_step)

        agent.mem.push_many(episode_buffer)
        USER_PROFILES = env.users_pool
        #------------------test episode-----------------------------
        for _ in tqdm(range((TEST_COLLECT_EPISODES)), desc = 'C51 Testing on Test DATA' ): #15 episodes per day since we have 15 users 
           
            user = USER_PROFILES[_ % len(USER_PROFILES)].copy()
            user[0], user[-1] = user[0] + random.choice([-0.25, 0.25]), user[-1] + random.choice([-1, 1])
            test_obs, _ = test_env.reset(options={"user": user}); test_state = test_agent.prep(test_obs)
            
            test_done, test_ep_reward = False, 0.0
            
            hybrid_test_obs, _ = hybrid_test_env.reset(options={"day": test_env.day, "user": test_env.profile})# reset  env with same day and user profile
            hybrid_test_state = hybrid_test_agent.prep(hybrid_test_obs)
            hybrid_test_done, hybrid_test_ep_reward = False, 0.0


            #test_graph = build_charge_trek_multigraph(test_agent.env.price_df, day=test_agent.env.day, arrival_time= test_agent.env.arrival_time)
            with ThreadPoolExecutor() as executor:
                f1 = executor.submit(test_env.helper_replay_benchmark, test_env.graph)
                f2 = executor.submit(test_env.helper_replay_benchmark, test_env.graph, strategy='stepwise')
                f3 = executor.submit(test_env.dump_charger)

                _, _, test_optimal_costs_episode = f1.result()
                _, _, test_stepwise_costs_episode = f2.result()
                _, _, test_dump_charger_cost = f3.result()

        
            check = True
            found_list_of_actions = False
            hybrid_action_idx = 0
            hybrid_test_actions = []

            while not test_done and not hybrid_test_done:
                # -- Normal test agent --
                test_act = test_agent.act(test_state, validation=True)
                test_nxt, test_r, test_done, _, test_data = test_env.step(test_act)
                test_has_arrived = test_data['reached_goal']
                test_obs, test_state = (test_nxt, test_agent.prep(test_nxt)) if not test_done else (None, None)
                test_ep_reward += test_r

                # -- Hybrid fallback agent --
                if check:
                    # Try normal agent policy
                    hybrid_test_act = hybrid_test_agent.act(hybrid_test_state, validation=True)
                    feas = hybrid_test_env.smart_feasibility_check(hybrid_test_act)
                    if feas == 0:
                        check = False  # Enter fallback mode

                if not check:
                    if not found_list_of_actions:
                        # Compute fallback plan once
                        _, hybrid_test_actions, _ = hybrid_test_env.helper_replay_benchmark(
                            strategy='stepwise',
                            start_node=hybrid_test_env.agent_pos,
                            graph=hybrid_test_env.graph
                        )
                        found_list_of_actions = True
                        hybrid_action_idx = 0

                    # Use fallback plan
                    if hybrid_action_idx < len(hybrid_test_actions):
                        hybrid_test_act = hybrid_test_actions[hybrid_action_idx]
                        hybrid_action_idx += 1
                    else:
                        # Backup safety: if we run out of fallback actions
                        if hybrid_test_env.current_soc < hybrid_test_env.goal_soc:
                            hybrid_test_act = 0
                        elif hybrid_test_env.current_soc > hybrid_test_env.goal_soc:
                            hybrid_test_act = 1
                        else:
                            hybrid_test_act = 2
                
                # Step hybrid agent
                hybrid_test_nxt, hybrid_test_r, hybrid_test_done, _, hybrid_test_data = hybrid_test_env.step(hybrid_test_act)
                hybrid_test_has_arrived = hybrid_test_data['reached_goal']
                hybrid_test_obs, hybrid_test_state = (hybrid_test_nxt, hybrid_test_agent.prep(hybrid_test_nxt)) if not hybrid_test_done else (None, None)
                hybrid_test_ep_reward += hybrid_test_r

            if not hybrid_test_has_arrived:
                print("⚠️ Hybrid agent failed to reach goal:")
                print("  - Day:", hybrid_test_env.day)
                print("  - Profile:", hybrid_test_env.profile)
                print("  - Final SoC:", hybrid_test_env.current_soc)
                print("  - Goal SoC:", hybrid_test_env.goal_soc)
                print("  - Time:", hybrid_test_env.current_time)




            test_rewards_block.append(test_ep_reward)
            test_arrival_rate.append(1 if test_has_arrived else 0)
            test_agent_costs_block.append(test_env.money)
            test_agent_soc_block.append(test_env.current_soc) 
            test_req_soc_block.append(test_env.goal_soc)
            test_soc_gap_block.append(test_env.current_soc - test_env.goal_soc)
            test_optimal_costs_block.append(test_optimal_costs_episode)
            test_stepwise_costs_block.append(test_stepwise_costs_episode)
            test_dump_charger_cost_block.append(test_dump_charger_cost)

            hybrid_test_rewards_block.append(hybrid_test_ep_reward)
            hybrid_test_arrival_rate.append(1 if hybrid_test_has_arrived else 0) 
            hybrid_test_agent_costs_block.append(hybrid_test_agent.env.money)
            hybrid_test_agent_soc_block.append(hybrid_test_agent.env.current_soc)
            hybrid_test_soc_gap_block.append(hybrid_test_agent.env.current_soc - hybrid_test_agent.env.goal_soc)


            

        # =========== 2. LEARNING BURST =========================
        losses_burst = []
        for _ in tqdm(range(min(block+1,OPT_STEPS_PER_BURST)), desc="Expert Training burst"): #OPT_STEPS_PER_BURST is the max 200
            loss = agent.optimise()
            updates_since_sync += 1
            if loss is not None:
                losses_burst.append(loss)

        # Sync target network
        if  updates_since_sync >= MAX_UPDATES_SYNC_INTERVAL :
            agent.tgt.load_state_dict(agent.net.state_dict())
            updates_since_sync = 0
        validation_agent.net.load_state_dict(agent.net.state_dict())
        test_agent.net.load_state_dict(agent.net.state_dict())
        hybrid_test_agent.net.load_state_dict(agent.net.state_dict())
        hybrid_agent.net.load_state_dict(agent.net.state_dict())


        # =========== 3. LOGGING ================================
        avg_reward = float(np.mean(rewards_block))
        avg_arrival_rate = float(np.mean(arrival_rate))
        avg_loss   = float(np.mean(losses_burst)) if losses_burst else 0.0
        avg_optimal_cost = float(np.mean(optimal_costs_block))
        avg_stepwise_cost = float(np.mean(stepwise_costs_block))
        avg_agent_cost = float(np.mean(agent_costs_block))
        avg_agent_soc = float(np.mean(agent_soc_block))
        avg_req_soc = float(np.mean(req_soc_block))
        avg_soc_gap = float(np.mean(soc_gap_block))  # This is the average difference between current SoC and goal SoC
        std_soc_gap = float(np.std(soc_gap_block))
        avg_dump_charger_cost = float(np.mean(dump_charger_cost_block))


        avg_validation_reward = float(np.mean(validation_rewards_block))
        avg_validation_arrival_rate = float(np.mean(validation_arrival_rate))
        avg_validation_agent_cost = float(np.mean(validation_agent_costs_block))
        avg_validation_agent_soc = float(np.mean(validation_agent_soc_block))
        avg_validation_soc_gap = float(np.mean(validation_soc_gap_block))
        std_validation_soc_gap = float(np.std(validation_soc_gap_block))


        avg_test_reward = float(np.mean(test_rewards_block))
        avg_test_arrival_rate = float(np.mean(test_arrival_rate))
        avg_test_agent_cost = float(np.mean(test_agent_costs_block))
        avg_test_agent_soc = float(np.mean(test_agent_soc_block))
        avg_test_req_soc = float(np.mean(test_req_soc_block))
        avg_test_soc_gap = float(np.mean(test_soc_gap_block))
        avg_test_optimal_cost = float(np.mean(test_optimal_costs_block))
        avg_test_stepwise_cost = float(np.mean(test_stepwise_costs_block))
        avg_test_dump_charger_cost = float(np.mean(test_dump_charger_cost_block))
        std_test_soc_gap = float(np.std(test_soc_gap_block))

        avg_hybrid_test_reward = float(np.mean(hybrid_test_rewards_block))
        avg_hybrid_test_arrival_rate = float(np.mean(hybrid_test_arrival_rate))
        avg_hybrid_test_agent_cost = float(np.mean(hybrid_test_agent_costs_block))
        avg_hybrid_test_agent_soc = float(np.mean(hybrid_test_agent_soc_block))
        avg_hybrid_test_soc_gap = float(np.mean(hybrid_test_soc_gap_block))

        avg_hybrid_reward = float(np.mean(hybrid_rewards_block))
        avg_hybrid_arrival_rate = float(np.mean(hybrid_arrival_rate))
        avg_hybrid_agent_cost = float(np.mean(hybrid_agent_costs_block))
        avg_hybrid_agent_soc = float(np.mean(hybrid_agent_soc_block))
        avg_hybrid_soc_gap = float(np.mean(hybrid_soc_gap_block))


         
    
        raw_row = [
                    json.dumps(safe_list(episode_list)),
                    json.dumps(safe_list(steps_list)),

                    json.dumps(safe_list(rewards_block)),
                    json.dumps(safe_list(losses_burst)),
                    json.dumps(safe_list(arrival_rate)),
                    json.dumps(safe_list(optimal_costs_block)),
                    json.dumps(safe_list(stepwise_costs_block)),
                    json.dumps(safe_list(agent_costs_block)),
                    json.dumps(safe_list(agent_soc_block)),
                    json.dumps(safe_list(req_soc_block)),
                    json.dumps(safe_list(soc_gap_block)),
                    json.dumps(safe_list(dump_charger_cost_block)),


                    json.dumps(safe_list(validation_rewards_block)),
                    json.dumps(safe_list(validation_arrival_rate)),
                    json.dumps(safe_list(validation_agent_costs_block)),
                    json.dumps(safe_list(validation_agent_soc_block)),
                    json.dumps(safe_list(validation_soc_gap_block)),


                    json.dumps(safe_list(test_rewards_block)),
                    json.dumps(safe_list(test_arrival_rate)),
                    json.dumps(safe_list(test_optimal_costs_block)),
                    json.dumps(safe_list(test_stepwise_costs_block)),
                    json.dumps(safe_list(test_agent_costs_block)),
                    json.dumps(safe_list(test_agent_soc_block)),
                    json.dumps(safe_list(test_req_soc_block)),
                    json.dumps(safe_list(test_soc_gap_block)),
                    json.dumps(safe_list(test_dump_charger_cost_block)),

                    json.dumps(safe_list(hybrid_rewards_block)),
                    json.dumps(safe_list(hybrid_arrival_rate)),
                    json.dumps(safe_list(hybrid_agent_costs_block)),
                    json.dumps(safe_list(hybrid_agent_soc_block)),
                    json.dumps(safe_list(hybrid_soc_gap_block)),

                    json.dumps(safe_list(hybrid_test_rewards_block)),
                    json.dumps(safe_list(hybrid_test_arrival_rate)),
                    json.dumps(safe_list(hybrid_test_agent_costs_block)),
                    json.dumps(safe_list(hybrid_test_agent_soc_block)),
                    json.dumps(safe_list(hybrid_test_soc_gap_block)),]

        row = [block, global_step, avg_reward,
                         avg_loss, avg_arrival_rate, agent.ε(), avg_soc_gap, avg_optimal_cost,std_soc_gap,
                         avg_stepwise_cost, avg_agent_cost, avg_agent_soc, avg_req_soc, avg_dump_charger_cost, 
                         avg_validation_reward, avg_validation_arrival_rate, avg_validation_agent_cost, avg_validation_agent_soc, avg_validation_soc_gap,std_validation_soc_gap,
                         avg_test_reward, avg_test_arrival_rate, avg_test_agent_cost, avg_test_agent_soc, avg_test_req_soc, avg_test_soc_gap,avg_test_optimal_cost,std_test_soc_gap,
                         avg_test_stepwise_cost, avg_test_dump_charger_cost,
                         avg_hybrid_test_reward, avg_hybrid_test_arrival_rate, avg_hybrid_test_agent_cost, avg_hybrid_test_agent_soc, avg_hybrid_test_soc_gap,
                         avg_hybrid_reward, avg_hybrid_arrival_rate, avg_hybrid_agent_cost, avg_hybrid_agent_soc, avg_hybrid_soc_gap]
    
        row = [round(x, 2) if isinstance(x, float) else x for x in row]
        full_raw = row + raw_row  # combine for full logging
        logger.writerow(full_raw); flog.flush()

        if block % PRINT_EVERY_BLOCK == 0:
            print(f"Blk {block:4d} | steps {global_step:7d} "
                  f"| R_avg {avg_reward:7.2f} | loss {avg_loss:6.3f} | avg arrival rate {avg_arrival_rate:.3f} | ε {agent.ε():.3f} | avg_soc_gap {avg_soc_gap:.2f} | avg_dump_charger_cost {avg_dump_charger_cost:.2f}"
                  f"| avg_optimal_cost {avg_optimal_cost:.2f} | avg_stepwise_cost {avg_stepwise_cost:.2f} | avg_agent_cost {avg_agent_cost:.2f} | avg_agent_soc {avg_agent_soc:.2f} | avg_req_soc {avg_req_soc:.2f}")

        # =========== 4. CHECKPOINT =============================
        if block % CHECK_BLOCK_INTERVAL == 0:
            ckpt = {
                "episode"   : block,
                "steps"     : global_step,
                "policy"    : agent.net.state_dict(),
                "target"    : agent.tgt.state_dict(),
                "optimizer" : agent.opt.state_dict(),
                #"buffer"    : list(agent.mem.mem)
            }
            if ckpt_path.exists():
                shutil.copy2(ckpt_path, bkp_path)
            atomic_save(ckpt, ckpt_path)
            dest_path = CKPT_DIR / f"c51_block_{block}_steps_{global_step}.pt"
            atomic_save(ckpt, dest_path)
            print(f"🗄️  Checkpoint saved at {ckpt_path} (block {block})")

      
        if block % BUFFER_SYNC_INTERVAL == 0:
                    # ─────────────── sync ───────────────
            print(f"📤 Syncing buffer from SSD ({FAST_BUFFER_PATH}) back to project ({PERM_BUFFER_PATH})...")
            PERM_BUFFER_PATH.mkdir(parents=True, exist_ok=True)
            #shutil.copytree(FAST_BUFFER_PATH, PERM_BUFFER_PATH, dirs_exist_ok=True)
            safe_copy_lmdb(FAST_BUFFER_PATH, PERM_BUFFER_PATH)
            print("✅ Buffer sync complete.")

          # =========== 5. STOP CONDITION =========================
        if global_step >= TOTAL_STEPS:
            break


print("✅ Finished at", global_step, "total environment steps")
