"""dagger_train_dqn.py
────────────────────────────────────────
* Collect expert‑labelled episodes in blocks (DAgger)
* After each block perform several supervised updates
* Uses on‑disk LMDB buffer so memory stays flat
"""

# ───────────────────────── Imports ──────────────────────────────
import os, csv, shutil, tempfile, torch, random
from pathlib import Path
import numpy as np
from tqdm import tqdm as base_tqdm
from utils.price_loader import load_caiso_prices, create_rgba_grid
from envs.chargetrek_env import ChargeTrekEnv
from agents.dagger_dqn_agent import DQNAgent, DEVICE
from utils.charge_trek_multigraph import build_charge_trek_multigraph
from concurrent.futures import ThreadPoolExecutor
import  functools
from utils.disk_replay_buffer import ExpertTransition,safe_copy_lmdb
import json
tqdm= functools.partial(base_tqdm, disable=False)

# Paths
FAST_BUFFER_PATH  = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "dagger_buffer"
PERM_BUFFER_PATH  = Path("buffers/dagger_buffer")

TEST_FAST_BUFFER_PATH  = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "test_dagger_buffer"
TEST_PERM_BUFFER_PATH  = Path("buffers/test_dagger_buffer")

# Restore previous buffer from project dir if it exists
if not FAST_BUFFER_PATH.exists():
    FAST_BUFFER_PATH.mkdir(parents=True, exist_ok=True)
    if PERM_BUFFER_PATH.exists():
        print(f"📥 Copying buffer from {PERM_BUFFER_PATH} to SSD {FAST_BUFFER_PATH}...")
        shutil.copytree(PERM_BUFFER_PATH, FAST_BUFFER_PATH, dirs_exist_ok=True)

if not TEST_FAST_BUFFER_PATH.exists():
    TEST_FAST_BUFFER_PATH.mkdir(parents=True, exist_ok=True)
    if TEST_PERM_BUFFER_PATH.exists():
        print(f"📥 Copying buffer from {TEST_PERM_BUFFER_PATH} to SSD {TEST_FAST_BUFFER_PATH}...")
        shutil.copytree(TEST_PERM_BUFFER_PATH, TEST_FAST_BUFFER_PATH, dirs_exist_ok=True)

# ───────────────────────── Hyper‑params ─────────────────────────

BUFFER_SYNC_INTERVAL  = 5
COLLECT_EPISODES      = 25   # episodes per block
TEST_COLLECT_EPISODES = 25
SUPERVISED_STEPS      = 100    # dagger_update() calls per block
TOTAL_STEPS           = 2_000_000
PRINT_EVERY_BLOCK     = 1
CHECK_BLOCK_INTERVAL  = 1

# ───────── Logging / checkpoint folders ─────────────────────────
LOG_DIR, CKPT_DIR = Path("logs"), Path("checkpoints/dagger_dqn")
LOG_DIR.mkdir(parents=True,exist_ok=True); CKPT_DIR.mkdir(parents=True,exist_ok=True)
ckpt_path = CKPT_DIR / "dagger_dqn_latest.pt"
bkp_path  = CKPT_DIR / "dagger_dqn_prev.pt"

# ----------------------- Helper: atomic save --------------------

def atomic_save(obj, path: Path):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        torch.save(obj, tmp.name); tmp.flush(); os.fsync(tmp.fileno())
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
env      = ChargeTrekEnv(grid, price_df,dayslist=train)
agent    = DQNAgent(env, buffer_path=str(FAST_BUFFER_PATH))  # Disk buffer, mode="dagger" inside agent
test_env = ChargeTrekEnv(grid, price_df,dayslist=test)
hybrid_test_env = ChargeTrekEnv(grid, price_df, dayslist=test)
hybrid_env = ChargeTrekEnv(grid, price_df,dayslist=train)
test_agent = DQNAgent(test_env, buffer_path = str(TEST_FAST_BUFFER_PATH))
hybrid_test_agent = DQNAgent(hybrid_test_env, buffer_path = None)
hybrid_agent = DQNAgent(hybrid_env, buffer_path = None)



# ───────── Resume from checkpoint if available ─────────────────
start_block = 0
if ckpt_path.exists():
    print("🔄 Resuming from", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    agent.net.load_state_dict(ckpt["policy"])
    agent.opt.load_state_dict(ckpt["optimizer"])
    agent.steps = ckpt["steps"]
    start_block = ckpt["episode"] + 1
else:
    print("🚀 Starting fresh DAgger training")


test_agent.net.load_state_dict(agent.net.state_dict())
hybrid_test_agent.net.load_state_dict(agent.net.state_dict())
hybrid_agent.net.load_state_dict(agent.net.state_dict())

# ───────── CSV logger (append mode) ─────────────────────────────
log_path = LOG_DIR / "dagger_train_dqn_log.csv"
new_file = not log_path.exists()
with log_path.open("a", newline="") as flog:
    logger = csv.writer(flog)
    if new_file:
        logger.writerow(["block", "steps", "sup_loss", "arrival_rate","avg_gap", "avg_optimal_cost","std_soc_gap",
                          "avg_stepwise_cost", "avg_agent_cost", "avg_agent_soc", "avg_req_soc", "avg_dump_charger_cost",
                          "test_sup_loss", "test_arrival_rate","test_avg_gap", "test_avg_optimal_cost","std_test_soc_gap",
                          "test_avg_stepwise_cost", "test_avg_agent_cost", "test_avg_agent_soc", "test_avg_req_soc", "test_avg_dump_charger_cost",
                          "hybrid_test_avg_arrival_rate","hybrid_test_avg_agent_cost","hybrid_test_avg_agent_soc","hybrid_test_avg_soc_gap",
                          "hybrid_avg_arrival_rate","hybrid_avg_agent_cost","hybrid_avg_agent_soc","hybrid_avg_soc_gap",

                          "raw_episodes","raw_steps",
                          "raw_sup_losses","raw_arrival_rate","raw_optimal_costs_block","raw_stepwise_costs_block","raw_agent_costs_block",
                          "raw_agent_soc_block","raw_req_soc_block","raw_soc_gap_block","raw_dump_charger_cost_block",

                          "raw_test_sup_losses","raw_test_arrival_rate","raw_test_optimal_costs_block","raw_test_stepwise_costs_block",
                          "raw_test_agent_costs_block","raw_test_agent_soc_block","raw_test_req_soc_block","raw_test_soc_gap_block",
                          "raw_test_dump_charger_cost_block",

                          "raw_hybrid_arrival_rate","raw_hybrid_agent_costs_block","raw_hybrid_agent_soc_block","raw_hybrid_soc_gap_block",

                          "raw_hybrid_test_arrival_rate","raw_hybrid_test_agent_costs_block",
                          "raw_hybrid_test_agent_soc_block","raw_hybrid_test_soc_gap_block"
                                            
                                            ])

    # ───── Main DAgger loop (blocks) ──────────────────────────
    global_step = agent.steps
    for block in range(start_block, 1_000_000):

        sup_losses = []
        optimal_costs_block = []
        stepwise_costs_block = []
        agent_costs_block = []
        agent_soc_block = []
        req_soc_block = []
        arrival_rate = []
        soc_gap_block = []
        dump_charger_cost_block = []

        episode_buffer = []
        test_episode_buffer = []


        hybrid_arrival_rate = []
        hybrid_agent_costs_block = []
        hybrid_agent_soc_block = []
        hybrid_soc_gap_block = []
        episode_list = []
        steps_list = []
        episode = 0

        # =========== 1. Data collection for training ======================
        USER_PROFILES = env.users_pool
        for _ in tqdm(range(COLLECT_EPISODES), desc=f"DAGGER Block {block} collect"):
            user = USER_PROFILES[_ % len(USER_PROFILES)].copy()
            user[0], user[-1] = user[0] + random.choice([-0.25, 0.25]), user[-1] + random.choice([-1, 1])
            obs, _ = env.reset(options={"user": user}); state = agent.prep(obs)
            done = False; has_arrived = False

            hybrid_obs, _ = hybrid_env.reset(options={"day": env.day, "user": env.profile})# reset  env with same day and user profile
            hybrid_state = hybrid_agent.prep(hybrid_obs)
            hybrid_done, hybrid_ep_reward = False, 0.0

            episode += 1
            with ThreadPoolExecutor() as executor:
                f1 = executor.submit(env.helper_replay_benchmark, env.graph)
                f2 = executor.submit(env.helper_replay_benchmark, env.graph, strategy='stepwise')
                f3 = executor.submit(env.dump_charger)

                _, _, optimal_costs_episode = f1.result()
                _, _, stepwise_costs_episode = f2.result()
                _, _, dump_charger_cost = f3.result()
            
            
            

            check = True
            found_list_of_actions = False
            hybrid_action_idx = 0
            hybrid_actions = []
            use_optimal = True
            while not done and global_step < TOTAL_STEPS:
                # expert action via Bellman‑Ford or fallback heuristic
                if use_optimal:
                    
                    _, optimal_actions, _ = env.helper_replay_benchmark(env.graph)
                    if optimal_actions:
                        #print("graph nodes:", graph.nodes)
                        #print("pos", env.agent_pos)
                        expert_action = optimal_actions[0]
                    else:
                        use_optimal = False
                        continue  # recompute using fallback next loop
                if not use_optimal:
                    if env.current_soc < env.goal_soc:
                        expert_action = 0
                    elif env.current_soc > env.goal_soc:
                        expert_action = 1
                    else:
                        expert_action = 2

                # store for imitation BEFORE stepping
                #agent.store_dagger(obs, expert_action)#this should store obs(narray) instead of state(tensor) to save space
                episode_buffer.append(ExpertTransition(obs, expert_action))
                # agent acts with its current policy
                agent_action = agent.act(state)
                nxt, _, done, _, data = env.simple_step(agent_action)
                has_arrived = data.get("reached_goal", False)
                global_step += 1
                obs, state = (nxt, agent.prep(nxt)) if not done else (None, None)

                # -- Hybrid fallback agent --
                if check:
                    # Try normal agent policy
                    hybrid_act = hybrid_agent.act(hybrid_state)
                    #feas = hybrid_test_env.dummy_step_feasibility_check(hybrid_test_act, graph=hybrid_test_env.graph)
                    
                    feas = hybrid_env.smart_feasibility_check(hybrid_act)
                    #print(f"Hybrid feasibility check: {feas}")
                    #print(f"Hybrid action: {hybrid_test_act}", type(hybrid_test_act))
                    #print(f"Hybrid agent soc: {hybrid_test_env.current_soc}, goal soc: {hybrid_test_env.goal_soc}, remaining steps: {hybrid_test_env.departure_time - hybrid_test_env.current_time}")

                   

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

                hybrid_nxt, hybrid_r, hybrid_done, _, hybrid_data = hybrid_env.simple_step(hybrid_act)
                hybrid_has_arrived = hybrid_data['reached_goal']
                hybrid_obs, hybrid_state = (hybrid_nxt, hybrid_agent.prep(hybrid_nxt)) if not hybrid_done else (None, None)
                
                

            arrival_rate.append(1 if has_arrived else 0)
            agent_costs_block.append(env.money)
            agent_soc_block.append(env.current_soc)
            req_soc_block.append(env.goal_soc)
            soc_gap_block.append(env.current_soc - env.goal_soc)
            optimal_costs_block.append(optimal_costs_episode)
            stepwise_costs_block.append(stepwise_costs_episode)
            dump_charger_cost_block.append(dump_charger_cost)


            hybrid_arrival_rate.append(1 if hybrid_has_arrived else 0)
            hybrid_agent_costs_block.append(hybrid_env.money)
            hybrid_agent_soc_block.append(hybrid_env.current_soc)
            hybrid_soc_gap_block.append(hybrid_env.current_soc - hybrid_env.goal_soc)

            episode_list.append(episode)
            steps_list.append(global_step)
#====================1.5 Test DATA Collection===================================================


        test_sup_losses = []
        test_optimal_costs_block = []
        test_stepwise_costs_block = []
        test_agent_costs_block = []
        test_agent_soc_block = []
        test_req_soc_block = []
        test_arrival_rate = []
        test_soc_gap_block = []
        test_dump_charger_cost_block = []


        hybrid_test_arrival_rate = []
        hybrid_test_agent_costs_block = []
        hybrid_test_agent_soc_block = []
        hybrid_test_soc_gap_block = []
    
        USER_PROFILES = env.users_pool
        for _ in tqdm(range(TEST_COLLECT_EPISODES), desc=f"DAGGER Block {block} test collect"):
            user = USER_PROFILES[_ % len(USER_PROFILES)].copy()
            user[0], user[-1] = user[0] + random.choice([-0.25, 0.25]), user[-1] + random.choice([-1, 1])
            test_obs, _ = test_env.reset(options={"user": user}); test_state = test_agent.prep(test_obs)
            #print("Agent pos after reset:", env.agent_pos)

            test_done = False; test_has_arrived = False

            hybrid_test_obs, _ = hybrid_test_env.reset(options={"day": test_env.day, "user": test_env.profile})# reset  env with same day and user profile
            hybrid_test_state = hybrid_test_agent.prep(hybrid_test_obs)
            hybrid_test_done, hybrid_test_ep_reward = False, 0.0

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
            test_use_optimal = True
            while not test_done and global_step < TOTAL_STEPS:
                # expert action via Bellman‑Ford or fallback heuristic
                if test_use_optimal:
                    
                    _, test_optimal_actions, _ = test_env.helper_replay_benchmark(test_env.graph)
                    if test_optimal_actions:
                        #print("graph nodes:", graph.nodes)
                        #print("pos", env.agent_pos)
                        expert_action = test_optimal_actions[0]
                    else:
                        test_use_optimal = False
                        continue  # recompute using fallback next loop
                if not test_use_optimal:
                    if test_env.current_soc < test_env.goal_soc:
                        expert_action = 0
                    elif test_env.current_soc > test_env.goal_soc:
                        expert_action = 1
                    else:
                        expert_action = 2

                # store for imitation BEFORE stepping
                #test_agent.store_dagger(test_obs, expert_action)#this should store obs(narray) instead of state(tensor) to save space
                test_episode_buffer.append(ExpertTransition(test_obs, expert_action))
                # agent acts with its current policy
                test_agent_action = test_agent.act(test_state)
                test_nxt, _, test_done, _, test_data = test_env.simple_step(test_agent_action)
                test_has_arrived = test_data.get("reached_goal", False)
                test_state = test_agent.prep(test_nxt) if not test_done else None

                

                # -- Hybrid fallback agent --
                if check:
                    # Try normal agent policy
                    hybrid_test_act = hybrid_test_agent.act(hybrid_test_state)
                    #feas = hybrid_test_env.dummy_step_feasibility_check(hybrid_test_act, graph=hybrid_test_env.graph)
                    
                    feas = hybrid_test_env.smart_feasibility_check(hybrid_test_act)
                    #print(f"Hybrid feasibility check: {feas}")
                    #print(f"Hybrid action: {hybrid_test_act}", type(hybrid_test_act))
                    #print(f"Hybrid agent soc: {hybrid_test_env.current_soc}, goal soc: {hybrid_test_env.goal_soc}, remaining steps: {hybrid_test_env.departure_time - hybrid_test_env.current_time}")

                   

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

                hybrid_test_nxt, hybrid_test_r, hybrid_test_done, _, hybrid_test_data = hybrid_test_env.simple_step(hybrid_test_act)
                hybrid_test_has_arrived = hybrid_test_data['reached_goal']
                hybrid_test_obs, hybrid_test_state = (hybrid_test_nxt, hybrid_test_agent.prep(hybrid_test_nxt)) if not hybrid_test_done else (None, None)
                

            test_arrival_rate.append(1 if test_has_arrived else 0)
            test_agent_costs_block.append(test_env.money)
            test_agent_soc_block.append(test_env.current_soc)
            test_req_soc_block.append(test_env.goal_soc)
            test_soc_gap_block.append(test_env.current_soc - test_env.goal_soc)
            test_optimal_costs_block.append(test_optimal_costs_episode)
            test_stepwise_costs_block.append(test_stepwise_costs_episode)
            test_dump_charger_cost_block.append(test_dump_charger_cost)


            hybrid_test_arrival_rate.append(1 if hybrid_test_has_arrived else 0)
            hybrid_test_agent_costs_block.append(hybrid_test_env.money)
            hybrid_test_agent_soc_block.append(hybrid_test_env.current_soc)
            hybrid_test_soc_gap_block.append(hybrid_test_env.current_soc - hybrid_test_env.goal_soc)

        #test_agent.mem.clear()
        test_agent.mem.push_many(test_episode_buffer)
        agent.mem.push_many(episode_buffer)
        test_agent.net.load_state_dict(agent.net.state_dict())
        hybrid_test_agent.net.load_state_dict(agent.net.state_dict())
        hybrid_agent.net.load_state_dict(agent.net.state_dict())
        hybrid_agent.net.eval()
        hybrid_test_agent.net.eval()
        test_agent.net.eval()

        # =========== 2. Supervised DAgger updates ============
        for _ in tqdm((range(min(block+1,SUPERVISED_STEPS))), desc="DAGGER Supervised updates"):
            loss = agent.dagger_update()
            test_loss = test_agent.dagger_eval_loss()
            test_agent.net.load_state_dict(agent.net.state_dict())
            if loss is not None: sup_losses.append(loss)
            if test_loss is not None: test_sup_losses.append(test_loss)
        

        # =========== 3. Logging ==============================
        avg_loss   = float(np.mean(sup_losses)) if sup_losses else 0.0
        avg_arrival_rate = float(np.mean(arrival_rate))
        avg_optimal_cost = float(np.mean(optimal_costs_block))
        avg_stepwise_cost = float(np.mean(stepwise_costs_block))
        avg_agent_cost = float(np.mean(agent_costs_block))
        avg_agent_soc = float(np.mean(agent_soc_block))
        avg_req_soc = float(np.mean(req_soc_block))
        avg_soc_gap = float(np.mean(soc_gap_block))  # This is the average difference between current SoC and goal SoC
        avg_dump_charger_cost = float(np.mean(dump_charger_cost_block))
        std_soc_gap = float(np.std(soc_gap_block))


        test_avg_loss   = float(np.mean(test_sup_losses)) if test_sup_losses else 0.0
        test_avg_arrival_rate = float(np.mean(test_arrival_rate))
        test_avg_optimal_cost = float(np.mean(test_optimal_costs_block))
        test_avg_stepwise_cost = float(np.mean(test_stepwise_costs_block))
        test_avg_agent_cost = float(np.mean(test_agent_costs_block))
        test_avg_agent_soc = float(np.mean(test_agent_soc_block))
        test_avg_req_soc = float(np.mean(test_req_soc_block))
        test_avg_soc_gap = float(np.mean(test_soc_gap_block))  # This is the average difference between current SoC and goal SoC
        test_avg_dump_charger_cost = float(np.mean(test_dump_charger_cost_block))
        test_std_soc_gap = float(np.std(test_soc_gap_block))


        hybrid_test_avg_arrival_rate = float(np.mean(hybrid_test_arrival_rate))
        hybrid_test_avg_agent_cost = float(np.mean(hybrid_test_agent_costs_block))
        hybrid_test_avg_agent_soc = float(np.mean(hybrid_test_agent_soc_block))
        hybrid_test_avg_soc_gap = float(np.mean(hybrid_test_soc_gap_block))


        hybrid_avg_arrival_rate = float(np.mean(hybrid_arrival_rate))
        hybrid_avg_agent_cost = float(np.mean(hybrid_agent_costs_block))
        hybrid_avg_agent_soc = float(np.mean(hybrid_agent_soc_block))
        hybrid_avg_soc_gap = float(np.mean(hybrid_soc_gap_block))



        raw_row = [
        json.dumps(safe_list(episode_list)),
        json.dumps(safe_list(steps_list)),

        json.dumps(safe_list(sup_losses)),
        json.dumps(safe_list(arrival_rate)),
        json.dumps(safe_list(optimal_costs_block)),
        json.dumps(safe_list(stepwise_costs_block)),
        json.dumps(safe_list(agent_costs_block)),
        json.dumps(safe_list(agent_soc_block)),
        json.dumps(safe_list(req_soc_block)),
        json.dumps(safe_list(soc_gap_block)),
        json.dumps(safe_list(dump_charger_cost_block)),

        json.dumps(safe_list(test_sup_losses)),
        json.dumps(safe_list(test_arrival_rate)),
        json.dumps(safe_list(test_optimal_costs_block)),
        json.dumps(safe_list(test_stepwise_costs_block)),
        json.dumps(safe_list(test_agent_costs_block)),
        json.dumps(safe_list(test_agent_soc_block)),
        json.dumps(safe_list(test_req_soc_block)),
        json.dumps(safe_list(test_soc_gap_block)),
        json.dumps(safe_list(test_dump_charger_cost_block)),

        json.dumps(safe_list(hybrid_arrival_rate)),
        json.dumps(safe_list(hybrid_agent_costs_block)),
        json.dumps(safe_list(hybrid_agent_soc_block)),
        json.dumps(safe_list(hybrid_soc_gap_block)),

        json.dumps(safe_list(hybrid_test_arrival_rate)),
        json.dumps(safe_list(hybrid_test_agent_costs_block)),
        json.dumps(safe_list(hybrid_test_agent_soc_block)),
        json.dumps(safe_list(hybrid_test_soc_gap_block)),]

        row = [block, global_step, avg_loss, avg_arrival_rate, avg_soc_gap, avg_optimal_cost,std_soc_gap,
                         avg_stepwise_cost, avg_agent_cost, avg_agent_soc, avg_req_soc, avg_dump_charger_cost,
                         test_avg_loss, test_avg_arrival_rate, test_avg_soc_gap, test_avg_optimal_cost,test_std_soc_gap,
                         test_avg_stepwise_cost, test_avg_agent_cost, test_avg_agent_soc, test_avg_req_soc, test_avg_dump_charger_cost,
                         hybrid_test_avg_arrival_rate,hybrid_test_avg_agent_cost,hybrid_test_avg_agent_soc,hybrid_test_avg_soc_gap,
                         hybrid_avg_arrival_rate,hybrid_avg_agent_cost,hybrid_avg_agent_soc,hybrid_avg_soc_gap,]
        
        row = [round(x, 2) if isinstance(x, float) else x for x in row]
        full_raw = row + raw_row  # combine for full logging
        logger.writerow(full_raw); flog.flush()

        if block % PRINT_EVERY_BLOCK == 0:
            print(
                f"Blk {block:3d} | steps {global_step:7d} | "
                f"sup_loss {avg_loss:.4f} | arrival {avg_arrival_rate:.3f} | avg_gap {avg_soc_gap:.2f} | avg_dump_charger_cost {avg_dump_charger_cost:.2f}"
                f"| avg_optimal_cost {avg_optimal_cost:.2f} | avg_stepwise_cost {avg_stepwise_cost:.2f} | avg_agent_cost {avg_agent_cost:.2f} | avg_agent_soc {avg_agent_soc:.2f} | avg_req_soc {avg_req_soc:.2f}")
            

        # =========== 4. Checkpoint ===========================
        if block % CHECK_BLOCK_INTERVAL == 0:
            ckpt = {
                "episode" : block,
                "steps"   : global_step,
                "policy"  : agent.net.state_dict(),
                "optimizer": agent.opt.state_dict()
            }
            if ckpt_path.exists(): shutil.copy2(ckpt_path, bkp_path)
            atomic_save(ckpt, ckpt_path)
            dest_path = CKPT_DIR / f"dagger_block_{block}_steps_{global_step}.pt"
            atomic_save(ckpt, dest_path)
            print(f"🗄️  Checkpoint saved at block {block}")

        if block % BUFFER_SYNC_INTERVAL == 0:
                    # ─────────────── sync ───────────────
            print(f"📤 Syncing buffer from SSD ({FAST_BUFFER_PATH}) back to project ({PERM_BUFFER_PATH})...")
            PERM_BUFFER_PATH.mkdir(parents=True, exist_ok=True)
            #shutil.copytree(FAST_BUFFER_PATH, PERM_BUFFER_PATH, dirs_exist_ok=True)
            safe_copy_lmdb(FAST_BUFFER_PATH, PERM_BUFFER_PATH)

            TEST_PERM_BUFFER_PATH.mkdir(parents=True, exist_ok=True)
            #shutil.copytree(TEST_FAST_BUFFER_PATH, TEST_PERM_BUFFER_PATH, dirs_exist_ok=True)
            safe_copy_lmdb(TEST_FAST_BUFFER_PATH, TEST_PERM_BUFFER_PATH)
            print("✅ Buffer sync complete.")


        # =========== 5. Stop condition =======================
        if global_step >= TOTAL_STEPS:
            break

print("✅ Finished at", global_step, "total env steps")
