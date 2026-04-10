import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.soc_mapper import charge_soc, discharge_soc
import time
from utils.price_loader import load_caiso_prices, create_rgba_grid
from benchmarks.magic_solver import find_optimal_path_bellman_ford
from utils.charge_trek_multigraph import build_charge_trek_multigraph
from benchmarks.realistic_solver import run_stepwise_replanning
import random
import torch
import copy

# ---- constants ----------------------------------------------------- #
R_MAX_PCT = 2
MIN_GAP_NORMALIZED = -0.06        # %SoC the EV can shift per 15-min step
FEAS_BONUS = 10        # terminal bonus/penalty

# ------------------------------------------------------------------ #
# Constants for day sampling (put near top of file or in a config)
TRAIN_DAY_START = 1        # inclusive
TRAIN_DAY_END   = 100     # inclusive

USERS_POOL = [
    # 1. Evening commuter: arrives at 16:00 with 25% SoC, needs 90% by 08:00 next day (16 hrs available)
    # Requires +65% → feasible within ~9.4h (using 8%/h then 4%/h charging rates)
    [16, 24, 90, 16 * 4],

    # 2. Dinner errand: arrives at 18:00 with 50% SoC, target 65% by 03:00 (9 hrs window)
    # Needs +15% → feasible within ~1.9h at 8%/h
    [18, 50, 66, 9 * 4],

    # 3. Night V2G trader: arrives at midnight with 90% SoC, discharges to 60% by 06:00
    # Needs −30% over 6 hrs → feasible for discharging use case
    [0, 90, 60, 7 * 4],

    # 4. Early bird: arrives at 03:00 with 50% SoC, target 100% by 14:00 (11 hrs window)
    # Needs +50% → feasible within ~8.8h
    [3, 50, 100, 11 * 4],

    # 5. Weekend day-trip: arrives at 08:00 with 75% SoC, wants full charge by 16:00 (8 hrs)
    # Needs +25% → feasible within ~5.6h
    [8, 76, 100, 9 * 4],

    # 6. Office worker: arrives at 09:00 with 30% SoC, target 80% by 17:00 (8 hrs window)
    # Needs +50% → feasible within ~6.3h
    [9, 30, 80, 10 * 4],

    # 7. Short errand: arrives at 14:30 with 60% SoC, needs 70% by 16:00 (1.5 hrs available)
    # Needs +10% → feasible within ~1.25h
    [14, 60, 70, 4 * 4],

    # 8. Heavy V2G participant: arrives at 20:00 with 95% SoC, discharges to 40% by 07:00 (11 hrs window)
    # Needs −55% discharge → feasible, no charging rate constraint
    [20, 96, 40, 11 * 4],

    # 9. Airport drop-off: arrives at 05:00 with 46% SoC, target 80% by 10:00 (5 hrs window)
    # Needs +34% → feasible within ~4.4h
    [5, 100, 70, 6 * 4],

    # 10. Late-nighter: arrives at 23:00 with 20% SoC, target 90% by 08:00 (9 hrs window)
    # Needs +70% → feasible in ~10h, fits available time
    [23, 90, 30, 10 * 4],

    # 11. Student: arrives at 10:00 with 40% SoC, target 70% by 15:00 (5 hrs)
    # Needs +30% → feasible within ~3.8h
    [10, 70, 40, 5 * 4],

    # 12. Daytime discharge trader: arrives at 08:00 with 85% SoC, discharges to 50% by 14:00 (6 hrs)
    # Needs −35% discharge → feasible
    [8, 86, 50, 6 * 4],

    # 13. Overnight top-up: arrives at 01:00 with 60% SoC, target 90% by 06:00 (5 hrs)
    # Needs +30% → feasible exactly within 5h
    [1, 94, 64, 6 * 4],

    # 14. Midday charger: arrives at 11:00 with 35% SoC, target 85% by 18:00 (7 hrs)
    # Needs +50% → feasible within ~6.3h
    [11, 35, 85, 8 * 4],

    # 15. Two-hour window: arrives at 15:00 with 65% SoC, target 75% by 17:00 (2 hrs window)
    # Needs +10% → feasible within ~1.25h
    [15, 65, 75, 3 * 4],
]






class ChargeTrekEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid, price_df, start_points=(0, 0), goal_soc=80, departure_time=95, day=0, arrival_time = 16, start_day = TRAIN_DAY_START, end_day = TRAIN_DAY_END, dayslist = list(range(TRAIN_DAY_START, TRAIN_DAY_END + 1))):
        super().__init__()
        self.grid = grid
        self.price_df = price_df
        self.day = day
        self.time_steps, self.soc_levels, _ = grid.shape
        self.goal_soc = goal_soc
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.profile = None  # will be set in reset()
        self.start_day = start_day
        self.dayslist = dayslist
        self.end_day = end_day
        self.has_solution = True  # tracks if a feasible solution exists for the current profile
        self.start_points = start_points or [(0, 0), (10, 20), (20, 50), (30, 75)]

        self.current_time = 0
        self.current_soc = 0.0
        self.agent_pos = (0, 0)
        self.users_pool = USERS_POOL
        self.money_bar = np.zeros((101, 4), dtype=np.uint8)  # shape: (101, RGBA)
        self.money_bar[:, 3] = 255  # full opacity

        self.money = 0.0
        self.graph = build_charge_trek_multigraph(price_df, day=day, arrival_time=arrival_time)

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.time_steps, self.soc_levels, 4), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        #print('Env init with day = ', self.day)



    """
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time, self.current_soc = self.start_points #self.np_random.choice(self.start_points)
        self.current_soc = float(self.current_soc)
        self.agent_pos = (self.current_time, int(self.current_soc))
        self.money_bar = np.zeros((101, 4), dtype=np.uint8)  # shape: (101, RGBA)
        self.money_bar[:, 3] = 255  # full opacity

        self.money = 0.0
        #self.day = self.day + 1
        print('Env reset with day = ', self.day)
        return self._get_observation(), {}
        """
    

      

    def reset(self, seed=None, options=None):
        """
        Resets the environment.

        options (dict) may contain:
            - "user" : tuple(arrival_step, start_soc, goal_soc, departure_step(in relative time))
            - "day"  : int  (index into price_df)
        """
        super().reset(seed=seed)

       
        # -------- 1. choose user profile --------------------------------- #
        if options and "user" in options:
            self.profile = options["user"]
        else:
            self.profile = random.choice(self.users_pool)

        arrival_step, start_soc, self.goal_soc, self.departure_time = self.profile
        self.arrival_time = arrival_step

        # -------- 2. choose training day -------------------------------- #
        if options and "day" in options:
            self.day = options["day"]
        else:
            #self.day = random.randint(self.start_day, self.end_day)
            #self.day = 50
            self.day = random.choice(self.dayslist)

        # -------- 3. initialise state exactly like original logic ------- #
        self.grid = create_rgba_grid(self.price_df, steps=96, soc_levels=101, arrival_time = self.arrival_time, day = self.day)
        self.current_time, self.current_soc = 0, float(start_soc) #0 because in our logic we always start at time 0 (relative time)
        self.agent_pos = (self.current_time, int(self.current_soc))
        self.graph = build_charge_trek_multigraph(self.price_df, day=self.day, arrival_time=self.arrival_time)
        self.money_bar = np.zeros((101, 4), dtype=np.uint8)
        self.money_bar[:, 3] = 255
        self.money = 0.0
        self.has_solution = True


        #print(f"Env reset with day={self.day}  profile={profile}")
        return self._get_observation(), {}


    def step(self, action):
        assert self.action_space.contains(action)

        # ---------- cache state BEFORE action --------------------------- #
        prev_time     = self.current_time
        prev_soc      = self.current_soc
        prev_money    = self.money
        trem     = self.departure_time - prev_time
        soc_gap  = self.goal_soc - self.current_soc 
        gap   = (abs(self.goal_soc - self.current_soc) - R_MAX_PCT * trem) / (self.goal_soc - 0) # scale to [0,1] range

        # ---------- apply action (charge / discharge) ------------------- #
        if action == 0 and self.current_soc < 100:
            self.current_soc = charge_soc(self.current_soc, dt=15)
            #print(f"Charging: {self.current_soc:.1f}% SoC")
        elif action == 1 and self.current_soc > 20:
            self.current_soc = discharge_soc(self.current_soc, dt=15)
            #print(f"Discharging: {self.current_soc:.1f}% SoC")
        elif action == 2:
            pass
            #print(f"Idle: {self.current_soc:.1f}% SoC")

        self.current_soc  = np.clip(self.current_soc, 0, 100)
        self.current_time += 1
        self.agent_pos     = (self.current_time, int(self.current_soc))

        # ---------- update money ---------------------------------------- #
        OFFSET = int(self.arrival_time * 60 // 15)
        idx    = prev_time + OFFSET + 96 * self.day
        price  = (self.price_df["real_price"].iloc[idx]
                if self.grid[prev_time, int(prev_soc), 3] != 255
                else self.price_df["forecast_price"].iloc[idx])

        BAT_CAP_KWH = 75               # battery nameplate
        ETA_C, ETA_D = 0.8, 0.8      # charging / discharging efficiencies
        max_predicted_price = max(self.price_df["forecast_price"].iloc[idx - prev_time : idx - prev_time +96]) #used to scall reward
        delta_soc   = (self.current_soc - prev_soc) / 100   # signed fraction
        E_bat       = BAT_CAP_KWH * delta_soc               # +ve charge, –ve discharge 
        price_kWh = price /100                              # $ per kWh we dicvided by 100 and not 1000 to simulate realistic retail prices (the data has only wholsales prices)

        if E_bat > 0:                                       # CHARGE
            E_grid = E_bat / ETA_C                          # what the meter sees
            cash   = + price_kWh * E_grid                   # cost → positive
        else:                                               # DISCHARGE
            E_grid = -E_bat * ETA_D                         # export (positive kWh)
            cash   = - price_kWh * E_grid                   # revenue → negative

        self.money += cash
        self._update_money_bar()


        # ---------- feasibility shaping -------------------------------- #
       
        
        
        if not self.has_solution and soc_gap > 0 and action == 0:
            feas_shaping = + 0.2
        elif not self.has_solution and soc_gap > 0 and action == 1:
            feas_shaping = -gap
        elif not self.has_solution and soc_gap > 0 and action == 2:
            feas_shaping = -gap/2
        elif not self.has_solution and soc_gap < 0 and action == 1:
            feas_shaping = + 0.2
        elif not self.has_solution and soc_gap < 0 and action == 0:
            feas_shaping = -gap
        elif not self.has_solution and soc_gap < 0 and action == 2:
            feas_shaping = -gap/2
        else:
            feas_shaping = 0.0
        feasibility_penalty = 0.0
        feas = self.smart_feasibility_check(None)
        #_,actions,feas = self.helper_replay_benchmark(graph=self.graph, start_node=self.agent_pos, goal_node=(self.departure_time, self.goal_soc), strategy='stepwise')
        if feas == 0 and self.has_solution == True:
            feasibility_penalty = -10
            self.has_solution = False
        
        #print('feas_shaping = ', feas_shaping, ' gap = ', gap, ' soc_gap = ', soc_gap)

        # ---------- compose reward ------------------------------------- #
        

        done = self.current_time >= self.departure_time
        soc_gap = self.goal_soc - self.current_soc
        
        reward_money = -cash / max_predicted_price          # saving / earning → positive
        #reward =  feas_shaping  + reward_money + feasibility_penalty
        reward =  reward_money + feasibility_penalty
        
        if done:
            if soc_gap == 0:
                reward += FEAS_BONUS
            """ 
            elif soc_gap == 1:
                reward -= FEAS_BONUS/2
            elif soc_gap == -1:
                reward += FEAS_BONUS/2
            elif soc_gap == 2:
                reward -= FEAS_BONUS/5
            elif soc_gap == -2:
                reward += FEAS_BONUS/5
            else :
                reward -=FEAS_BONUS
            """
        
        

            
        return self._get_observation(), reward, done, False, {"money": self.money, "reached_goal": True if soc_gap == 0 else False, "feasibility_penalty":feasibility_penalty}


    def render(self, return_image=False):
        vis_grid = self.grid.copy()
        t_current = self.current_time

        for t in range(self.time_steps):
            #print(f" time steps {self.time_steps} soc steps {self.soc_levels}")
            if t > t_current + 1:
                vis_grid[t, :, 2] = 0
                vis_grid[t, :, 3] = 255

        x, y = self.agent_pos
        vis_grid[x, int(y)] = [0, 0, 0, 255]
        vis_grid[self.departure_time, self.goal_soc] = [255, 255, 0, 255]

        image = vis_grid.transpose(1, 0, 2)[::-1]

        spacer = np.zeros((image.shape[0], 1, 4), dtype=np.uint8)
        spacer[:,:, 3] = 255
        bar = self.money_bar.reshape((101, 1, 4))
        
        final_image = np.concatenate([image, spacer, bar], axis=1)

        if return_image:
            return final_image.astype(np.uint8)
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.imshow(final_image, aspect='auto')
        plt.title(f"Time: {t_current}, SoC: {self.current_soc:.1f}, Money: {self.money:.2f}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        """

    def _get_observation(self):
        return self.render(return_image=True)

    def _update_money_bar(self):
        BAR_HEIGHT = 101
        CENTER = BAR_HEIGHT // 2
        MAX_MONEY_ABS = 100  # expected max gain/loss in $

        self.money_bar[:, :3] = 0  # reset RGB channels

        scaled = int(np.clip((self.money / MAX_MONEY_ABS) * (BAR_HEIGHT // 2), -CENTER, CENTER))

        if scaled > 0:
            for i in range(scaled):
                self.money_bar[CENTER - 1 - i, 0] = 255  # red channel
        elif scaled < 0:
            for i in range(-scaled):
                self.money_bar[CENTER + i, 1] = 255  # green channel


    def dummy_reset(self):
        """
        Performs a reset-like action without altering day, money, or internal RNG.
        Only resets SoC, time, and money bar state.

        Returns:
            observation: full grid
            info: empty dict
        """
        self.money = 0.0
        self.money_bar[:, :3] = 0
        return self._get_observation(), {}


    def replay_benchmark(self, key='real', start_node=None, goal_node=None, render=True, delay=0.1, strategy='optimal'):
        """
        Internally computes and replays the optimal path using Bellman-Ford based on stored price_df and utils graph generator.

        Parameters:
            key: 'real' or 'forecast' to choose edge weights
            start_node: (t, soc) tuple. If None, uses current agent position
            goal_node: (t, soc) tuple. If None, defaults to (departure_time, goal_soc)
            render: whether to render the environment at each step
            delay: time delay between renders (in seconds)
            strategy: 'optimal' for full plan, 'stepwise' for real-time replanning

        Returns:
            path: list of (t, soc)
            actions: list of discrete actions taken
            total_cost: cumulative cost along the path
        """
        graph = build_charge_trek_multigraph(self.price_df, day=self.day, arrival_time= self.arrival_time)
        #print('Replay Bench init with day = ', self.day)

        if start_node is None:
            start_node = self.agent_pos
        if goal_node is None:
            goal_node = (self.departure_time, self.goal_soc)

        if strategy == 'stepwise':
            path, cost = run_stepwise_replanning(graph, start_node, goal_node[0], goal_node[1], key_forecast='forecast', key_real='real')
        else:
            path, cost = find_optimal_path_bellman_ford(graph, start_node, goal_node, key=key)

        if not path:
            return [], [], float('inf')

        actions = []
        for i in range(1, len(path)):
            prev_soc = path[i-1][1]
            curr_soc = path[i][1]
            if curr_soc > prev_soc:
                actions.append(0)
            elif curr_soc < prev_soc:
                actions.append(1)
            else:
                actions.append(2)
        """
        # Use dummy reset to maintain current day/state
        self.agent_pos = path[0]
        self.current_time, self.current_soc = path[0]
        self.dummy_reset()

        for action in actions:
            _, _, done, _, _ = self.step(action)
            if render:
                self.render()
                time.sleep(delay)
            if done:
                break
        """

        return path, actions, cost


    def helper_replay_benchmark(self, graph,key='real', start_node=None, goal_node=None, render=True, delay=0.1, strategy='optimal'):
        """
        Internally computes and replays the optimal path using Bellman-Ford based on stored price_df and utils graph generator.

        Parameters:
            key: 'real' or 'forecast' to choose edge weights
            start_node: (t, soc) tuple. If None, uses current agent position
            goal_node: (t, soc) tuple. If None, defaults to (departure_time, goal_soc)
            render: whether to render the environment at each step
            delay: time delay between renders (in seconds)
            strategy: 'optimal' for full plan, 'stepwise' for real-time replanning

        Returns:
            path: list of (t, soc)
            actions: list of discrete actions taken
            total_cost: cumulative cost along the path
        """
        
        #print("Helper Replay Bench init with pos:", self.agent_pos)


        if start_node is None:
            start_node = self.agent_pos
        if goal_node is None:
            goal_node = (self.departure_time, self.goal_soc)

        if strategy == 'stepwise':
            
            path, cost = run_stepwise_replanning(graph, start_node, goal_node[0], goal_node[1], key_forecast='forecast', key_real='real')
            
        else:
            path, cost = find_optimal_path_bellman_ford(graph, start_node, goal_node, key=key)

            if not path:
                #print("No path found", self.profile, "day", self.day, "start_node", start_node, "goal_node", goal_node)
                return [], [], float('inf')

        actions = []
        for i in range(1, len(path)):
            prev_soc = path[i-1][1]
            curr_soc = path[i][1]
            if curr_soc > prev_soc:
                actions.append(0)
            elif curr_soc < prev_soc:
                actions.append(1)
            else:
                actions.append(2)
      
        return path, actions, cost



    def range_helper_replay_benchmark(self, graph,key='real', start_node=None, goal_node=None, render=True, delay=0.1, strategy='optimal'):
        """
        Internally computes and replays the optimal path using Bellman-Ford based on stored price_df and utils graph generator.

        Parameters:
            key: 'real' or 'forecast' to choose edge weights
            start_node: (t, soc) tuple. If None, uses current agent position
            goal_node: (t, soc) tuple. If None, defaults to (departure_time, goal_soc)
            render: whether to render the environment at each step
            delay: time delay between renders (in seconds)
            strategy: 'optimal' for full plan, 'stepwise' for real-time replanning

        Returns:
            path: list of (t, soc)
            actions: list of discrete actions taken
            total_cost: cumulative cost along the path
        """
        
        #print("Helper Replay Bench init with pos:", self.agent_pos)


        if start_node is None:
            start_node = self.agent_pos
        if goal_node is None:
            goal_node = (self.departure_time, self.goal_soc)

        if strategy == 'stepwise':
            
            path, cost = run_stepwise_replanning(graph, start_node, goal_node[0], goal_node[1], key_forecast='forecast', key_real='real')
            
        else:
            path = None
            cost = +np.inf
            exact_goal_path = None
            exact_goal_cost = +np.inf
            for soc in range(self.goal_soc,101):
                goal_node = (self.departure_time, soc)
                _path, _cost = find_optimal_path_bellman_ford(graph, start_node, goal_node, key=key)
                if _cost < cost:
                    path = _path
                    cost = _cost
                if soc == self.goal_soc:
                    exact_goal_path = _path
                    exact_goal_cost = _cost


            if not path:
                #print("No path found", self.profile, "day", self.day, "start_node", start_node, "goal_node", goal_node)
                return [], [], float('inf')

        actions = []
        for i in range(1, len(path)):
            prev_soc = path[i-1][1]
            curr_soc = path[i][1]
            if curr_soc > prev_soc:
                actions.append(0)
            elif curr_soc < prev_soc:
                actions.append(1)
            else:
                actions.append(2)
      
        return  cost, exact_goal_cost

    def full_replay_benchmark(self, key='real', start_node=None, goal_node=None, render=False, delay=0.1, strategy='optimal', agent=None):
        """
        Replays a planned trajectory using one of three strategies:
        - 'optimal': full Bellman-Ford plan
        - 'stepwise': step-by-step replanning
        - 'dqn': follow a provided DQN agent
        - 'dump': use the dump_charger method to reach goal SoC

        Note: the environment should already be reset to the desired start state before calling.

        Parameters:
            key: 'real' or 'forecast' for graph weights (unused for 'dqn')
            start_node: (t, soc) tuple; defaults to current agent_pos
            goal_node: (t, soc) tuple; defaults to (departure_time, goal_soc)
            render: whether to render at each step
            delay: pause between renders
            strategy: 'optimal' | 'stepwise' | 'dqn'
            agent: a C51Agent with policy_net loaded (required for 'dqn')

        Returns:
            path: list of (t, soc)
            actions: list of discrete actions
            total_cost: total money spent (env.money)
        """
        
        # determine start and goal
        if start_node is None:
            start_node = self.agent_pos
        if goal_node is None:
            goal_node = (self.departure_time, self.goal_soc)
        """
        # DQN strategy: assume env already at start_node
        if strategy == 'dqn':
            if agent is None:
                raise ValueError("DQN strategy requires an agent instance")
            device = next(agent.net.parameters()).device  
            path = [start_node]
            actions = []
            done = False
            # replay until done
            while not done:
                obs = self._get_observation()
                state = agent.prep(obs).to(device)
                with torch.no_grad():
                    qvals = (agent.net(state.unsqueeze(0)) * agent.support).sum(2)
                    action = int(qvals.argmax(1).item())
                actions.append(action)
                _, _, done, _, _ = self.step(action)
                path.append(self.agent_pos)
                if render:
                    self.render()
                    time.sleep(delay)
            return path, actions, self.money
            """
      # DQN strategy: assume env already at start_node
        if strategy == 'dqn':
            if agent is None:
                raise ValueError("DQN strategy requires an agent instance")

            path = [start_node]
            actions = []
            done = False

            while not done:
                device = next(agent.net.parameters()).device  
                obs = self._get_observation()
                state = agent.prep(obs).to(device)  # move to correct device

                act = agent.act(state, validation=True)       # use greedy policy
                actions.append(act)

                nxt, r, done, _, _ = self.simple_step(act)
                path.append(self.agent_pos)

                if render:
                    self.render()
                    time.sleep(delay)

            return path, actions, self.money
        
        if strategy == 'dump':
            

           
            return self.dump_charger()



        # Graph-based: build multigraph once
        graph = build_charge_trek_multigraph(self.price_df, day=self.day, arrival_time=getattr(self, 'arrival_time', 0))
        # compute path
        if strategy == 'stepwise':
            path, cost = run_stepwise_replanning(graph, start_node, goal_node[0], goal_node[1], key_forecast='forecast', key_real='real')
        else:
            path, cost = find_optimal_path_bellman_ford(graph, start_node, goal_node, key=key)

        if not path:
            return [], [], float('inf')

        # translate to actions
        actions = []
        for i in range(1, len(path)):
            prev_soc, curr_soc = path[i-1][1], path[i][1]
            if curr_soc > prev_soc:
                actions.append(0)
            elif curr_soc < prev_soc:
                actions.append(1)
            else:
                actions.append(2)

        return path, actions, cost
    
    def dump_charger(self):
    
        # Clone the environment
        env_clone = copy.deepcopy(self)
        actions = []
        path  = [env_clone.agent_pos]
        

        # Run until we reach the goal SoC
        while True:
            if env_clone.current_soc < env_clone.goal_soc:
                action = 0  # charge
            elif env_clone.current_soc > env_clone.goal_soc:
                action = 1  # discharge
            else:
                action = 2  # idle

            _, _, done, _, info = env_clone.simple_step(action)
            
            actions.append(action)
            path.append(env_clone.agent_pos)
            if done:
                break

        return path , actions, env_clone.money
    
    def dummy_step_feasibility_check(self,action, graph = None):
        """ Performs a dummy step to check feasibility of the action without modifying the environment's state.
        This is useful for evaluating actions without actually applying them.
        Parameters:
            action: the action to check (0 = charge, 1 = discharge, 2 = idle)
            graph: the multigraph used for pathfinding
        Returns:
            1 if the action is feasible (i.e., leads to a valid path),
            0 if not feasible (i.e., leads to an infeasible path).
        """
        if graph is None:
            graph = self.graph
        # Clone the environment
        env_clone =  self.clone()
        
        env_clone.simple_step(action)
        _ ,_, feas = env_clone.helper_replay_benchmark(graph=graph)
        

        

        return feas != np.inf  # return 1 if feasible, 0 if not feasible


    def smart_feasibility_check(self, action=None):
        """
        Check if the goal SoC can be reached from:
        - the current state (if action is None), or
        - the state after applying one action (0=charge, 1=discharge, 2=idle).

        Returns:
            True if goal is reachable within remaining time, False otherwise.
        """
        # Step 1: compute time left
        time_left = self.departure_time - self.current_time
        if time_left <= 0:
            return False

        # Step 2: simulate one step if action is defined
        if action is None:
            temp_soc = int(self.current_soc)
        elif action == 0:
            temp_soc = charge_soc(self.current_soc, dt=15)
            time_left -= 1
        elif action == 1:
            temp_soc = discharge_soc(self.current_soc, dt=15)
            time_left -= 1
        elif action == 2:
            temp_soc = int(self.current_soc)
            time_left -= 1
        else:
            raise ValueError(f"Invalid action: {action}. Expected 0 (charge), 1 (discharge), 2 (idle), or None.")

        goal = int(self.goal_soc)
        steps = 0

        #print(f"[FeasCheck] Action={action}, SoC={temp_soc}, Goal={goal}, Time left={time_left}")

        # Step 3: simulate path toward goal
        if temp_soc < goal:
            tmp_soc_1 = temp_soc
            tmp_soc_2 = temp_soc
            steps_1 = steps_2 = 0
            while tmp_soc_1 < goal and steps_1 < time_left:
                tmp_soc_1 = charge_soc(tmp_soc_1, dt=15)
                steps_1 += 1
            while tmp_soc_2 < goal and steps_2 < time_left - 1:
                tmp_soc_2 = charge_soc(tmp_soc_2, dt=15)
                steps_2 += 1
            steps_2 += 1
            return tmp_soc_1 == goal or tmp_soc_2 == goal

        elif temp_soc > goal:
            tmp_soc_1 = temp_soc
            tmp_soc_2 = temp_soc
            steps_1 = steps_2 = 0
            while tmp_soc_1 > goal and steps_1 < time_left:
                tmp_soc_1 = discharge_soc(tmp_soc_1, dt=15)
                steps_1 += 1
            while tmp_soc_2 > goal and steps_2 < time_left - 1:
                tmp_soc_2 = discharge_soc(tmp_soc_2, dt=15)
                steps_2 += 1
            steps_2 += 1
            return tmp_soc_1 == goal or tmp_soc_2 == goal

        else:
            # Already at goal
            return True





    def simple_step(self, action):
        assert self.action_space.contains(action)

        # ---------- cache state BEFORE action --------------------------- #
        prev_time     = self.current_time
        prev_soc      = self.current_soc
        prev_money    = self.money
        trem     = self.departure_time - prev_time
        soc_gap  = self.goal_soc - self.current_soc 
        gap   = (abs(self.goal_soc - self.current_soc) - R_MAX_PCT * trem) / (self.goal_soc - 0) # scale to [0,1] range

        # ---------- apply action (charge / discharge) ------------------- #
        if action == 0 and self.current_soc < 100:
            self.current_soc = charge_soc(self.current_soc, dt=15)
            #print(f"Charging: {self.current_soc:.1f}% SoC")
        elif action == 1 and self.current_soc > 20:
            self.current_soc = discharge_soc(self.current_soc, dt=15)
            #print(f"Discharging: {self.current_soc:.1f}% SoC")
        elif action == 2:
            pass
            #print(f"Idle: {self.current_soc:.1f}% SoC")

        self.current_soc  = np.clip(self.current_soc, 0, 100)
        self.current_time += 1
        self.agent_pos     = (self.current_time, int(self.current_soc))

        # ---------- update money ---------------------------------------- #
        OFFSET = int(self.arrival_time * 60 // 15)
        idx    = prev_time + OFFSET + 96 * self.day
        price  = (self.price_df["real_price"].iloc[idx]
                if self.grid[prev_time, int(prev_soc), 3] != 255
                else self.price_df["forecast_price"].iloc[idx])

        BAT_CAP_KWH = 75               # battery nameplate
        ETA_C, ETA_D = 0.8, 0.8      # charging / discharging efficiencies
        try:
            max_predicted_price = max(self.price_df["forecast_price"].iloc[idx - prev_time : idx - prev_time +96]) #used to scall reward
        except ValueError:
            print("OFFSET : ", OFFSET)
            print("index : ", idx)
            print('day : ',self.day)
            print('prev_time : ',prev_time)
            print('arrival time', self.arrival_time)
            print('[idx - prev_time : idx - prev_time +96]', '[',idx - prev_time, ':',idx - prev_time +96,']')
        delta_soc   = (self.current_soc - prev_soc) / 100   # signed fraction
        E_bat       = BAT_CAP_KWH * delta_soc               # +ve charge, –ve discharge 
        price_kWh = price /100                              # $ per kWh we dicvided by 100 and not 1000 to simulate realistic retail prices (the data has only wholsales prices)

        if E_bat > 0:                                       # CHARGE
            E_grid = E_bat / ETA_C                          # what the meter sees
            cash   = + price_kWh * E_grid                   # cost → positive
        else:                                               # DISCHARGE
            E_grid = -E_bat * ETA_D                         # export (positive kWh)
            cash   = - price_kWh * E_grid                   # revenue → negative

        self.money += cash
        self._update_money_bar()


        # ---------- feasibility shaping -------------------------------- #
       
        
        
        if gap >= MIN_GAP_NORMALIZED and soc_gap > 0 and action == 0:
            feas_shaping = + 0.2
        elif gap >= MIN_GAP_NORMALIZED and soc_gap > 0 and action == 1:
            feas_shaping = -gap
        elif gap >= MIN_GAP_NORMALIZED and soc_gap > 0 and action == 2:
            feas_shaping = -gap/2
        elif gap >= MIN_GAP_NORMALIZED and soc_gap < 0 and action == 1:
            feas_shaping = + 0.2
        elif gap >= MIN_GAP_NORMALIZED and soc_gap < 0 and action == 0:
            feas_shaping = -gap
        elif gap >= MIN_GAP_NORMALIZED and soc_gap < 0 and action == 2:
            feas_shaping = -gap/2
        else:
            feas_shaping = 0.0
        
        #print('feas_shaping = ', feas_shaping, ' gap = ', gap, ' soc_gap = ', soc_gap)

        # ---------- compose reward ------------------------------------- #
        

        done = self.current_time >= self.departure_time
        soc_gap = self.goal_soc - self.current_soc
        
        reward_money = -cash / max_predicted_price          # saving / earning → positive
        #reward =  feas_shaping  + reward_money
        reward = reward_money
        
        if done:
            if soc_gap == 0:
                reward += FEAS_BONUS
            """ 
            elif soc_gap == 1:
                reward -= FEAS_BONUS/2
            elif soc_gap == -1:
                reward += FEAS_BONUS/2
            elif soc_gap == 2:
                reward -= FEAS_BONUS/5
            elif soc_gap == -2:
                reward += FEAS_BONUS/5
            else :
                reward -=FEAS_BONUS
            """
        
            
            
        return self._get_observation(), reward, done, False, {"money": self.money, "reached_goal": True if soc_gap == 0 else False}
    

    def clone(self):
        clone = ChargeTrekEnv(
            grid=self.grid.copy(),
            price_df=self.price_df,
            day=self.day,
            arrival_time=self.arrival_time,
            goal_soc=self.goal_soc,
            departure_time=self.departure_time
        )
        clone.current_soc = self.current_soc
        clone.current_time = self.current_time
        clone.agent_pos = self.agent_pos
        clone.money = self.money
        clone.money_bar = self.money_bar.copy()
        clone.has_solution = self.has_solution
        clone.graph = self.graph  # if immutable or cacheable
        return clone

