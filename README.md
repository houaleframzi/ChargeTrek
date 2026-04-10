# ChargeTrek: Visual Reinforcement Learning for EV Charging in V2G Systems

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![Framework](https://img.shields.io/badge/Framework-Gymnasium-orange.svg)](#)
[![RL](https://img.shields.io/badge/RL-C51%20%7C%20DAgger-blueviolet.svg)](#)

**ChargeTrek** is a **visual reinforcement learning (Visual RL)** framework that turns electric vehicle (EV) charging/discharging into an **Atari‑style decision‑making problem**. Instead of feeding the agent numerical tables or hand‑crafted features, ChargeTrek encodes electricity price signals, forecast errors, and battery dynamics directly into **RGBA images**. A convolutional neural network (CNN) then learns to navigate this visual world, deciding when to **charge**, **discharge**, or **stay idle**.

The project targets **vehicle‑to‑grid (V2G)** scenarios and is built to be:
- **Interpretable** – you can literally *see* what the agent sees.
- **Robust** – works under real‑world price uncertainty.
- **Safe** – hard constraints on battery state‑of‑charge are always respected.

---

## Table of Contents

1. [What Problem Does ChargeTrek Solve?](#what-problem-does-chargetrek-solve)
2. [Key Features](#key-features)
3. [Environment Design (The “Game”)](#environment-design-the-game)
4. [Supported Algorithms & Baselines](#supported-algorithms--baselines)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
   - [Playing with a Trained Agent](#playing-with-a-trained-agent)
   - [Training Your Own Agent](#training-your-own-agent)
7. [Project Structure](#project-structure)
8. [Data Source](#data-source)
9. [Performance Highlights](#performance-highlights)
10. [Deployment Perspective](#deployment-perspective)
11. [Citation](#citation)
12. [License](#license)
13. [Contributing](#contributing)

---

## What Problem Does ChargeTrek Solve?

Electricity prices are volatile, and forecasts are never perfect. An EV owner (or fleet operator) wants to:
- **Minimise charging costs**.
- **Sell energy back to the grid** (V2G) when prices are high.
- **Guarantee a desired battery level** by the next departure.

Traditional approaches use linear programming or model‑predictive control, which rely on accurate forecasts and are often opaque. ChargeTrek takes a different route: it teaches an **AI agent to *see* the problem** and learn a policy directly from pixels, much like DeepMind’s DQN plays Atari games.

---

## Key Features

- **Visual RL Environment** – A custom `gymnasium` environment that renders the charging task as a 2D grid image.
- **Three Action Modes** – Charge (+), Discharge (–), Idle (→).
- **Multiple Training Strategies**
  - **C51 (Distributional DQN)** – pure RL with a categorical value distribution.
  - **DAgger (Dataset Aggregation)** – imitation learning that mixes expert demonstrations with the agent’s own experience.
  - **IB‑C51** – C51 initialised with expert trajectories.
- **Safety Guarantee** – A **Bellman‑Ford**‑based fallback planner ensures the agent never violates SoC constraints.
- **Real‑World Data** – Uses **CAISO** (California ISO) day‑ahead and real‑time prices.
- **Scalable Training** – On‑disk replay buffers (`LMDB`) allow training on millions of transitions without running out of RAM.
- **Interactive Visualisation** – Watch the agent move through the grid in real‑time with `matplotlib`.

---

## Environment Design (The “Game”)

The environment is a **2D grid**:

- **X‑axis** → Time (up to 24 h, 15‑min resolution → 96 steps)
- **Y‑axis** → State of Charge (SoC, 0 % to 100 %, 101 discrete levels)

Each **cell** is an **RGBA pixel** encoding:

| Channel | Meaning |
|---------|---------|
| **R**   | Red intensity proportional to electricity **price** (expensive → bright red) |
| **G**   | Green intensity proportional to **cheapness** (low price → bright green) |
| **B**   | Blue = **forecast error sign** (100 if real price > forecast, else 0) |
| **A**   | Alpha = **uncertainty magnitude** (transparent = low error, opaque = high error) |

The agent starts at a specific `(time, SoC)` cell and can move **up** (charge), **down** (discharge), or **right** (idle). The goal is to reach the **target SoC** at the **departure time** while maximising cumulative reward (which reflects monetary profit).

## Demonstration

**Environment demonstration example

<p align="center">
  <img src="/visual_example_chargetrek.gif" width="800" alt="ChargeTrek Demonstration">
</p>

<p align="center"><em>Figure: ChargeTrek Demonstration</em></p>

---

## Supported Algorithms & Baselines

| Method | Description |
|--------|-------------|
| **C51** | Distributional DQN that learns a full value distribution. |
| **DAgger‑DQN** | Supervised learning on expert‑labelled data, iteratively refined with the agent’s own actions. |
| **IB‑C51** | C51 pre‑trained with imitation (behaviour cloning) and then fine‑tuned. |
| **Immediate Charger** | Naïve human‑like behaviour (charge immediately until full). |
| **Stepwise Planner** | Short‑term optimisation that re‑plans every 15 minutes. |
| **Optimal (Oracle)** | Full future knowledge via Bellman‑Ford on the price graph – **upper performance bound**. |

All RL agents use the same CNN backbone (3 convolutional layers + 2 fully‑connected layers) that processes the `(4, 101, 96)` image.

---

## Installation

> **Requires Python 3.10** (as specified in the repository). The code was tested on Python 3.10.12.

### 1. Clone the repository

```bash
git clone https://github.com/houaleframzi/ChargeTrek.git
cd ChargeTrek
