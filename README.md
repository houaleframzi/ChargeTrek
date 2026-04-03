# 🔋 ChargeTrek: Visual Reinforcement Learning for EV Charging in V2G Systems

**ChargeTrek** is a **visual reinforcement learning (Visual RL)** framework for optimizing electric vehicle (EV) charging and discharging in **vehicle-to-grid (V2G)** systems.

It reformulates EV charging as an **Atari-style visual decision-making problem**, where agents learn directly from image-based representations of electricity prices, uncertainty, and battery state.

---

## 🚀 Overview

Traditional EV charging optimization relies on static or numerical models with limited interpretability. ChargeTrek introduces a **gamified, visual learning environment** that enables agents to:

* Learn from **high-dimensional visual inputs (images)**
* Adapt to **uncertain and dynamic electricity prices**
* Make **sequential decisions** (charge / discharge / idle)
* Respect **battery and mobility constraints**

The result is a framework that is both **interpretable and performant**, bridging reinforcement learning, imitation learning, and energy systems optimization.

---

## 🎮 Environment Design

ChargeTrek models EV charging as a **2D grid-based game**:

* **X-axis** → Time (up to 24h, 15-min resolution)
* **Y-axis** → State of Charge (SoC, 0–100%)

Each state is encoded as an **RGBA image**:

* 🔴 **Red / Green** → Electricity price (expensive vs cheap)
* 🔵 **Blue** → Forecast error direction
* ⚪ **Alpha** → Uncertainty magnitude

The agent navigates this grid:

* ↗ Charge
* ↘ Discharge
* → Idle

---

## 👁️ Visual Reinforcement Learning

ChargeTrek leverages **Visual RL**, where:

* The environment is treated as an **image**
* Policies are learned using **Convolutional Neural Networks (CNNs)**
* No handcrafted features are required

This allows the agent to:

* Recognize **spatial and temporal patterns**
* Interpret **price signals visually**
* Learn **robust strategies under uncertainty**

---

## 🤖 Methods

### Reinforcement Learning

* **C51 (Distributional DQN)**
  Learns value distributions for stable decision-making

### Imitation Learning

* **Imitation-Based Reward Shaping**
* **DAgger (Dataset Aggregation)**

### Hybrid Approach

* Combines learned policies with **graph-based fallback (Bellman-Ford)**
* Ensures **feasibility and safety (SoC constraints always satisfied)**

---

## 📊 Baselines

ChargeTrek is benchmarked against:

* **Immediate Charger** → naive human-like behavior
* **Stepwise Planner** → short-term optimization
* **Optimal (Oracle)** → full future knowledge (upper bound)

---

## 📈 Key Results

* Up to **34% cost reduction** vs. typical charging behavior
* **DAgger agents** achieve near-optimal performance
* Visual RL enables:

  * Better **interpretability**
  * Stable learning
  * Strong generalization

---

## ⚡ Data

* Source: **CAISO electricity market**
* Includes:

  * Day-ahead price forecasts
  * Real-time prices (15-min resolution)
* Captures **real-world volatility and uncertainty**

---

## 🚗 Deployment Perspective

* Designed for **real-time inference on EV embedded systems**
* Lightweight execution (no training required onboard)
* Compatible with:

  * Vehicle-side decision-making
  * Cloud retraining + OTA updates

---

## 🧠 Research Contributions

* First **visual RL framework** for EV charging optimization
* Novel **image-based state representation (price + uncertainty)**
* Integration of **RL, IL, and optimization** in one environment
* Demonstrates **gamification as a tool for energy optimization**

---

## 📄 Citation

If you use this repository, please cite:

```
Houalef A-R, Mendoza JE, Delavernhe F, Senouci S-M. ChargeTrek: A gamified visual learning framework for EV charging in V2G systems . Applied Energy. doi: https://doi.org/10.1016/j.apenergy.2026.127853
```

---

## 📌 Notes

This repository is under active development.
**Files and code structure will be updated soon after refactoring.**
