import numpy as np

# EV system parameters
BATTERY_CAPACITY_KWH = 75.0
CHARGER_POWER_KW = 7.5
EFFICIENCY = 0.80  # 80%

# Derived parameters
ENERGY_PER_PERCENT = BATTERY_CAPACITY_KWH / 100.0  # 0.75 kWh per 1% SoC
POWER_FLOW = CHARGER_POWER_KW * EFFICIENCY         # usable power = 6.0 kW
ENERGY_PER_MINUTE = POWER_FLOW / 60.0              # kWh/min = 0.1
DELTA_SOC = ENERGY_PER_MINUTE / ENERGY_PER_PERCENT # % SoC/min = 0.1333...

def charge_soc(current_soc, dt=1):
    soc = float(current_soc)
    if soc >= 100:
        return 100.0
    if soc > 80:
        # Throttle charging in CV phase
        slowdown = 0.5 + 0.025 * (soc - 80)
        factor = min(1.0, max(0.5, slowdown))
        delta = DELTA_SOC * factor
    else:
        # Full speed below 80% (CC phase)
        delta = DELTA_SOC
    return int(min(100.0, soc + delta * dt))

def discharge_soc(current_soc, dt=1):
    soc = float(current_soc)
    if soc <= 0:
        return 0.0
    if soc > 80:
        delta = DELTA_SOC  # full speed
    elif soc >= 20:
        delta = DELTA_SOC  # constant rate
    else:
        slowdown = 0.5 + 0.025 * (20 - soc)
        factor = min(1.0, max(0.5, slowdown))
        delta = DELTA_SOC * factor
    return int(max(0.0, soc - delta * dt))
