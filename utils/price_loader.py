import pandas as pd
import numpy as np
import os

def load_caiso_prices(data_dir, node_name="SMD4_ASR-APND LMP"):
    # Real-time (Q1 + Q2)
    rt_q1_path = os.path.join(data_dir, "caiso_lmp_rt_15min_interfaces_2025Q1.csv")
    rt_q2_path = os.path.join(data_dir, "caiso_lmp_rt_15min_interfaces_2025Q2.csv")
    #rt_q3_path = os.path.join(data_dir, "caiso_lmp_rt_15min_interfaces_2024Q3.csv")
    #rt_q4_path = os.path.join(data_dir, "caiso_lmp_rt_15min_interfaces_2024Q4.csv")
    rt_q1 = pd.read_csv(rt_q1_path, skiprows=3)
    rt_q2 = pd.read_csv(rt_q2_path, skiprows=3)
    #rt_q3 = pd.read_csv(rt_q3_path, skiprows=3)
    #rt_q4 = pd.read_csv(rt_q4_path, skiprows=3)
    rt = pd.concat([rt_q1, rt_q2], ignore_index=True)

    # Parse timestamps and keep desired node
    rt["timestamp"] = pd.to_datetime(rt["UTC Timestamp (Interval Ending)"])
    rt = rt[["timestamp", node_name]].rename(columns={node_name: "real_price"})
    rt = rt.sort_values("timestamp")

    # Day-ahead
    da_path = os.path.join(data_dir, "caiso_lmp_da_hr_interfaces_2025.csv")
    da = pd.read_csv(da_path, skiprows=3)
    da["timestamp"] = pd.to_datetime(da["UTC Timestamp (Interval Ending)"])
    da = da[["timestamp", node_name]].rename(columns={node_name: "forecast_price"})
    da = da.sort_values("timestamp")

    # Resample to 15-min to match RT
    da = da.set_index("timestamp").resample("15min").ffill().reset_index()

    # Merge on timestamps
    merged = pd.merge_asof(rt, da, on="timestamp", direction="nearest")
    merged["real_price"] = merged["real_price"].fillna(merged["forecast_price"])
    return merged

def create_rgba_grid(price_df,arrival_time = 16 ,steps=96, soc_levels=101, day=0):
    import numpy as np

    START_HOUR = int(arrival_time)
    STEP_SIZE_MIN = 15
    OFFSET = START_HOUR * 60 // STEP_SIZE_MIN  # = 64
    DAY_START = OFFSET + 96 * day
    DAY_END = DAY_START + steps

    # Slice data for the current day
    daily_forecast = price_df["forecast_price"].iloc[DAY_START:DAY_END].reset_index(drop=True)
    daily_real = price_df["real_price"].iloc[DAY_START:DAY_END].reset_index(drop=True)


    # Per-day normalization of forecast
    f_min, f_max = daily_forecast.min(), daily_forecast.max()
    forecast_norm = (daily_forecast - f_min) / (f_max - f_min + 1e-6)
    forecast_scaled = (forecast_norm * 255).astype(np.uint8)

    # Error sign (B channel)
    error_sign = np.where((daily_real - daily_forecast) > 0, 100, 0)

    # Global 95th percentile error for alpha scaling
    all_errors = np.abs(price_df["real_price"] - price_df["forecast_price"])
    error_max = np.percentile(all_errors, 95) or 1  # avoid division by zero

    error_real = np.abs(daily_real - daily_forecast)
    alpha_scaled = (1 - np.clip(error_real / error_max, 0, 1)) * 255
    alpha_scaled = alpha_scaled.astype(np.uint8)

    grid = np.zeros((steps, soc_levels, 4), dtype=np.uint8)
    for t in range(min(steps, len(daily_forecast))):
        for s in range(soc_levels):
            r = forecast_scaled.iloc[t]
            g = 255 - r
            b = error_sign[t]
            a = alpha_scaled[t]
            grid[t, s] = [r, g, b, a]

    return grid