import networkx as nx
from utils.soc_mapper import charge_soc, discharge_soc

def build_charge_trek_multigraph(price_df,arrival_time = 16,day=0, start_time=0, end_time=95, soc_levels=101):
    """
    Builds a directed multigraph where each edge stores both real and forecast price-based weights.
    Charging and discharging functions are imported from utils.soc_mapper.

    Parameters:
        price_df: DataFrame with 'real_price' and 'forecast_price'
        day: int, which day to build (0-indexed)
        start_time: int, first time step (default 0)
        end_time: int, last time step (default 95)
        soc_levels: int, number of SoC levels (typically 101)

    Returns:
        G: networkx.MultiDiGraph with two parallel edges per action (real and forecast)
    """
    dt = 15
    OFFSET = int(arrival_time * 60 // dt)
    BAT_CAP_KWH = 75               # battery nameplate
    kwh_per_soc = BAT_CAP_KWH / 100  # 75kWh battery, 80% efficiency
    ETA_C, ETA_D = 0.8, 0.8  

    base_index = OFFSET + 96 * day
    G = nx.MultiDiGraph()
    #print('Graph built for day = ', day)

    for t in range(int(start_time), int(end_time)):
        for s in range(soc_levels):
            current_node = (t, s)
            #print(f"Processing node: {current_node}")
            real_price = price_df["real_price"].iloc[base_index + t]
            forecast_price = price_df["forecast_price"].iloc[base_index + t]

            # Charge
            new_soc = int(charge_soc(s, dt))
            if new_soc <= 100:
                delta_soc = new_soc - s
                cost_real = (real_price / 100) * delta_soc * kwh_per_soc / ETA_C
                cost_forecast = (forecast_price / 100) * delta_soc * kwh_per_soc/ ETA_C
                G.add_edge(current_node, (t + 1, new_soc), key="real", weight=cost_real)
                G.add_edge(current_node, (t + 1, new_soc), key="forecast", weight=cost_forecast)

            # Discharge
            new_soc = int(discharge_soc(s, dt))
            if new_soc >= 20:
                delta_soc = s - new_soc
                gain_real = -(real_price / 100) * delta_soc * kwh_per_soc * ETA_D
                gain_forecast = -(forecast_price / 100) * delta_soc * kwh_per_soc * ETA_D
                G.add_edge(current_node, (t + 1, new_soc), key="real", weight=gain_real)
                G.add_edge(current_node, (t + 1, new_soc), key="forecast", weight=gain_forecast)

            # Idle
            G.add_edge(current_node, (t + 1, s), key="real", weight=0)
            G.add_edge(current_node, (t + 1, s), key="forecast", weight=0)

    return G
