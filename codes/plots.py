import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_vehicle_duration_hist(df, max_minutes=60, bin_size=2):
    # agregacja: pierwszy i ostatni timestamp (dane są posortowane)
    durations = df.groupby("vehicle_id")["timestamp"].agg(["first", "last"])
    
    # czas w minutach
    durations["duration_min"] = (durations["last"] - durations["first"]) / 60

    values = durations["duration_min"].values

    # podział: < max i >= max
    below = values[values < max_minutes]
    above = values[values >= max_minutes]

    # koszyki dla < max
    bins = np.arange(0, max_minutes, bin_size)
    counts, edges = np.histogram(below, bins=np.append(bins, max_minutes))

    # dodanie ostatniego bucketu (60+)
    counts = np.append(counts, len(above))

    # procenty
    counts_percent = counts / counts.sum() * 100

    # etykiety
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)]
    labels.append(f"{max_minutes}+")

    # wykres
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(counts_percent)), counts_percent)

    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.xlabel("Duration [minutes]")
    plt.ylabel("Percentage [%]")
    plt.title("Vehicle duration histogram (last bin = 60+ min)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return durations

def plot_stop_histogram_percentage(df, speed_threshold=10, include_gaps=True, 
                                   gap_threshold_sec=60, min_stop_min=2, max_limit_min=60):
    """
    Analyzes vehicle stops with a minimum duration filter and percentage-based histogram.
    
    Parameters:
    - min_stop_min: Stops shorter than this (e.g., 2 min) are filtered out as "false stops".
    - max_limit_min: The "X+" overflow bin boundary.
    """
    
    # 1. Data Prep & Duration Calculation
    df = df.sort_values(['vehicle_id', 'timestamp']).copy()
    df['duration_sec'] = df.groupby('vehicle_id')['timestamp'].diff().shift(-1)
    
    # 2. Define Stop Segments
    low_speed = df['speed'] < speed_threshold
    if include_gaps:
        is_gap = df['duration_sec'] > gap_threshold_sec
        df['is_stop_segment'] = low_speed | is_gap
    else:
        df['is_stop_segment'] = low_speed
        
    df = df.dropna(subset=['duration_sec'])
    
    # 3. Group Continuous Blocks
    df['block_id'] = (df['is_stop_segment'] != df['is_stop_segment'].shift(1)).cumsum()
    stop_events = df[df['is_stop_segment']].groupby(['vehicle_id', 'block_id'])['duration_sec'].sum()
    
    # Convert to minutes
    durations_min = stop_events / 60
    
    # 4. Filter out "False Stops" (Minimum Duration Filter)
    valid_stops = durations_min[durations_min >= min_stop_min]
    
    if valid_stops.empty:
        print("No stops found matching the criteria.")
        return None

    # 5. Prepare Data for Histogram (Clipping for the Overflow Bin)
    plot_data = np.clip(valid_stops, 0, max_limit_min)
    
    # Define Bins: 2-minute intervals starting from 0
    bins = np.arange(0, max_limit_min + 2, 2)
    
    # 6. Plotting with Percentages
    plt.figure(figsize=(12, 7))
    
    # Calculate weights to show percentages instead of counts
    weights = np.ones(len(plot_data)) / len(plot_data) * 100
    
    n, bins_edges, patches = plt.hist(plot_data, bins=bins, weights=weights, 
                                      color='#2ecc71', edgecolor='black', alpha=0.8)
    
    # Customize X-axis labels
    plt.xticks(bins) # Show every 4 mins on the axis for readability
    ax = plt.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    if labels:
        labels[-1] = f"{max_limit_min}+"
    ax.set_xticklabels(labels)
    
    # Highlight the overflow bin
    if len(patches) > 0:
        patches[-1].set_facecolor('#e67e22') 
    
    # Formatting
    plt.title(f'Vehicle Stop Duration Distribution (%)\n'
              f'Filtering: >{min_stop_min} min stops | Speed < {speed_threshold} km/h', fontsize=14)
    plt.xlabel('Stop Duration [minutes]', fontsize=12)
    plt.ylabel('Percentage of All Valid Stops [%]', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add percentage symbol to Y-axis ticks
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xlim(min_stop_min, max_limit_min)
    plt.show()
    
    # Quick Statistics
    print(f"Total stops analyzed: {len(durations_min)}")
    print(f"Stops after filtering (<{min_stop_min} min removed): {len(valid_stops)}")
    print(f"Median stop duration: {valid_stops.median():.2f} minutes")
    
    return valid_stops


