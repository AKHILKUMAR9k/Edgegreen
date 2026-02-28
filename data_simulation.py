import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_historical_data(periods=100, freq='s'):
    """Generates historical solar irradiance data."""
    np.random.seed(42)  # For reproducibility, though in real-time we'd want true randomness
    
    end_time = datetime.now()
    start_time = end_time - timedelta(seconds=periods)
    
    time_index = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # Base irradiance (simulating daytime) + some noise + some sinusoidal pattern
    base_irradiance = 800 + 100 * np.sin(np.linspace(0, 4 * np.pi, len(time_index)))
    noise = np.random.normal(0, 15, len(time_index))
    
    irradiance = base_irradiance + noise
    
    # Add a sudden drop for anomaly demonstration (optional, we can trigger this dynamically)
    # df.loc[df.index[-20:], 'irradiance'] *= 0.7  
    
    df = pd.DataFrame({'timestamp': time_index, 'irradiance': irradiance})
    df.set_index('timestamp', inplace=True)
    return df

def generate_new_data_point(last_timestamp, last_value, drop_active=False):
    """Generates a single new data point, optionally with a simulated cloud cover drop."""
    new_timestamp = last_timestamp + timedelta(seconds=1)
    
    # Random walk with mean reversion to a daytime average
    mean_level = 800
    reversion_speed = 0.1
    noise = np.random.normal(0, 10)
    
    change = reversion_speed * (mean_level - last_value) + noise
    new_value = last_value + change
    
    if drop_active:
        new_value *= 0.6 # simulate 40% drop

    return new_timestamp, max(0, new_value) # Ensure non-negative
