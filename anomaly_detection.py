def detect_anomaly(current_value, forecasted_values):
    """
    Detects if there is a significant drop (>15%) in the forecasted values 
    compared to the current reading.
    """
    if len(forecasted_values) == 0:
        return False, "No forecast available"
        
    # Calculate the minimum forecasted value in the window
    min_forecast = forecasted_values['predicted_irradiance'].min()
    
    # Avoid division by zero
    if current_value <= 0:
        return False, "Current value too low to calculate drop %"
        
    # Calculate percentage drop
    drop_percentage = ((current_value - min_forecast) / current_value) * 100
    
    if drop_percentage > 15:
        # Detected!
        msg = f"Warning: Predicted {drop_percentage:.1f}% drop in next 30s! (Current: {current_value:.1f}, Min Forecast: {min_forecast:.1f})"
        return True, msg
        
    return False, "Operating normally"
