import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

class MockForecaster:
    def __init__(self):
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, max_depth=3)
        self._is_trained = False
        
    def _create_features(self, df):
        """Creates time series features from datetime index"""
        df = df.copy()
        df['second'] = df.index.second
        df['minute'] = df.index.minute
        return df
        
    def _create_lag_features(self, df, lags=3):
        """Creates lag features"""
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['irradiance'].shift(i)
        return df.dropna()

    def train(self, df):
        """Trains a mock model on recent historical data"""
        if len(df) < 50:
            return False # Need more data
            
        df_feat = self._create_features(df)
        df_feat = self._create_lag_features(df_feat)
        
        # Super simple mock training: predict the mean + some trend
        X = df_feat.drop('irradiance', axis=1)
        y = df_feat['irradiance']
        
        self.model.fit(X, y)
        self._is_trained = True
        return True
        
    def predict_next_30s(self, current_data):
        """Forecasts the next 30 seconds given the latest context"""
        if not self._is_trained:
            # Fallback if not trained: just return current value + noise
            last_val = current_data['irradiance'].iloc[-1]
            last_time = current_data.index[-1]
            
            future_times = pd.date_range(start=last_time + pd.Timedelta(seconds=1), periods=30, freq='S')
            forecast_vals = [last_val + np.random.normal(0, 5) for _ in range(30)]
            
            forecast_df = pd.DataFrame({'timestamp': future_times, 'predicted_irradiance': forecast_vals})
            forecast_df.set_index('timestamp', inplace=True)
            return forecast_df
            
        # Simplified prediction loop (in reality, we'd feed predictions back sequentially, or use a multi-output model)
        # For prototype speed, we'll generate mock features for the next 30s based on the last known values
        last_val = current_data['irradiance'].iloc[-1]
        last_time = current_data.index[-1]
        
        future_times = pd.date_range(start=last_time + pd.Timedelta(seconds=1), periods=30, freq='S')
        forecast_vals = []
        
        current_val = last_val
        for t in future_times:
             # Basic random walk for the mock forecast
             change = np.random.normal(0, 2)
             current_val = current_val + change
             forecast_vals.append(current_val)
             
        forecast_df = pd.DataFrame({'timestamp': future_times, 'predicted_irradiance': forecast_vals})
        forecast_df.set_index('timestamp', inplace=True)
        return forecast_df
