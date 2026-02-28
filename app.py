import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from data_simulation import generate_historical_data, generate_new_data_point
from forecasting import MockForecaster
from anomaly_detection import detect_anomaly

# Set page layout
st.set_page_config(layout="wide", page_title="EdgeGreen Prototype")

st.title("EdgeGreen Prototype - Real-Time Solar Forecasting")

st.markdown("""
<div style='display: flex; gap: 15px; font-size: 0.95em; font-weight: 500; color: #a3e635; margin-bottom: 20px; background: rgba(163, 230, 53, 0.1); border: 1px solid rgba(163, 230, 53, 0.2); padding: 8px 15px; border-radius: 20px; width: fit-content;'>
    <span>🟢 Edge AI Mode: Active</span>
    <span>⚡ Running on Low-Power Inference</span>
</div>
""", unsafe_allow_html=True)

# --- CSS Styling ---
# A premium, glassmorphic dark-theme look requested by the user instructions
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
         color: #38bdf8 !important;
         font-weight: 600;
         text-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
    }
    
    /* Metrics and Data Cards (Glassmorphism) */
    div[data-testid="stMetricValue"] {
        color: #a7f3d0 !important;
        font-family: monospace;
        font-size: 2.5rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 1.1rem !important;
    }
    
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Stabilization Warning */
    .stabilization-warning {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(185, 28, 28, 0.2));
        border-left: 4px solid #ef4444;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 1rem;
        color: #fca5a5;
        font-weight: 600;
        animation: pulse-red 2s infinite;
    }
    .stabilization-normal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1));
        border-left: 4px solid #10b981;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 1rem;
        color: #6ee7b7;
        font-weight: 600;
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    }
</style>
""", unsafe_allow_html=True)


# --- State Initialization ---
if 'history_df' not in st.session_state:
    st.session_state.history_df = generate_historical_data(periods=100)

if 'forecaster' not in st.session_state:
    st.session_state.forecaster = MockForecaster()
    # Mock training initially
    st.session_state.forecaster.train(st.session_state.history_df.iloc[:-20])

if 'is_running' not in st.session_state:
    st.session_state.is_running = False

if 'stability_score' not in st.session_state:
    st.session_state.stability_score = 99.0

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls")
    
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶ Start Stream", use_container_width=True):
            st.session_state.is_running = True
    with col_stop:
        if st.button("🛑 Stop Stream", use_container_width=True):
            st.session_state.is_running = False
            
    st.markdown("---")
    st.subheader("Simulate Anomalies")
    # Button to instantly drop irradiance by ~40%
    simulate_drop = st.button("Trigger Cloud Cover Drop", use_container_width=True, type="primary")


# --- Main Layout ---
col1, col2 = st.columns([2, 1])

# Placeholders for dynamic content
with col1:
    st.subheader("Live Forecast & Telemetry")
    chart_placeholder = st.empty()

with col2:
    st.subheader("System Status")
    status_placeholder = st.empty()
    metric_placeholder = st.empty()
    score_placeholder = st.empty()

# --- Main App Loop ---

while st.session_state.is_running:
    # 1. Update Data
    last_idx = st.session_state.history_df.index[-1]
    last_val = st.session_state.history_df['irradiance'].iloc[-1]
    
    # Check if user clicked the "Simulate Drop" button
    # Note: Streamlit buttons revert to False after the app reruns. 
    # To sustain the drop briefly, we trigger it for this tick.
    drop_active = simulate_drop 
    
    new_time, new_val = generate_new_data_point(last_idx, last_val, drop_active=drop_active)
    
    # Append the new point
    new_row = pd.DataFrame({'irradiance': [new_val]}, index=[new_time])
    st.session_state.history_df = pd.concat([st.session_state.history_df, new_row]).tail(120) # Keep last 2 minutes

    # 2. Forecasting
    df_context = st.session_state.history_df.tail(60) # use last 60s for forecast context
    forecast_df = st.session_state.forecaster.predict_next_30s(df_context)
    
    # If the user just pressed "Trigger Cloud Cover Drop", our mock predictor needs to 
    # "predict" the drop across the upcoming 30s to trigger the stabilization warning.
    # We will manually inject the drop into the *forecast* if the button was clicked
    # so that the system detects it 30s ahead.
    if drop_active:
        forecast_df['predicted_irradiance'] *= 0.6
        
    last_actual_value = df_context['irradiance'].iloc[-1]
    
    # 3. Anomaly Detection
    is_anomaly, msg = detect_anomaly(last_actual_value, forecast_df)

    if is_anomaly:
        st.session_state.stability_score = max(72.0, st.session_state.stability_score - 4.5)
    else:
        st.session_state.stability_score = min(99.8, st.session_state.stability_score + 1.2)

    # 4. Update UI
    
    # Graph Update
    fig = go.Figure()
    
    # Actual Data Trace
    fig.add_trace(go.Scatter(
        x=st.session_state.history_df.index,
        y=st.session_state.history_df['irradiance'],
        mode='lines',
        name='Actual Irradiance',
        line=dict(color='#38bdf8', width=2),
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.1)'
    ))
    
    # Forecast Data Trace
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['predicted_irradiance'],
        mode='lines',
        name='Predicted (Next 30s)',
        line=dict(color='#fbbf24', width=2, dash='dash')
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        xaxis=dict(showgrid=False, title='Time', color='#94a3b8'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Irradiance (W/m²)', color='#94a3b8', range=[0, 1100]),
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")
    
    # Status Update
    if is_anomaly:
        status_html = f"""
        <div class="status-card">
            <h4>Grid Stability</h4>
            <div class="stabilization-warning" style="line-height: 1.6;">
                🔴 <b>Forecasted Drop > 15% Detected</b><br>
                ⚡ <b>Inverter Stabilization Activated</b><br>
                📊 <b>Load Redistribution Initiated</b><br>
                <small style="margin-top:10px; display:block;"><i>Details: {msg}</i></small>
            </div>
        </div>
        """
        # Requirement: print "Inverter Stabilization Triggered"
        print("Inverter Stabilization Triggered")
    else:
        status_html = f"""
        <div class="status-card">
            <h4>Grid Stability</h4>
            <div class="stabilization-normal">
                ✅ Nominal Output<br>
                <small>No significant drops predicted.</small>
            </div>
        </div>
        """
        
    status_placeholder.markdown(status_html, unsafe_allow_html=True)
    
    # Metrics
    metric_placeholder.metric(label="Live Irradiance", value=f"{last_actual_value:.0f} W/m²")
    score_placeholder.metric(label="Grid Stability Score", value=f"{st.session_state.stability_score:.1f}%")

    # Sleep for real-time effect
    time.sleep(1)
    
    # Break early if button is clicked to avoid rerunning forever outside streamlit's control
    # But streamlit's loop handles rerunning app.py top-to-bottom.
    # Since we are in a while loop, we must call st.rerun() if we want it to check for new button states
    if st.session_state.is_running:
        st.rerun()

else:
    # Render static state if not running
    # 1. Update Data
    last_actual_value = st.session_state.history_df['irradiance'].iloc[-1]
    
    # 2. Forecasting
    df_context = st.session_state.history_df.tail(60)
    forecast_df = st.session_state.forecaster.predict_next_30s(df_context)
    
    # 3. Anomaly Detection
    is_anomaly, msg = detect_anomaly(last_actual_value, forecast_df)

    if is_anomaly:
        st.session_state.stability_score = max(72.0, st.session_state.stability_score - 4.5)
    else:
        st.session_state.stability_score = min(99.8, st.session_state.stability_score + 1.2)

    # 4. Update UI
    
    # Graph Update
    fig = go.Figure()
    
    # Actual Data Trace
    fig.add_trace(go.Scatter(
        x=st.session_state.history_df.index,
        y=st.session_state.history_df['irradiance'],
        mode='lines',
        name='Actual Irradiance',
        line=dict(color='#38bdf8', width=2),
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.1)'
    ))
    
    # Forecast Data Trace
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['predicted_irradiance'],
        mode='lines',
        name='Predicted (Next 30s)',
        line=dict(color='#fbbf24', width=2, dash='dash')
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        xaxis=dict(showgrid=False, title='Time', color='#94a3b8'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Irradiance (W/m²)', color='#94a3b8', range=[0, 1100]),
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Status Update
    if is_anomaly:
        status_html = f"""
        <div class="status-card">
            <h4>Grid Stability</h4>
            <div class="stabilization-warning" style="line-height: 1.6;">
                🔴 <b>Forecasted Drop > 15% Detected</b><br>
                ⚡ <b>Inverter Stabilization Activated</b><br>
                📊 <b>Load Redistribution Initiated</b><br>
                <small style="margin-top:10px; display:block;"><i>Details: {msg}</i></small>
            </div>
        </div>
        """
        print("Inverter Stabilization Triggered")
    else:
        status_html = f"""
        <div class="status-card">
            <h4>Grid Stability</h4>
            <div class="stabilization-normal">
                ✅ Nominal Output<br>
                <small>{msg}</small>
            </div>
        </div>
        """
        
    status_placeholder.markdown(status_html, unsafe_allow_html=True)
    metric_placeholder.metric(label="Live Irradiance", value=f"{last_actual_value:.0f} W/m²")
    
    # Notice that we add the score metric here as well, because this is the static part.
    if 'score_placeholder' in globals() or 'score_placeholder' in locals():
        score_placeholder.metric(label="Grid Stability Score", value=f"{st.session_state.stability_score:.1f}%")
    else:
        st.metric(label="Grid Stability Score", value=f"{st.session_state.stability_score:.1f}%")
    
    st.info("Press 'Start Stream' to begin live simulation.")
