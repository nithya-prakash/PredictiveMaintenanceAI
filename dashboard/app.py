import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

API_URL = "http://api:8000/api/v1" # Docker service name

st.set_page_config(page_title="Industrial AI | Predictive Maintenance", layout="wide")

st.title("🏭 Manufacturing Predictive Maintenance AI")
st.markdown("Real-time monitoring, failure forecasting, and explainable AI for industrial machinery.")

# Sidebar for Machine Selection
st.sidebar.header("Machine Selection")
machine_id = st.sidebar.number_input("Machine ID", min_value=1, max_value=1000, value=1)

# Synthetic current data inputs for the dashboard
st.sidebar.header("Sensor Telemetry")
temp = st.sidebar.slider("Temperature (°C)", 50.0, 150.0, 75.0)
pressure = st.sidebar.slider("Pressure (psi)", 50.0, 150.0, 100.0)
vib = st.sidebar.slider("Vibration (mm/s)", 0.0, 3.0, 0.5)
rpm = st.sidebar.slider("RPM", 1000.0, 2000.0, 1500.0)
volt = st.sidebar.slider("Voltage (V)", 200.0, 250.0, 220.0)
curr = st.sidebar.slider("Current (A)", 10.0, 30.0, 15.0)
oil = st.sidebar.slider("Oil Quality (%)", 0.0, 100.0, 80.0)

payload = {
    "machine_id": machine_id,
    "temperature": temp,
    "pressure": pressure,
    "vibration": vib,
    "rpm": rpm,
    "voltage": volt,
    "current": curr,
    "oil_quality": oil
}

col1, col2, col3 = st.columns(3)

if st.sidebar.button("Run AI Diagnostics"):
    with st.spinner("Running ML Inference..."):
        try:
            # Parallel API calls would be better, but doing serial for simplicity in prototyping
            rul_res = requests.post(f"{API_URL}/predict-rul", json=payload).json()
            fail_res = requests.post(f"{API_URL}/predict-failure", json=payload).json()
            anom_res = requests.post(f"{API_URL}/anomaly", json=payload).json()
            rec_res = requests.post(f"{API_URL}/recommend-maintenance", json=payload).json()
            xai_res = requests.post(f"{API_URL}/explain", json=payload).json()
            
            with col1:
                st.subheader("Remaining Useful Life")
                st.metric(label="RUL (Cycles)", value=f"{rul_res['predicted_rul']:.1f}")
                
            with col2:
                st.subheader("Failure Probability")
                prob_pct = fail_res['failure_probability'] * 100
                st.metric(label="Probability", value=f"{prob_pct:.1f}%")
                if fail_res['failure_imminent']:
                    st.error("⚠️ Failure Imminent")
                    
            with col3:
                st.subheader("Anomaly Detection")
                if anom_res['is_anomaly']:
                    st.error("🚨 Anomaly Detected")
                else:
                    st.success("✅ Nominal")
                    
            st.divider()
            
            st.subheader("Maintenance Recommendation")
            if rec_res['urgency'] == "HIGH":
                st.error(f"**Action:** {rec_res['action']} \n\n **Reason:** {rec_res['reason']}")
            elif rec_res['urgency'] == "MEDIUM":
                st.warning(f"**Action:** {rec_res['action']} \n\n **Reason:** {rec_res['reason']}")
            else:
                st.success(f"**Action:** {rec_res['action']} \n\n **Reason:** {rec_res['reason']}")
                
            st.divider()
            
            st.subheader("Explainable AI (SHAP Feature Importance)")
            importance = xai_res['feature_importance']
            
            if importance:
                # Plotly Bar Chart
                fig = go.Figure(go.Bar(
                    x=list(importance.values())[::-1],
                    y=list(importance.keys())[::-1],
                    orientation='h',
                    marker_color=['red' if v > 0 else 'blue' for v in list(importance.values())[::-1]]
                ))
                fig.update_layout(title="Impact on Failure Probability (Red = Increases risk, Blue = Decreases risk)")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Failed to connect to AI Backend: {e}")
else:
    st.info("Adjust sensor telemetry in the sidebar and click 'Run AI Diagnostics'.")
