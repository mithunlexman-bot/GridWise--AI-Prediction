import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="GridWise AI", page_icon="⚡", layout="wide")

REQUIRED_COLUMNS = [
    "timestamp", "building", "room", "temperature_c", "humidity_pct",
    "occupancy", "hvac_kw", "lighting_kw", "equipment_kw", "energy_kwh"
]
NUMERIC_COLUMNS = [
    "temperature_c", "humidity_pct", "occupancy", "hvac_kw",
    "lighting_kw", "equipment_kw", "energy_kwh"
]
FEATURES = [
    "hour", "day_of_week", "month", "temperature_c", "humidity_pct",
    "occupancy", "hvac_kw", "lighting_kw", "equipment_kw"
]

def make_sample_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h")
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    occupancy = np.where((hour >= 8) & (hour <= 18) & (weekday < 5),
                         rng.integers(12, 65, n), rng.integers(0, 12, n))
    temperature = 24 + 6*np.sin(2*np.pi*(hour-6)/24) + rng.normal(0, 1.8, n)
    humidity = np.clip(rng.normal(58, 9, n), 25, 90)
    hvac = np.clip(1.0 + 0.075*np.maximum(temperature-22, 0) +
                   0.012*occupancy + rng.normal(0, .25, n), .2, 8)
    lighting = np.where(occupancy > 8, rng.uniform(.5, 1.8, n),
                        rng.uniform(.05, .35, n))
    equipment = np.clip(.35 + .025*occupancy + rng.normal(0, .2, n), .1, 3.5)
    energy = hvac + lighting + equipment + rng.normal(0, .25, n)
    energy = np.clip(energy, .1, None)
    # A few unusual records make the anomaly module demonstrable.
    anomaly_idx = rng.choice(n, size=max(5, n//100), replace=False)
    energy[anomaly_idx] *= rng.uniform(1.8, 2.5, len(anomaly_idx))
    buildings = rng.choice(["Block A", "Block B", "Block C"], n)
    rooms = rng.choice(["Lab 101", "Classroom 202", "Office 303", "Library"], n)
    return pd.DataFrame({
        "timestamp": timestamps, "building": buildings, "room": rooms,
        "temperature_c": temperature.round(2), "humidity_pct": humidity.round(2),
        "occupancy": occupancy, "hvac_kw": hvac.round(2),
        "lighting_kw": lighting.round(2), "equipment_kw": equipment.round(2),
        "energy_kwh": energy.round(2)
    })

def prepare_data(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    data = df[REQUIRED_COLUMNS].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["timestamp"] + NUMERIC_COLUMNS)
    if len(data) < 20:
        raise ValueError("At least 20 valid rows are required for model training.")
    data = data.sort_values("timestamp").reset_index(drop=True)
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month
    return data

def train_prediction_model(data):
    split = max(1, int(len(data) * 0.8))
    if split >= len(data):
        split = len(data) - 1
    train, test = data.iloc[:split], data.iloc[split:]
    model = RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=2)
    model.fit(train[FEATURES], train["energy_kwh"])
    predictions = model.predict(test[FEATURES])
    mae = mean_absolute_error(test["energy_kwh"], predictions)
    rmse = np.sqrt(mean_squared_error(test["energy_kwh"], predictions))
    r2 = r2_score(test["energy_kwh"], predictions) if len(test) > 1 else float("nan")
    return model, {"MAE": mae, "RMSE": rmse, "R²": r2}, test, predictions

def get_recommendations(data):
    recs = []
    avg_occ = data["occupancy"].mean()
    avg_hvac = data["hvac_kw"].mean()
    avg_light = data["lighting_kw"].mean()
    if avg_occ < 10 and avg_hvac > 1.5:
        recs.append("Review HVAC operating schedules during low-occupancy periods.")
    if avg_light > 1:
        recs.append("Review lighting schedules and consider occupancy-based lighting controls.")
    if data["energy_kwh"].tail(max(5, len(data)//10)).mean() > data["energy_kwh"].mean() * 1.15:
        recs.append("Recent energy use is above the dataset average; inspect operating hours and equipment loads.")
    if avg_hvac >= avg_light:
        recs.append("HVAC is a major average load. Check temperature setpoints, filters, and maintenance schedules.")
    if not recs:
        recs.append("Continue monitoring energy trends and compare usage across buildings and rooms.")
    return recs

def assistant_reply(question, data, metrics, anomaly_count):
    q = question.lower()
    avg = data["energy_kwh"].mean()
    peak = data["energy_kwh"].max()
    if any(word in q for word in ["average", "mean", "usage", "consumption"]):
        return f"The average energy consumption in the current dataset is {avg:.2f} kWh per record."
    if any(word in q for word in ["peak", "highest", "maximum"]):
        return f"The highest recorded energy consumption is {peak:.2f} kWh per record."
    if "anomal" in q or "unusual" in q:
        return f"The anomaly detector flagged {anomaly_count} records for review. These are unusual patterns, not proof of equipment failure."
    if "saving" in q or "save" in q or "reduce" in q:
        return "Start by reviewing HVAC schedules during low occupancy, checking lighting schedules, and comparing equipment loads. Use the What-if Simulator to estimate scenarios."
    if "model" in q or "accuracy" in q or "error" in q:
        return f"On the chronological holdout data, MAE is {metrics['MAE']:.2f} kWh and RMSE is {metrics['RMSE']:.2f} kWh. These metrics depend on data quality and should not be treated as guaranteed real-world accuracy."
    return "I can explain average consumption, peak usage, anomaly results, model evaluation, and possible energy-saving actions. Ask a question about one of these topics."

st.title("⚡ GridWise AI")
st.caption("Intelligent Energy Optimization and Smart Building Management")
st.info("Decision-support demonstration: predictions and anomaly flags help with investigation; the app does not control electrical equipment.")

with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose data", ["Sample dataset", "Upload CSV"])
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload building energy CSV", type=["csv"])
    st.markdown("---")
    page = st.radio("Navigation", [
        "Overview Dashboard", "Energy Prediction", "Anomaly Detection",
        "What-if Simulator", "Optimization Recommendations", "AI Energy Assistant", "Data and Guide"
    ])

try:
    raw = pd.read_csv(uploaded) if source == "Upload CSV" and uploaded is not None else make_sample_data()
    if source == "Upload CSV" and uploaded is None:
        st.warning("Upload a CSV file to use your own building data. Sample data is shown for now.")
    data = prepare_data(raw)
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

model, metrics, test, predictions = train_prediction_model(data)
anomaly_features = data[["energy_kwh"] + FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
detector = IsolationForest(contamination=0.03, random_state=42)
data["anomaly_label"] = detector.fit_predict(anomaly_features)
data["anomaly_status"] = np.where(data["anomaly_label"] == -1, "Anomaly", "Normal")
anomaly_count = int((data["anomaly_status"] == "Anomaly").sum())

if page == "Overview Dashboard":
    st.subheader("Overview Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total energy", f"{data.energy_kwh.sum():,.1f} kWh")
    c2.metric("Average per record", f"{data.energy_kwh.mean():.2f} kWh")
    c3.metric("Peak per record", f"{data.energy_kwh.max():.2f} kWh")
    c4.metric("Anomalies flagged", f"{anomaly_count}")
    trend = data.groupby(data["timestamp"].dt.date, as_index=False)["energy_kwh"].sum()
    trend.columns = ["date", "energy_kwh"]
    st.plotly_chart(px.line(trend, x="date", y="energy_kwh", title="Daily energy consumption (kWh)"), use_container_width=True)
    by_building = data.groupby("building", as_index=False)["energy_kwh"].mean()
    st.plotly_chart(px.bar(by_building, x="building", y="energy_kwh", title="Average energy by building"), use_container_width=True)

elif page == "Energy Prediction":
    st.subheader("Energy Prediction – Random Forest Regression")
    st.write("Enter a building scenario to estimate energy consumption for one interval.")
    c1, c2, c3 = st.columns(3)
    with c1:
        hour = st.slider("Hour of day", 0, 23, 10)
        day = st.selectbox("Day of week", list(range(7)), index=1, format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x])
        month = st.slider("Month", 1, 12, 6)
    with c2:
        temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 55.0)
        occupancy = st.number_input("Occupancy", 0, 500, 25)
    with c3:
        hvac = st.number_input("HVAC load (kW)", 0.0, 100.0, 2.0)
        lighting = st.number_input("Lighting load (kW)", 0.0, 100.0, 0.8)
        equipment = st.number_input("Equipment load (kW)", 0.0, 100.0, 1.2)
    scenario = pd.DataFrame([[hour, day, month, temp, humidity, occupancy, hvac, lighting, equipment]], columns=FEATURES)
    estimate = float(model.predict(scenario)[0])
    st.metric("Estimated energy consumption", f"{estimate:.2f} kWh")
    st.caption("This estimate is based on the selected dataset and is not a guaranteed measurement.")
    st.markdown("**Chronological holdout evaluation**")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE']:.3f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.3f}")
    m3.metric("R²", f"{metrics['R²']:.3f}" if np.isfinite(metrics["R²"]) else "N/A")
    comparison = pd.DataFrame({"Actual": test["energy_kwh"].values, "Predicted": predictions})
    st.plotly_chart(px.line(comparison, title="Actual vs predicted energy (holdout records)"), use_container_width=True)

elif page == "Anomaly Detection":
    st.subheader("Anomaly Detection – Isolation Forest")
    st.write(f"Flagged **{anomaly_count}** of **{len(data)}** records for review.")
    st.caption("An anomaly is an unusual pattern, not proof of a fault or equipment failure.")
    st.dataframe(data.loc[data.anomaly_status == "Anomaly", REQUIRED_COLUMNS + ["anomaly_status"]], use_container_width=True)
    st.plotly_chart(px.scatter(data, x="timestamp", y="energy_kwh", color="anomaly_status", title="Energy records and anomaly flags"), use_container_width=True)

elif page == "What-if Simulator":
    st.subheader("What-if Energy Simulator")
    hvac_reduction = st.slider("Assumed HVAC reduction (%)", 0, 50, 10)
    light_reduction = st.slider("Assumed lighting reduction (%)", 0, 50, 10)
    tariff = st.number_input("Electricity tariff (₹ per kWh)", min_value=0.0, value=8.0, step=0.5)
    hvac_energy = data["hvac_kw"].sum()
    light_energy = data["lighting_kw"].sum()
    saved = hvac_energy * hvac_reduction/100 + light_energy * light_reduction/100
    cost_saved = saved * tariff
    a, b, c = st.columns(3)
    a.metric("Estimated energy saving", f"{saved:.2f} kWh")
    b.metric("Estimated cost saving", f"₹{cost_saved:,.2f}")
    c.metric("Estimated current energy", f"{data.energy_kwh.sum():,.2f} kWh")
    st.caption("Scenario estimate only. It assumes the supplied HVAC and lighting values represent comparable energy totals. Actual savings depend on occupancy, weather, equipment efficiency, schedules, and tariff.")

elif page == "Optimization Recommendations":
    st.subheader("Energy Optimization Recommendations")
    for i, rec in enumerate(get_recommendations(data), 1):
        st.markdown(f"**{i}.** {rec}")
    st.markdown("Recommendations are rule-based suggestions. Validate operational changes with building staff.")

elif page == "AI Energy Assistant":
    st.subheader("AI Energy Assistant")
    st.write("This starter assistant uses a local rule-based fallback and works without an API key.")
    question = st.text_input("Ask about your energy data", placeholder="What is the average energy consumption?")
    if st.button("Get answer"):
        if question.strip():
            st.success(assistant_reply(question, data, metrics, anomaly_count))
        else:
            st.warning("Please enter a question.")

elif page == "Data and Guide":
    st.subheader("Data and Guide")
    st.markdown("Your CSV must contain these exact columns:")
    st.code(",".join(REQUIRED_COLUMNS), language="text")
    st.write("Each row should represent one building/room reading at a consistent time interval.")
    st.dataframe(data.head(20), use_container_width=True)
    st.download_button("Download current data as CSV", data=data[REQUIRED_COLUMNS].to_csv(index=False).encode("utf-8"),
                       file_name="gridwise_energy_data.csv", mime="text/csv")
    st.markdown("**Important:** Synthetic data is for demonstration and testing. Use measured building data to assess real performance.")
