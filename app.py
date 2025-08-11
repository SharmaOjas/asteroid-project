# FILE: app.py

import streamlit as st
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import plotly.express as px

# ----------------------- App Config -----------------------
st.set_page_config(page_title="🌌 NASA Asteroid Dashboard", layout="wide")
st.title("🚀 Integrated NASA Asteroid Threat Analysis")

# ----------------------- API Config -----------------------
API_KEY = st.secrets["NASA_API_KEY"]
BASE_URL = "https://api.nasa.gov/neo/rest/v1/feed"

# ----------------------- Session State Init -----------------------
if 'model' not in st.session_state:
    st.session_state.model = None
if 'detailed_data' not in st.session_state:
    st.session_state.detailed_data = None
if 'daily_counts' not in st.session_state:
    st.session_state.daily_counts = None

# ----------------------- Data Fetch Function -----------------------
@st.cache_data(show_spinner="🚀 Fetching NEO data from NASA...")
def fetch_neo_data(start_date, end_date, api_key):
    params = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "api_key": api_key
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()

    daily_counts = []
    detailed_data = []

    for date_str, neos in data["near_earth_objects"].items():
        daily_counts.append({"date": date_str, "count": len(neos)})
        for neo in neos:
            est_diam = neo["estimated_diameter"]["kilometers"]
            approach = neo["close_approach_data"][0]
            detailed_data.append({
                "name": neo["name"],
                "date": date_str,
                "speed_km_h": float(approach["relative_velocity"]["kilometers_per_hour"]),
                "miss_distance_km": float(approach["miss_distance"]["kilometers"]),
                "diameter_min_km": est_diam["estimated_diameter_min"],
                "diameter_max_km": est_diam["estimated_diameter_max"],
                "is_hazardous": neo["is_potentially_hazardous_asteroid"]
            })
    return pd.DataFrame(daily_counts), pd.DataFrame(detailed_data)

# ----------------------- Model Training -----------------------
def train_model(data):
    if data is None or len(data) < 10:
        st.warning("⚠️ Insufficient data to train model.")
        return None

    X = data[["speed_km_h", "miss_distance_km", "diameter_min_km", "diameter_max_km"]]
    y = data["is_hazardous"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    st.success(f"✅ Model trained — Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}")
    return model

# ----------------------- Forecast Function -----------------------


# ----------------------- Sidebar -----------------------
with st.sidebar:
    st.header("🔧 Controls")
    today = datetime.today()
    start_date = st.date_input("Start Date", today - timedelta(days=7))
    end_date = st.date_input("End Date", today)
    if start_date > end_date:
        st.error("❌ Invalid date range.")
        st.stop()

    if st.button("🚀 Fetch NASA Data"):
        daily_df, detailed_df = fetch_neo_data(start_date, end_date, API_KEY)
        st.session_state.daily_counts = daily_df
        st.session_state.detailed_data = detailed_df
        st.session_state.model = None

    if st.session_state.detailed_data is not None:
        if st.button("🤖 Train Prediction Model"):
            st.session_state.model = train_model(st.session_state.detailed_data)

    page = st.radio("📄 Select Page", ["Dashboard Overview", "Asteroid Analysis", "Hazard Prediction", "Forecast", "Weekly Report"])

# ----------------------- Pages -----------------------

# Dashboard Overview
if page == "Dashboard Overview":
    st.subheader("🌌 NASA NEO Dashboard")
    if st.session_state.daily_counts is not None:
        df = st.session_state.detailed_data
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Asteroids", st.session_state.daily_counts["count"].sum())
        col2.metric("Hazardous Count", df["is_hazardous"].sum())
        col3.metric("Avg Speed", f"{df['speed_km_h'].mean():,.0f} km/h")

        st.line_chart(st.session_state.daily_counts.set_index("date"))
        fig, ax = plt.subplots()
        sns.countplot(x="is_hazardous", data=df, ax=ax)
        st.pyplot(fig)

# Asteroid Analysis
elif page == "Asteroid Analysis":
    st.subheader("🔭 Detailed Asteroid Data")
    if st.session_state.detailed_data is not None:
        df = st.session_state.detailed_data.copy()
        df["avg_diameter"] = (df["diameter_min_km"] + df["diameter_max_km"]) / 2
        st.dataframe(df)

        fig, ax = plt.subplots()
        sns.histplot(df["avg_diameter"], bins=20, kde=True, ax=ax)
        ax.set_xlabel("Avg Diameter (km)")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.scatterplot(x="miss_distance_km", y="speed_km_h", hue="is_hazardous", data=df, ax=ax)
        ax.set_xscale('log')
        st.pyplot(fig)

# Hazard Prediction
elif page == "Hazard Prediction":
    st.subheader("⚠️ Predict Hazard")
    if st.session_state.model is not None:
        if st.button("🔎 Predict Current Data"):
            features = st.session_state.detailed_data[["speed_km_h", "miss_distance_km", "diameter_min_km", "diameter_max_km"]]
            pred = st.session_state.model.predict(features)
            results = st.session_state.detailed_data.copy()
            results["predicted"] = pred
            st.dataframe(results[["name", "is_hazardous", "predicted"]])
            st.success(f"🎯 Prediction Accuracy: {accuracy_score(results['is_hazardous'], pred):.2%}")

        with st.form("manual_pred"):
            st.write("📋 Enter NEO details manually:")
            s = st.slider("Speed (km/h)", 0, 200000, 50000)
            d = st.slider("Miss Distance (km)", 1000, 100000000, 1000000)
            d1 = st.slider("Min Diameter (km)", 0.001, 5.0, 0.1)
            d2 = st.slider("Max Diameter (km)", 0.001, 10.0, 0.3)
            if st.form_submit_button("Predict"):
                sample = pd.DataFrame([[s, d, d1, d2]], columns=["speed_km_h", "miss_distance_km", "diameter_min_km", "diameter_max_km"])
                out = st.session_state.model.predict(sample)[0]
                st.success("🚨 Predicted: Hazardous" if out else "✅ Predicted: Not Hazardous")

# Forecast
elif page == "Forecast":

    st.subheader("🕰️ NEO Forecast (Time Series)")
    st.info("📈 Forecast feature is coming soon! Stay tuned 🚀")

# Weekly Report
elif page == "Weekly Report":
    st.subheader("Weekly Asteroid Hazard Summary")
    df = st.session_state.detailed_data
    if df is not None:
        df["date"] = pd.to_datetime(df["date"])
        summary = df.groupby("date").agg({
            "name": "count",
            "is_hazardous": "sum",
            "speed_km_h": "mean",
            "miss_distance_km": "min"
        }).reset_index()

        total_asteroids = summary["name"].sum()
        hazardous_count = summary["is_hazardous"].sum()
        closest_miss = summary["miss_distance_km"].min()
        fastest_speed = df["speed_km_h"].max()

        st.markdown(f"""
        ## Weekly Earth vs Asteroids Report
        - **{int(total_asteroids)}** asteroids flew near Earth this week  
        - **{int(hazardous_count)}** were *potentially hazardous*  
        - Closest approach was just **{closest_miss:,.0f} km**  
        - Fastest asteroid zipped by at **{fastest_speed:,.0f} km/h**
        """)

        st.markdown("---")
        st.markdown("### 📊 Daily Breakdown")
        summary.columns = ["Date", "Total Asteroids", "Hazardous Count", "Avg Speed (km/h)", "Closest Miss (km)"]
        st.dataframe(summary)

# ----------------------- Footer -----------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Built with 💫 [NASA NEO API](https://api.nasa.gov/) + Scikit-learn")
