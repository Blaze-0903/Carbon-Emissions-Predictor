import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Global CO₂ Emissions Predictor 🌍",
    page_icon="🌿",
    layout="centered"
)

# --- Load Model ---
try:
    model = joblib.load("xgboost_co2_predictor_model.joblib")
    st.success("✅ Model loaded successfully.")
except FileNotFoundError:
    st.error("🚫 Model file not found. Ensure 'xgboost_co2_predictor_model.joblib' is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- Load Data ---
try:
    df = pd.read_csv("owid-co2-data.csv")
    st.success("✅ Data file loaded successfully.")
except FileNotFoundError:
    st.error("🚫 Data file not found. Ensure 'owid-co2-data.csv' is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# --- Data Preprocessing ---
world_co2 = df[df['country'] == 'World'][['year', 'co2', 'population', 'gdp']].copy()
world_co2 = world_co2.sort_values(by='year').reset_index(drop=True)

# Interpolate missing values
world_co2['co2'] = world_co2['co2'].interpolate(method='linear', limit_direction='both')
world_co2['population'] = world_co2['population'].interpolate(method='linear', limit_direction='both')
world_co2['gdp'] = world_co2['gdp'].interpolate(method='linear', limit_direction='both')

# Feature engineering
world_co2['co2_diff'] = world_co2['co2'].diff()
for i in range(1, 4):
    world_co2[f'co2_diff_lag_{i}'] = world_co2['co2_diff'].shift(i)
world_co2['population_diff'] = world_co2['population'].diff()
world_co2['gdp_diff'] = world_co2['gdp'].diff()

poly = PolynomialFeatures(degree=2, include_bias=False)
world_co2['trend'] = poly.fit_transform(world_co2[['year']])[:, 1]

# Drop NA rows
world_co2.dropna(inplace=True)
world_co2 = world_co2.reset_index(drop=True)

# --- Feature List ---
features = [
    'year',
    'co2_diff_lag_1', 'co2_diff_lag_2', 'co2_diff_lag_3',
    'population_diff', 'gdp_diff',
    'population', 'gdp',
    'trend'
]

# --- Fit Scaler ---
scaler = StandardScaler()
scaler.fit(world_co2[features])

# --- Streamlit UI ---
st.title("🌱 Global CO₂ Emissions Prediction App")
st.markdown("Predict future global CO₂ emissions using historical data and machine learning.")

# Sidebar input
st.sidebar.header("📊 Input Configuration")
min_pred_year = int(world_co2['year'].max()) + 1
max_pred_year = min_pred_year + 10
year_to_predict = st.sidebar.slider("Select Prediction Year", min_value=min_pred_year, max_value=max_pred_year, value=min_pred_year)

# --- Prediction Logic ---
st.header(f"📈 Prediction for {year_to_predict}")

# Get last known values
last_row = world_co2.iloc[-1]
last_year = last_row['year']
last_co2 = last_row['co2']
last_co2_diff = world_co2['co2_diff'].tail(3).tolist()
last_population = last_row['population']
last_gdp = last_row['gdp']

# Estimate growth rates
recent_data = world_co2[world_co2['year'] > world_co2['year'].max() - 5]
avg_pop_growth = (recent_data['population'].diff() / recent_data['population'].shift(1)).mean()
avg_gdp_growth = (recent_data['gdp'].diff() / recent_data['gdp'].shift(1)).mean()

if np.isnan(avg_pop_growth): avg_pop_growth = 0.01
if np.isnan(avg_gdp_growth): avg_gdp_growth = 0.02

# Project values
years_ahead = year_to_predict - last_year
projected_population = last_population * (1 + avg_pop_growth) ** years_ahead
projected_gdp = last_gdp * (1 + avg_gdp_growth) ** years_ahead

# Prepare input
input_data = {
    'year': year_to_predict,
    'co2_diff_lag_1': last_co2_diff[-1] if len(last_co2_diff) >= 1 else 0,
    'co2_diff_lag_2': last_co2_diff[-2] if len(last_co2_diff) >= 2 else 0,
    'co2_diff_lag_3': last_co2_diff[-3] if len(last_co2_diff) >= 3 else 0,
    'population_diff': projected_population - last_population,
    'gdp_diff': projected_gdp - last_gdp,
    'population': projected_population,
    'gdp': projected_gdp,
    'trend': poly.transform([[year_to_predict]])[:, 1][0]
}

input_df = pd.DataFrame([input_data], columns=features)

# --- Prediction and Plot ---
try:
    X_scaled = scaler.transform(input_df)
    predicted_diff = model.predict(X_scaled)[0]
    predicted_co2 = last_co2 + predicted_diff

    st.success(f"📊 Predicted CO₂ emissions for *{year_to_predict}: **{predicted_co2:.2f} MtCO₂*")

    # Plotting
    st.subheader("📉 Emissions Trend")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(world_co2['year'], world_co2['co2'], label="Historical Emissions", color='blue')
    ax.scatter(year_to_predict, predicted_co2, color='red', s=100, label="Prediction")
    ax.plot([last_year, year_to_predict], [last_co2, predicted_co2], linestyle='--', color='red')

    ax.set_xlabel("Year")
    ax.set_ylabel("Global CO₂ Emissions (MtCO₂)")
    ax.set_title("Global CO₂ Emissions: Historical vs Predicted")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

# --- Footer ---
st.markdown("---")
st.caption("Made with 💚 using Streamlit | Powered by Machine Learning")