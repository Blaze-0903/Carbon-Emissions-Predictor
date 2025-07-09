import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import base64
import os
from datetime import datetime

# --- Helper function to get base64 image ---
def get_base64_image(image_path):
    """Converts an image file to a base64 string for embedding in CSS."""
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"❌ Background image not found at: {image_path}. Please ensure the 'static' folder and '1.png' are in the correct location.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading background image: {e}")
        return None

# --- Page Configuration ---
st.set_page_config(
    page_title="Carbon Emission Predictor 🌍",
    page_icon="🌿",
    layout="centered"
)

# --- Path to your background image ---
background_image_path = os.path.join('static', '1.png')
bg_image_base64 = get_base64_image(background_image_path)

# --- Custom CSS for Background and Translucency ---
if bg_image_base64: # Only apply background if image loaded successfully
    st.markdown(
        f"""
        <style>
        body {{
            background-image: url("data:image/png;base64,{bg_image_base64}"); /* Embed base64 image */
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        /* Target the main content area for translucency */
        .stApp {{
            background-color: rgba(20, 0, 0, 0.7); /* Darker, slightly reddish black with 70% opacity */
            border-radius: 10px;
            padding: 20px;
            margin: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5); /* More prominent shadow for dark background */
        }}
        /* Adjusting sidebar for consistency and translucency */
        .st-emotion-cache-vk3ypu {{ /* This class targets the sidebar container */
            background-color: rgba(20, 0, 0, 0.8); /* Slightly less translucent and darker */
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5); /* More prominent shadow */
        }}
        /* General container adjustments for rounded corners and subtle shadow */
        .block-container {{
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25); /* Lighter shadow for internal blocks */
            padding: 10px;
            margin-bottom: 10px;
        }}
        /* Adjusting for potential other main content containers */
        .main .block-container {{
            background-color: rgba(20, 0, 0, 0.7); /* Ensure main content blocks also have translucency */
        }}
        /* Adjust text color for better readability on dark background */
        h1, h2, h3, h4, h5, h6, .stMarkdown, label, .st-emotion-cache-10q7q25 {{
            color: #ffcccb !important; /* Light red/pink for headings and important text */
        }}
        /* Specific adjustments for selectbox and slider labels */
        .st-emotion-cache-1wq7w1s p, .st-emotion-cache-1wq7w1s label {{
            color: #ffcccb !important;
        }}
        /* Adjust success message color for contrast */
        .stSuccess {{
            background-color: rgba(100, 0, 0, 0.5); /* Darker red for success background */
            color: #ffcccb !important; /* Light red for success text */
            border-left: 5px solid #ff4d4d; /* More prominent red border */
        }}
        /* Adjust error message color for contrast */
        .stError {{
            background-color: rgba(100, 0, 0, 0.5); /* Darker red for error background */
            color: #ffcccb !important; /* Light red for error text */
            border-left: 5px solid #ff4d4d; /* More prominent red border */
        }}
        /* Adjust warning message color for contrast */
        .stWarning {{
            background-color: rgba(100, 0, 0, 0.5); /* Darker red for warning background */
            color: #ffcccb !important; /* Light red for warning text */
            border-left: 5px solid #ff4d4d; /* More prominent red border */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Background image could not be loaded. Displaying app without background image.")


st.title("🌱 Carbon Emission Prediction App")
st.markdown("Choose a prediction mode from the sidebar.")

# Get current year
current_year = datetime.now().year

# --- Sidebar for Mode Selection ---
st.sidebar.header("🔍 Prediction Mode")
mode = st.sidebar.radio("Select Mode", ["Global CO₂ Prediction", "Country-wise GHG Prediction"])

# --- GLOBAL CO₂ PREDICTION ---
if mode == "Global CO₂ Prediction":
    st.header("🌍 Global CO₂ Emissions Prediction")

    try:
        global_model = joblib.load("xgboost_co2_predictor_model.joblib")
    except Exception as e:
        st.error(f"❌ Could not load global model: {e}")
        st.stop()

    try:
        global_df = pd.read_csv("owid-co2-data.csv")
    except Exception as e:
        st.error(f"❌ Could not load CO₂ data: {e}")
        st.stop()

    world_co2 = global_df[global_df['country'] == 'World'][['year', 'co2', 'population', 'gdp']].copy()
    world_co2.sort_values(by='year', inplace=True)
    world_co2.reset_index(drop=True, inplace=True)

    world_co2['co2'] = world_co2['co2'].interpolate(method='linear', limit_direction='both')
    world_co2['population'] = world_co2['population'].interpolate(method='linear', limit_direction='both')
    world_co2['gdp'] = world_co2['gdp'].interpolate(method='linear', limit_direction='both')

    world_co2['co2_diff'] = world_co2['co2'].diff()
    for i in range(1, 4):
        world_co2[f'co2_diff_lag_{i}'] = world_co2['co2_diff'].shift(i)
    world_co2['population_diff'] = world_co2['population'].diff()
    world_co2['gdp_diff'] = world_co2['gdp'].diff()

    poly = PolynomialFeatures(degree=2, include_bias=False)
    world_co2['trend'] = poly.fit_transform(world_co2[['year']])[:, 1]

    world_co2.dropna(inplace=True)
    world_co2.reset_index(drop=True, inplace=True)

    features = [
        'year',
        'co2_diff_lag_1', 'co2_diff_lag_2', 'co2_diff_lag_3',
        'population_diff', 'gdp_diff',
        'population', 'gdp',
        'trend'
    ]

    scaler = StandardScaler()
    scaler.fit(world_co2[features])

    # Get the last historical data point for global prediction
    last_historical_row_global = world_co2.iloc[-1]
    max_historical_year_global = int(last_historical_row_global['year'])
    last_actual_co2_global = last_historical_row_global['co2']
    last_co2_diff_values_global = world_co2['co2_diff'].tail(3).tolist()
    last_population_global = last_historical_row_global['population']
    last_gdp_global = last_historical_row_global['gdp']

    st.sidebar.markdown(f"---")
    st.sidebar.subheader("Future Prediction Range (Global)")
    start_pred_year_global = st.sidebar.slider(
        "📆 Start Prediction Year",
        min_value=max_historical_year_global + 1,
        max_value=current_year + 1, # Allow starting from current year or next historical year
        value=current_year # Default to current year
    )
    num_years_to_predict_global = st.sidebar.slider(
        "🔢 Number of Years to Predict Ahead",
        min_value=1,
        max_value=20, # Predict up to 20 years ahead
        value=5 # Default to 5 years
    )
    end_pred_year_global = start_pred_year_global + num_years_to_predict_global - 1
    st.sidebar.info(f"Predicting from {start_pred_year_global} to {end_pred_year_global}")

    st.subheader(f"📈 Predicted CO₂ Emissions for World")
    global_progress_bar = st.progress(0)
    global_status_text = st.empty()

    forecasted_years_global = []
    forecasted_co2_global = []

    # Calculate average growth rates for population and GDP for extrapolation
    recent_data_global = world_co2[world_co2['year'] > max_historical_year_global - 5].copy()
    avg_pop_growth_global = (recent_data_global['population'].diff() / recent_data_global['population'].shift(1)).mean()
    avg_gdp_growth_global = (recent_data_global['gdp'].diff() / recent_data_global['gdp'].shift(1)).mean()

    if np.isnan(avg_pop_growth_global): avg_pop_growth_global = 0.01
    if np.isnan(avg_gdp_growth_global): avg_gdp_growth_global = 0.02

    current_co2_for_forecast = last_actual_co2_global
    current_co2_diff_lags = last_co2_diff_values_global[:] # Copy for mutable list

    for i, year_to_predict_single in enumerate(range(start_pred_year_global, end_pred_year_global + 1)):
        global_status_text.text(f"Predicting for year: {year_to_predict_single}...")
        global_progress_bar.progress((i + 1) / num_years_to_predict_global)

        years_ahead_from_last_hist = year_to_predict_single - max_historical_year_global
        projected_population_single = last_population_global * (1 + avg_pop_growth_global) ** years_ahead_from_last_hist
        projected_gdp_single = last_gdp_global * (1 + avg_gdp_growth_global) ** years_ahead_from_last_hist

        input_data_single = {
            'year': year_to_predict_single,
            'co2_diff_lag_1': current_co2_diff_lags[-1] if len(current_co2_diff_lags) >= 1 else 0,
            'co2_diff_lag_2': current_co2_diff_lags[-2] if len(current_co2_diff_lags) >= 2 else 0,
            'co2_diff_lag_3': current_co2_diff_lags[-3] if len(current_co2_diff_lags) >= 3 else 0,
            'population_diff': projected_population_single - (last_population_global * (1 + avg_pop_growth_global) ** (years_ahead_from_last_hist - 1) if years_ahead_from_last_hist > 0 else last_population_global),
            'gdp_diff': projected_gdp_single - (last_gdp_global * (1 + avg_gdp_growth_global) ** (years_ahead_from_last_hist - 1) if years_ahead_from_last_hist > 0 else last_gdp_global),
            'population': projected_population_single,
            'gdp': projected_gdp_single,
            'trend': poly.transform([[year_to_predict_single]])[:, 1][0]
        }
        # Special handling for first year's diff calculation if it's directly after historical data
        if year_to_predict_single == max_historical_year_global + 1:
            input_data_single['population_diff'] = projected_population_single - last_population_global
            input_data_single['gdp_diff'] = projected_gdp_single - last_gdp_global
        else:
            # For subsequent years, calculate diff based on previous projected values
            prev_projected_pop = last_population_global * (1 + avg_pop_growth_global) ** (years_ahead_from_last_hist - 1)
            prev_projected_gdp = last_gdp_global * (1 + avg_gdp_growth_global) ** (years_ahead_from_last_hist - 1)
            input_data_single['population_diff'] = projected_population_single - prev_projected_pop
            input_data_single['gdp_diff'] = projected_gdp_single - prev_projected_gdp


        input_df_single = pd.DataFrame([input_data_single], columns=features)

        try:
            X_scaled_single = scaler.transform(input_df_single)
            predicted_diff_single = global_model.predict(X_scaled_single)[0]

            current_co2_for_forecast += predicted_diff_single # Integrate the difference
            forecasted_years_global.append(year_to_predict_single)
            forecasted_co2_global.append(current_co2_for_forecast)

            # Update lagged differences for the next iteration
            current_co2_diff_lags.pop(0)
            current_co2_diff_lags.append(predicted_diff_single)

        except Exception as e:
            st.error(f"❌ Prediction failed for {year_to_predict_single}: {e}")
            break # Stop forecasting if an error occurs

    global_progress_bar.empty()
    global_status_text.empty()

    if forecasted_co2_global:
        st.success(f"📊 Forecasted Global CO₂ Emissions from {start_pred_year_global} to {end_pred_year_global} are shown below.")

        forecast_df_global = pd.DataFrame({
            'year': forecasted_years_global,
            'predicted_co2': forecasted_co2_global
        })
        st.dataframe(forecast_df_global)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(world_co2['year'], world_co2['co2'], label="Historical Emissions", color='blue')
        ax.plot(forecast_df_global['year'], forecast_df_global['predicted_co2'], marker='o', color='red', label="Predicted Emissions")
        ax.set_xlabel("Year")
        ax.set_ylabel("Global CO₂ Emissions (MtCO₂)")
        ax.set_title("Global CO₂ Emissions: Historical and Predicted")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("No global predictions were generated.")


# --- COUNTRY-WISE GHG PREDICTION ---
else:
    st.header("🌐 Country-wise GHG Emissions Prediction")

    try:
        country_model = joblib.load("ghg_model.pkl")
        country_scaler = joblib.load("ghg_scaler.pkl")
        country_df = pd.read_csv("preprocessed_data.csv")
    except Exception as e:
        st.error(f"❌ Could not load model/data: {e}")
        st.stop()

    country = st.sidebar.selectbox("🌍 Select Country", sorted(country_df['country'].unique()))

    # Get the last historical data for the selected country
    country_historical_data = country_df[country_df['country'] == country].copy()
    if country_historical_data.empty:
        st.error("🚫 No historical data available for the selected country.")
        st.stop()

    max_historical_year_country = int(country_historical_data['year'].max())
    last_historical_row_country = country_historical_data[country_historical_data['year'] == max_historical_year_country].iloc[0]

    st.sidebar.markdown(f"---")
    st.sidebar.subheader("Future Prediction Range (Country-wise)")
    start_pred_year_country = st.sidebar.slider(
        "📆 Start Prediction Year",
        min_value=max_historical_year_country + 1,
        max_value=current_year + 1, # Allow starting from current year or next historical year
        value=current_year # Default to current year
    )
    num_years_to_predict = st.sidebar.slider(
        "🔢 Number of Years to Predict Ahead",
        min_value=1,
        max_value=20, # Predict up to 20 years ahead
        value=5 # Default to 5 years
    )
    end_pred_year_country = start_pred_year_country + num_years_to_predict - 1
    st.sidebar.info(f"Predicting from {start_pred_year_country} to {end_pred_year_country}")


    # Define input features for the country model (excluding 'total_ghg' as it's the target)
    input_features_country = [
        'population', 'gdp',
        'total_ghg_lag1', 'total_ghg_lag2', 'total_ghg_lag3',
        'co2_lag1', 'methane_lag1', 'nitrous_oxide_lag1'
    ]

    # Initialize lists to store forecasted data
    forecasted_years = []
    forecasted_ghg = []

    # Get the last known values for iterative forecasting
    # These will be updated in the loop
    lagged_total_ghg = [last_historical_row_country[f'total_ghg_lag{i}'] for i in range(1, 4)]
    lagged_co2 = [last_historical_row_country[f'co2_lag{i}'] for i in range(1, 4)]
    lagged_methane = [last_historical_row_country[f'methane_lag{i}'] for i in range(1, 4)]
    lagged_nitrous_oxide = [last_historical_row_country[f'nitrous_oxide_lag{i}'] for i in range(1, 4)]

    # Calculate average growth rates for population and GDP for extrapolation
    recent_country_data = country_historical_data[country_historical_data['year'] > max_historical_year_country - 5].copy()
    avg_pop_growth_rate_country = (recent_country_data['population'].diff() / recent_country_data['population'].shift(1)).mean()
    avg_gdp_growth_rate_country = (recent_country_data['gdp'].diff() / recent_country_data['gdp'].shift(1)).mean()

    if np.isnan(avg_pop_growth_rate_country): avg_pop_growth_rate_country = 0.01
    if np.isnan(avg_gdp_growth_rate_country): avg_gdp_growth_rate_country = 0.02


    st.subheader(f"📈 Predicted GHG Emissions for {country}")
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, year_to_predict_single in enumerate(range(start_pred_year_country, end_pred_year_country + 1)):
        status_text.text(f"Predicting for year: {year_to_predict_single}...")
        progress_bar.progress((i + 1) / num_years_to_predict)

        # Extrapolate population and GDP for the current prediction year
        years_ahead_from_last_hist = year_to_predict_single - max_historical_year_country
        projected_population_single = last_historical_row_country['population'] * (1 + avg_pop_growth_rate_country) ** years_ahead_from_last_hist
        projected_gdp_single = last_historical_row_country['gdp'] * (1 + avg_gdp_growth_rate_country) ** years_ahead_from_last_hist

        # Prepare input data for the current prediction year
        input_data_single = {
            'population': projected_population_single,
            'gdp': projected_gdp_single,
            'total_ghg_lag1': lagged_total_ghg[-1],
            'total_ghg_lag2': lagged_total_ghg[-2],
            'total_ghg_lag3': lagged_total_ghg[-3],
            'co2_lag1': lagged_co2[-1],
            'methane_lag1': lagged_methane[-1],
            'nitrous_oxide_lag1': lagged_nitrous_oxide[-1]
        }

        input_df_single = pd.DataFrame([input_data_single], columns=input_features_country)

        try:
            X_scaled_single = country_scaler.transform(input_df_single)
            predicted_ghg_single = country_model.predict(X_scaled_single)[0]

            forecasted_years.append(year_to_predict_single)
            forecasted_ghg.append(predicted_ghg_single)

            # Update lagged values for the next iteration
            lagged_total_ghg.pop(0)
            lagged_total_ghg.append(predicted_ghg_single)

            # Simplification: For co2, methane, nitrous_oxide,
            # assume their growth is proportional to total_ghg growth based on last historical ratios.
            if last_historical_row_country['total_ghg'] != 0:
                co2_ratio = last_historical_row_country['co2'] / last_historical_row_country['total_ghg']
                methane_ratio = last_historical_row_country['methane'] / last_historical_row_country['total_ghg']
                nitrous_oxide_ratio = last_historical_row_country['nitrous_oxide'] / last_historical_row_country['total_ghg']
            else: # Fallback if historical total_ghg is zero
                co2_ratio = 0.7
                methane_ratio = 0.15
                nitrous_oxide_ratio = 0.05

            lagged_co2.pop(0)
            lagged_co2.append(predicted_ghg_single * co2_ratio)
            lagged_methane.pop(0)
            lagged_methane.append(predicted_ghg_single * methane_ratio)
            lagged_nitrous_oxide.pop(0)
            lagged_nitrous_oxide.append(predicted_ghg_single * nitrous_oxide_ratio)

        except Exception as e:
            st.error(f"❌ Prediction failed for {year_to_predict_single}: {e}")
            break # Stop forecasting if an error occurs

    progress_bar.empty()
    status_text.empty()

    if forecasted_ghg:
        st.success(f"📊 Forecasted GHG Emissions for {country} from {start_pred_year_country} to {end_pred_year_country} are shown below.")

        forecast_df = pd.DataFrame({
            'year': forecasted_years,
            'predicted_ghg': forecasted_ghg
        })
        st.dataframe(forecast_df)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(country_historical_data['year'], country_historical_data['total_ghg'], label="Historical GHG Emissions", color='blue')
        ax.plot(forecast_df['year'], forecast_df['predicted_ghg'], marker='o', color='red', label="Predicted Emissions")
        ax.set_xlabel("Year")
        ax.set_ylabel("Total GHG Emissions (MtCO₂e)")
        ax.set_title(f"{country} - GHG Emissions: Historical and Predicted")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("No predictions were generated.")

# --- Footer ---
st.markdown("---")
st.caption("Made with 💚 using Streamlit | Powered by Machine Learning")
