# 🌍📈 Carbon Emissions Predictor

This repository contains a **Streamlit web application** designed to forecast **global CO₂ emissions** and **country-specific Greenhouse Gas (GHG) emissions** using robust machine learning models.

---

## 📌 Prediction Modes

### 🔹 Global CO₂ Prediction
Forecasts future global CO₂ emissions based on comprehensive historical 'World' data.

### 🔹 Country-wise GHG Prediction
Predicts future total GHG emissions for selected individual countries, leveraging their historical emission patterns.

The underlying models are meticulously trained on the `owid-co2-data.csv` dataset, which includes rich emission metrics, population demographics, and GDP figures.

---

## 🚀 Live Streamlit App

Access the deployed Streamlit application here:  
🔗 [Carbon Emissions Predictor App](https://carbon-emissions-predictor-wnqfmucbp8d7km3vrrbzfv.streamlit.app/)

---

## ✨ Features

- **Dual Prediction Modes**: Switch seamlessly between global CO₂ and country-wise GHG predictions.
- **Interactive Forecasting**: Select and predict emissions for customizable future years.
- **Historical Data Visualization**: Visual plots of trends and predictions.
- **User-Friendly Interface**: Built with Streamlit for an intuitive user experience.
- **Themed Aesthetics**: Dark background with consistent theme.

---

## 📸 Screenshots

> Replace these placeholders with real screenshots.

| Global Prediction | Country-wise Prediction |
|-------------------|--------------------------|
| ![image](https://github.com/user-attachments/assets/ef7a19a6-7228-4d07-82af-94c82cdcb274)| ![image](https://github.com/user-attachments/assets/c2c379c8-d7df-4159-8f24-150f11dc624f)
 |

- 📋 App Sidebar and Controls

---

## 📁 Project Structure

```
├── app.py                            # Main Streamlit app
├── carbon predictor.ipynb            # Jupyter notebook for model development
├── owid-co2-data.csv                 # Source dataset
├── xgboost_co2_predictor_model.joblib  # Global model
├── ghg_model.pkl                     # Country-wise GHG model
├── ghg_scaler.pkl                    # StandardScaler for GHG model
├── preprocessed_data.csv            # Cleaned dataset for GHG prediction
├── static/
│   └── 1.png                         # Background image
├── requirements.txt                  # Python dependencies
```

---

## 🛠️ Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Blaze-0903/Carbon-Emissions-Predictor.git
cd Carbon-Emissions-Predictor
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
```

### 3. Activate the Environment

- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

- **Windows (CMD)**:
  ```bash
  venv\Scripts\activate.bat
  ```

- **Windows (PowerShell)**:
  ```powershell
  venv\Scripts\Activate.ps1
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Prepare Required Files

Ensure these files exist in the root directory:
- `owid-co2-data.csv`
- `xgboost_co2_predictor_model.joblib`
- `ghg_model.pkl`
- `ghg_scaler.pkl`
- `preprocessed_data.csv`
- `static/1.png`

If missing, generate them via your preprocessing script (e.g. `create_ghg_assets.py`).

### 6. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🤝 Contributing

Contributions are welcome! Fork this repo, make changes, and submit a pull request.

---

## 📄 License

[Specify your license here, e.g., MIT License]

---

> Made with 💚 using Python, Streamlit, Pandas, and XGBoost.
