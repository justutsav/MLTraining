import streamlit as st
import joblib
import pandas as pd
import numpy as np


# 1. Load the Compressed Model and Data
@st.cache_resource
def load_assets():
    """Load the pre-trained model and dropdown lists from pkl files."""
    model = joblib.load('crop_model.pkl')
    areas = joblib.load('areas_list.pkl')
    items = joblib.load('items_list.pkl')
    return model, areas, items


model, area_options, item_options = load_assets()

# 2. UI Layout
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="wide")
st.title("🌾 Precision Crop Yield Prediction")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("📍 Location & Crop")
    area = st.selectbox("Area (Country)", options=area_options)
    item = st.selectbox("Item (Crop Type)", options=item_options)
    year = st.number_input("Year", min_value=1990, max_value=2030, value=2024, step=1)

with col2:
    st.header("🧪 Environmental Factors")
    rainfall = st.number_input("Average Rainfall (mm/year)", min_value=0.0, value=1485.0, step=10.0)
    pesticides = st.number_input("Pesticides usage (tonnes)", min_value=0.0, value=121.0, step=0.1)
    temp = st.slider("Average Temperature (°C)", min_value=-10.0, max_value=50.0, value=16.37)

st.markdown("---")

# 3. Prediction Logic
if st.button("Calculate Expected Yield", type="primary"):
    # IMPORTANT: Features must match the EXACT column names the model was trained on.
    # The Pipeline handles OneHotEncoding and StandardScaling internally.
    input_data = pd.DataFrame(
        [[area, item, year, rainfall, pesticides, temp]],
        columns=['Area', 'Item', 'Year',
                 'average_rain_fall_mm_per_year',
                 'pesticides_tonnes', 'avg_temp']
    )

    try:
        prediction = model.predict(input_data)
        st.balloons()
        st.metric(label="Predicted Yield (hg/ha)", value=f"{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {e}. Check if your input features match the model training.")
