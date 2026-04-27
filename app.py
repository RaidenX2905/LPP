import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Laptop Price Predictor Pro",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background: linear-gradient(45deg, #ff8a00, #e52e71);
        color: white;
        font-weight: 800;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(229, 46, 113, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(229, 46, 113, 0.4);
        background: linear-gradient(45deg, #e52e71, #ff8a00);
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        color: #e52e71;
        margin: 1rem 0;
    }
    
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-weight: 600 !important;
        color: #333 !important;
    }
    
    .hero-text {
        text-align: center;
        padding: 2rem 0;
    }
    
    .hero-text h1 {
        font-size: 4rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#333, #777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_assets():
    model = joblib.load('random_forest_model.joblib')
    encoders = joblib.load('label_encoder_map.joblib')
    # Load raw data for choices
    df_raw = pd.read_csv('data/laptops.csv')
    # Preprocess raw data for choices (same as training)
    df_raw["rating"] = df_raw["rating"].fillna(df_raw["rating"].median())
    df_raw["no_of_ratings"] = df_raw["no_of_ratings"].fillna(df_raw["no_of_ratings"].median())
    df_raw["no_of_reviews"] = df_raw["no_of_reviews"].fillna(df_raw["no_of_reviews"].median())
    return model, encoders, df_raw

try:
    model, encoders, df_raw = load_assets()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Sidebar for additional info
with st.sidebar:
    st.title("About")
    st.info("This AI-powered tool predicts the market price of a laptop based on its specifications with ~88% accuracy.")
    st.markdown(f"[🚀 Live App Link](https://laptop-price-prediction-by-jsh.streamlit.app/)")
    st.markdown("---")
    st.subheader("Model Info")
    st.write("**Algorithm:** Random Forest Regressor")
    st.write("**Dataset:** 980+ Laptop listings")
    st.markdown("---")
    st.write("Created by Antigravity AI")

# Main Page
st.markdown('<div class="hero-text"><h1>Laptop Price Predictor</h1></div>', unsafe_allow_html=True)

# Hero image
col_img_1, col_img_2, col_img_3 = st.columns([1, 2, 1])
with col_img_2:
    # Look for the generated image or use a placeholder if not found
    img_path = [f for f in os.listdir('.') if f.startswith('laptop_hero_image') and f.endswith('.png')]
    if img_path:
        st.image(img_path[0], use_container_width=True)
    else:
        st.image("https://images.unsplash.com/photo-1496181133206-80ce9b88a853?auto=format&fit=crop&q=80&w=1000", use_container_width=True)

st.markdown("---")

# Inputs
with st.container():
    st.subheader("💻 Technical Specifications")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        name = st.selectbox("Brand / Model Series", sorted(df_raw['name'].unique()))
        processor = st.selectbox("Processor", sorted(df_raw['processor'].unique()))
        ram = st.selectbox("RAM Size", sorted(df_raw['ram'].unique()))
        
    with col2:
        os_type = st.selectbox("Operating System", sorted(df_raw['os'].unique()))
        storage = st.selectbox("Storage", sorted(df_raw['storage'].unique()))
        display = st.number_input("Display Size (inch)", value=15.6, step=0.1)
        
    with col3:
        rating = st.slider("Market Rating", 1.0, 5.0, 4.3, 0.1)
        num_ratings = st.number_input("Total Ratings count", value=100, step=10)
        num_reviews = st.number_input("Total Reviews count", value=20, step=5)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Logic
if st.button("Calculate Predicted Price"):
    try:
        # Encode inputs
        encoded_input = [
            encoders['name'].transform([name])[0],
            encoders['processor'].transform([processor])[0],
            encoders['ram'].transform([ram])[0],
            encoders['os'].transform([os_type])[0],
            encoders['storage'].transform([storage])[0],
            display,
            rating,
            num_ratings,
            num_reviews
        ]
        
        # Reshape for prediction
        input_array = np.array(encoded_input).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        
        # Display Result
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Estimated Market Value</h3>
                <div class="prediction-value">₹ {prediction:,.2f}</div>
                <p>Based on current market trends and configurations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.balloons()
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")
st.caption("© 2026 Laptop Price Prediction Project. All rights reserved.")
