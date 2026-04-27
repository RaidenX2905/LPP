import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration for a premium look without animations
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a more premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_data_and_model():
    # Paths to files
    model_path = 'random_forest_model.joblib'
    le_map_path = 'label_encoder_map.joblib'
    csv_path = 'data/laptops.csv'

    # Load model and encoder map
    rf = joblib.load(model_path)
    le_map = joblib.load(le_map_path)
    
    # Load dataset for dropdown choices
    df_raw = pd.read_csv(csv_path)
    
    # Preprocessing (same as training)
    df_raw["rating"] = df_raw["rating"].fillna(df_raw["rating"].median())
    df_raw["no_of_ratings"] = df_raw["no_of_ratings"].fillna(df_raw["no_of_ratings"].median())
    df_raw["no_of_reviews"] = df_raw["no_of_reviews"].fillna(df_raw["no_of_reviews"].median())
    
    return rf, le_map, df_raw

def main():
    st.title("💻 Laptop Price Predictor")
    st.markdown("""
    Estimate the price of a laptop based on its specifications. 
    This model uses a **Random Forest Regressor** with high accuracy (~87% R2 score).
    """)

    try:
        rf, le_map, df_raw = load_data_and_model()
    except Exception as e:
        st.error(f"Error loading model or data: {e}. Ensure you are running from the correct directory.")
        return



    # Get unique values for dropdowns
    name_choices = sorted(df_raw['name'].unique().tolist())
    processor_choices = sorted(df_raw['processor'].unique().tolist())
    ram_choices = sorted(df_raw['ram'].unique().tolist())
    os_choices = sorted(df_raw['os'].unique().tolist())
    storage_choices = sorted(df_raw['storage'].unique().tolist())

    with st.container():
        st.subheader("Input Specifications")
        col1, col2 = st.columns(2)

        with col1:
            name_val = st.selectbox("Laptop Name", name_choices)
            processor_val = st.selectbox("Processor Type", processor_choices)
            ram_val = st.selectbox("RAM Configuration", ram_choices)
            os_val = st.selectbox("Operating System", os_choices)
            storage_val = st.selectbox("Storage Type", storage_choices)

        with col2:
            display_size_val = st.number_input("Display Size (in inch)", min_value=10.0, max_value=40.0, value=15.6, step=0.1)
            rating_val = st.slider("Rating", min_value=1.0, max_value=5.0, value=4.3, step=0.1)
            no_of_ratings_val = st.number_input("Number of Ratings", min_value=0, value=90)
            no_of_reviews_val = st.number_input("Number of Reviews", min_value=0, value=11)

    if st.button("Predict Price", type="primary"):
        # Encode categorical inputs
        try:
            encoded_name = le_map['name'].transform([name_val])[0]
            encoded_processor = le_map['processor'].transform([processor_val])[0]
            encoded_ram = le_map['ram'].transform([ram_val])[0]
            encoded_os = le_map['os'].transform([os_val])[0]
            encoded_storage = le_map['storage'].transform([storage_val])[0]

            # Prepare feature vector
            feature_columns = ['name', 'processor', 'ram', 'os', 'storage', 'display(in inch)', 'rating', 'no_of_ratings', 'no_of_reviews']
            input_data = [
                encoded_name,
                encoded_processor,
                encoded_ram,
                encoded_os,
                encoded_storage,
                display_size_val,
                rating_val,
                no_of_ratings_val,
                no_of_reviews_val
            ]
            input_df = pd.DataFrame([input_data], columns=feature_columns)

            # Predict
            prediction = rf.predict(input_df)[0]
            
            # Display result
            st.markdown(f"""
            <div class='prediction-card'>
                <h2>Estimated Price</h2>
                <h1 style='color: #ff4b4b;'>₹ {prediction:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # NO balloons as requested.
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == '__main__':
    main()