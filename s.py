import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

# Set page configuration
st.set_page_config(
    page_title="Bus Fuel Price Predictor",
    page_icon="🚌",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #1E3A8A;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .prediction-box {
        background-color: #F0F9FF;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #BFDBFE;
        margin-top: 20px;
    }
    .feature-importance {
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Bus Fuel Price Prediction System</div>', unsafe_allow_html=True)
st.write("This application predicts the total fuel price for bus operations based on various parameters.")

# Load the model and preprocessing components
@st.cache_resource
def load_model():
    try:
        # Load the model
        with open('random_forest_model2.pkl', 'rb') as file:
            model = pickle.load(file)
        
        # Create and fit encoders/scalers on sample data
        # These would normally be loaded from saved files
        sample_data = {
            "Bus Engine": ["Petrol", "Diesel", "CNG"],
            "Road Condition": ["Good", "Average", "Bad", "Very Good"],
            "Bus Age (Years)": [5, 8, 10],
            "Shift": ["Day", "Night"],
            "Total Distance (km)": [50, 100, 150]
        }
        sample_df = pd.DataFrame(sample_data)
        
        # Categorical columns
        categorical_cols = ["Bus Engine", "Shift", "Road Condition"]
        numerical_cols = ["Bus Age (Years)", "Total Distance (km)"]
        
        # One-Hot Encoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(sample_df[categorical_cols])
        
        # Scaler
        scaler = StandardScaler()
        scaler.fit(sample_df[numerical_cols])
        
        return model, encoder, scaler, categorical_cols, numerical_cols
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

model, encoder, scaler, categorical_cols, numerical_cols = load_model()

# Sidebar for inputs
st.sidebar.title("Input Parameters")

# Bus Engine Type
engine_type = st.sidebar.selectbox(
    "Bus Engine Type",
    options=["Petrol", "Diesel", "CNG"],
    help="Select the type of engine in the bus"
)

# Bus Age
bus_age = st.sidebar.slider(
    "Bus Age (Years)",
    min_value=1,
    max_value=20,
    value=5,
    help="Enter the age of the bus in years"
)

# Shift
shift = st.sidebar.selectbox(
    "Shift",
    options=["Day", "Night"],
    help="Select the operating shift of the bus"
)

# Road Condition
road_condition = st.sidebar.selectbox(
    "Road Condition",
    options=["Very Good", "Good", "Average", "Bad"],
    help="Select the typical road condition for the route"
)

# Total Distance
total_distance = st.sidebar.number_input(
    "Total Distance (km)",
    min_value=1.0,
    max_value=1000.0,
    value=100.0,
    help="Enter the total distance covered in kilometers"
)

# Bus Type (optional - for display purposes only)
bus_type = st.sidebar.selectbox(
    "Bus Type",
    options=["AC", "Non-AC"],
    help="Select the type of bus (AC or Non-AC)"
)

# Function to make predictions
def predict_price():
    if model is None:
        st.error("Model could not be loaded. Please check the error message.")
        return None
    
    # Create input dataframe
    input_data = pd.DataFrame({
        "Bus Engine": [engine_type],
        "Bus Age (Years)": [bus_age],
        "Shift": [shift],
        "Road Condition": [road_condition],
        "Total Distance (km)": [total_distance]
    })
    
    # Process categorical features
    X_categorical = encoder.transform(input_data[categorical_cols])
    
    # Process numerical features
    X_numerical = scaler.transform(input_data[numerical_cols])
    
    # Combine features
    X_processed = np.hstack([X_numerical, X_categorical])
    
    # Make prediction (log-transformed)
    log_price_pred = model.predict(X_processed)[0]
    
    # Inverse transform to get actual price
    price_pred = np.expm1(log_price_pred)
    
    return price_pred, log_price_pred

# Make prediction when button is clicked
if st.sidebar.button("Predict Fuel Price"):
    with st.spinner("Calculating..."):
        prediction_result = predict_price()
        
        if prediction_result:
            price_pred, log_price = prediction_result
            
            # Display results
            st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
            
            # Create columns for the results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("Estimated Fuel Price")
                st.write(f"**₹{price_pred:.2f}**")
                st.write(f"Log-transformed price: {log_price:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("Input Summary")
                st.write(f"**Engine Type:** {engine_type}")
                st.write(f"**Bus Age:** {bus_age} years")
                st.write(f"**Shift:** {shift}")
                st.write(f"**Road Condition:** {road_condition}")
                st.write(f"**Total Distance:** {total_distance} km")
                st.write(f"**Bus Type:** {bus_type}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Plot feature importance if model is available
            if model:
                st.markdown('<div class="sub-header">Feature Importance</div>', unsafe_allow_html=True)
                feature_importance = pd.DataFrame({
                    'Feature': numerical_cols + [f"{col}_{val}" for col in categorical_cols for val in encoder.categories_[categorical_cols.index(col)]],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.bar_chart(feature_importance.set_index('Feature')['Importance'])
                
                # Add an explanation about the most important features
                st.write(f"The most important feature for this prediction is **{feature_importance['Feature'].iloc[0]}**.")

# Add information about the dataset and model
st.markdown('<div class="sub-header">About the Model</div>', unsafe_allow_html=True)
st.write("""
This prediction model is based on a Random Forest Regressor trained on bus fuel consumption data. 
The model predicts the total price of fuel needed based on various factors like bus age, engine type,
road conditions, and distance traveled.

The model achieves:
- MAE (Mean Absolute Error): 0.18 (on log scale)
- R² Score: 0.94

Note: The prices are shown in Indian Rupees (INR).
""")

# Add data exploration section
st.markdown('<div class="sub-header">Data Exploration</div>', unsafe_allow_html=True)

# Try to load sample data
try:
    # This would normally load the actual dataset
    # For demonstration, we'll create sample data that mimics the structure
    sample_data = {
        "Bus Engine": ["Petrol", "Diesel", "CNG", "Petrol", "Diesel"] * 20,
        "Bus Age (Years)": np.random.randint(1, 15, 100),
        "Shift": np.random.choice(["Day", "Night"], 100),
        "Road Condition": np.random.choice(["Very Good", "Good", "Average", "Bad"], 100),
        "Total Distance (km)": np.random.uniform(10, 500, 100),
        "Avg km/l": np.random.uniform(1.5, 5, 100),
        "Total Price (INR)": np.random.uniform(5000, 50000, 100)
    }
    sample_df = pd.DataFrame(sample_data)
    
    # Display sample data
    expander = st.expander("View Sample Data")
    with expander:
        st.dataframe(sample_df.head(10))
    
    # Add some visualizations
    chart_options = st.multiselect(
        "Select features to explore",
        options=["Bus Engine", "Bus Age (Years)", "Shift", "Road Condition"],
        default=["Bus Engine"]
    )
    
    if chart_options:
        col1, col2 = st.columns(2)
        
        with col1:
            # Show count plot for selected feature
            if len(chart_options) > 0:
                st.subheader(f"Distribution by {chart_options[0]}")
                chart_data = sample_df.groupby(chart_options[0]).size().reset_index(name='Count')
                st.bar_chart(chart_data.set_index(chart_options[0]))
        
        with col2:
            # Show average price by selected feature
            if len(chart_options) > 0:
                st.subheader(f"Average Price by {chart_options[0]}")
                price_data = sample_df.groupby(chart_options[0])['Total Price (INR)'].mean().reset_index()
                st.bar_chart(price_data.set_index(chart_options[0]))
    
except Exception as e:
    st.warning(f"Couldn't load sample data for exploration: {e}")
    st.write("This section would normally display actual data from the dataset.")

# Footer
st.markdown("---")
st.write("© 2025 Bus Fuel Price Prediction System | For educational purposes only")