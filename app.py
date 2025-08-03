import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Hyundai Palisade Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 2026 Hyundai Palisade Price Predictor")
st.markdown("""
This app predicts whether a Hyundai Palisade listing is priced fairly based on its characteristics.
Enter the vehicle details below to get a price prediction and evaluation.
""")

# Load model function
@st.cache_resource
def load_model():
    try:
        # Try joblib first (better compatibility)
        import joblib
        model = joblib.load('palisade_price_model.pkl')
        return model
    except:
        try:
            # Fallback to pickle
            with open('palisade_price_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Model loading failed. The app will work in demo mode with sample predictions.")
            return None

# Load the model
model = load_model()

# Sidebar for inputs
st.sidebar.header("Vehicle Characteristics")

# Model features based on your trained model
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vehicle Specifications")
    
    # Trim level - exact values from your model
    trim_level = st.selectbox("Trim Level", [
        "Calligraphy", "Limited", "SEL", "SEL Convenience"
    ])
    
    # Drivetrain - exact values from your model
    drivetrain = st.selectbox("Drivetrain", [
        "AWD", "FWD"
    ])
    
    # City MPG - numerical feature from your model
    city_mpg = st.number_input("City MPG", min_value=15, max_value=25, value=19, step=1)

with col2:
    st.subheader("Color Options")
    
    # Exterior color - exact values from your model
    ext_color = st.selectbox("Exterior Color", [
        "Abyss Black", "Classy Blue", "Creamy White", "Ecotronic Gray"
    ])
    
    # Interior color - exact values from your model
    int_color = st.selectbox("Interior Color", [
        "Black", "Brown", "Gray", "Gray/Navy", "Navy/Brown"
    ])

# Asking price input
st.subheader("Price Evaluation")
asking_price = st.number_input("Asking Price ($)", min_value=20000, max_value=80000, value=45000, step=500)

# Create feature vector exactly as your model expects
def prepare_input(trim, drivetrain, city_mpg, ext, interior):
    # Create a single-row DataFrame with all zero columns (one-hot encoded)
    cols = [
        'Trim_Calligraphy','Trim_Limited','Trim_SEL','Trim_SEL Convenience',
        'Drivetrain_AWD','Drivetrain_FWD','City_mpg',
        'ExtColor_Abyss Black','ExtColor_Classy Blue','ExtColor_Creamy White','ExtColor_Ecotronic Gray',
        'IntColor_Black','IntColor_Brown','IntColor_Gray','IntColor_Gray/Navy','IntColor_Navy/Brown'
    ]
    row = {col: 0 for col in cols}
    row[f'Trim_{trim}'] = 1
    row[f'Drivetrain_{drivetrain}'] = 1
    row[f'ExtColor_{ext}'] = 1
    row[f'IntColor_{interior}'] = 1
    row['City_mpg'] = city_mpg
    return pd.DataFrame([row])

# Prediction button
if st.button("Predict Price", type="primary"):
    # Create feature vector using your model's exact format
    features = prepare_input(trim_level, drivetrain, city_mpg, ext_color, int_color)
    
    if model is not None:
        try:
            # Make prediction
            predicted_price = model.predict(features)[0]
            
            # Calculate price difference
            price_diff = asking_price - predicted_price
            price_diff_pct = (price_diff / predicted_price) * 100
            
            # Display results
            st.subheader("Price Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted Fair Price",
                    value=f"${predicted_price:,.0f}"
                )
            
            with col2:
                st.metric(
                    label="Asking Price",
                    value=f"${asking_price:,.0f}"
                )
            
            with col3:
                st.metric(
                    label="Price Difference",
                    value=f"${price_diff:,.0f}",
                    delta=f"{price_diff_pct:.1f}%"
                )
            
            # Price evaluation
            if abs(price_diff_pct) <= 5:
                st.success("✅ **Fair Price!** This vehicle appears to be priced reasonably.")
            elif price_diff_pct > 5:
                st.warning(f"⚠️ **Overpriced!** This vehicle appears to be ${price_diff:,.0f} ({price_diff_pct:.1f}%) above fair market value.")
            else:
                st.info(f"💰 **Good Deal!** This vehicle appears to be ${abs(price_diff):,.0f} ({abs(price_diff_pct):.1f}%) below fair market value.")
            
            # Visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Predicted Price', 'Asking Price'],
                y=[predicted_price, asking_price],
                marker_color=['lightblue', 'coral' if price_diff > 0 else 'lightgreen'],
                text=[f'${predicted_price:,.0f}', f'${asking_price:,.0f}'],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Price Comparison",
                yaxis_title="Price ($)",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        # Demo mode - provide sample prediction
        st.warning("Demo Mode: Model not available, showing sample prediction")
        
        # Simple heuristic-based prediction for demo using actual feature values
        base_price = 38000  # Base price
        trim_adjustment = {"SEL": 0, "SEL Convenience": 2000, "Limited": 8000, "Calligraphy": 12000}
        awd_adjustment = 2000 if drivetrain == "AWD" else 0
        mpg_adjustment = (city_mpg - 19) * 500  # Adjust based on fuel efficiency
        
        # Color adjustments (premium colors might cost more)
        ext_color_adjustment = {"Abyss Black": 0, "Classy Blue": 500, "Creamy White": 300, "Ecotronic Gray": 0}
        int_color_adjustment = {"Black": 0, "Brown": 800, "Gray": 0, "Gray/Navy": 600, "Navy/Brown": 1000}
        
        predicted_price = (base_price + trim_adjustment.get(trim_level, 0) + 
                         awd_adjustment + mpg_adjustment + 
                         ext_color_adjustment.get(ext_color, 0) + 
                         int_color_adjustment.get(int_color, 0))
        
        price_diff = asking_price - predicted_price
        price_diff_pct = (price_diff / predicted_price) * 100
        
        st.subheader("Demo Price Analysis Results")
        st.info("Note: This is a demonstration using simplified calculations. Load your trained model for accurate predictions.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Estimated Price",
                value=f"${predicted_price:,.0f}"
            )
        
        with col2:
            st.metric(
                label="Asking Price",
                value=f"${asking_price:,.0f}"
            )
        
        with col3:
            st.metric(
                label="Price Difference",
                value=f"${price_diff:,.0f}",
                delta=f"{price_diff_pct:.1f}%"
            )

# Additional information
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. **Enter vehicle details** in the form above
    2. **Set the asking price** you want to evaluate
    3. **Click "Predict Price"** to get the analysis
    4. **Review the results** to see if the price is fair
    
    **Price Evaluation Guide:**
    - ✅ **Fair Price**: Within 5% of predicted value
    - ⚠️ **Overpriced**: More than 5% above predicted value  
    - 💰 **Good Deal**: More than 5% below predicted value
    """)

with st.expander("🔧 Model Information"):
    if model is not None:
        st.success("✅ Price prediction model loaded successfully")
        # Show the exact features used
        st.write("**Model Features:**")
        features_list = [
            'Trim_Calligraphy','Trim_Limited','Trim_SEL','Trim_SEL Convenience',
            'Drivetrain_AWD','Drivetrain_FWD','City_mpg',
            'ExtColor_Abyss Black','ExtColor_Classy Blue','ExtColor_Creamy White','ExtColor_Ecotronic Gray',
            'IntColor_Black','IntColor_Brown','IntColor_Gray','IntColor_Gray/Navy','IntColor_Navy/Brown'
        ]
        for i, feature in enumerate(features_list, 1):
            st.write(f"{i}. {feature}")
        st.write(f"**Total Features:** {len(features_list)}")
    else:
        st.warning("⚠️ Model not loaded - running in demo mode")
        st.markdown("""
        **To use your trained model:**
        1. Ensure the model file `palisade_price_model.pkl` is in the same directory
        2. Install joblib: `pip install joblib`
        3. Restart the app
        4. The model should load automatically
        """)