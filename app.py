import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  # Make sure model.pkl is in the same directory

model = load_model()

# Function to set faded background image with clear text
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255,255,255,0.6), rgba(255,255,255,0.6)), 
                        url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            color: black !important;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: black !important;
        }}
        .css-1cpxqw2, .css-10trblm, .css-1d391kg {{
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
set_background("bankruptcy_wallpaper.jpg")  # Make sure this image exists in the same folder

# App title
st.title("💼 Bankruptcy Prediction System")
st.markdown("Use financial and operational risk metrics to predict company bankruptcy.")

# Sidebar inputs
st.sidebar.header("📊 Enter Company Features")

def user_input_features():
    industrial_risk = st.sidebar.selectbox('Industrial Risk', [0, 0.5, 1])
    management_risk = st.sidebar.selectbox('Management Risk', [0, 0.5, 1])
    financial_flexibility = st.sidebar.selectbox('Financial Flexibility', [0, 0.5, 1])
    credibility = st.sidebar.selectbox('Credibility', [0, 0.5, 1])
    competitiveness = st.sidebar.selectbox('Competitiveness', [0, 0.5, 1])
    operating_risk = st.sidebar.selectbox('Operating Risk', [0, 0.5, 1])

    data = {
        'industrial_risk': industrial_risk,
        'management_risk': management_risk,
        'financial_flexibility': financial_flexibility,
        'credibility': credibility,
        'competitiveness': competitiveness,
        'operating_risk': operating_risk
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display input
st.subheader("🔍 Selected Input Features")
st.write(input_df)

# Predict button
if st.button("🔮 Predict Bankruptcy Status"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)

    result = "🟢 **Non-Bankrupt**" if prediction == 1 else "🔴 **Bankrupt**"
    st.subheader("📈 Prediction Result")
    st.success(f"The company is predicted to be: {result}")

    st.subheader("📊 Prediction Probability")
    st.write({
        "Bankruptcy": round(prediction_proba[0][0]*100, 2),
        "Non-Bankruptcy": round(prediction_proba[0][1]*100, 2)
    })

# Footer
st.markdown("---")
st.markdown("🚀 Developed with ❤️ using Streamlit")
