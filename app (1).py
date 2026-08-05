import streamlit as st
import joblib
import numpy as np

st.title("🚢 Titanic Survival Prediction App")
st.write("પેસેન્જરની વિગતો દાખલ કરો અને જુઓ કે તે બચશે કે નહીં:")

# Inputs
pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
age = st.slider("Age", 1, 100, 25)
fare = st.slider("Fare ($)", 0.0, 500.0, 32.0)
sex = st.selectbox("Gender", ["Male", "Female"])

sex_male = 1 if sex == "Male" else 0

if st.button("Predict"):
    model = joblib.load('titanic_model.pkl')
    features = np.array([[pclass, age, fare, sex_male]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.success("🎉 પેસેન્જર બચી જશે! (Survived)")
        st.balloons()
    else:
        st.error("💀 પેસેન્જર બચી શકશે નહીં (Not Survived)")
