import streamlit as st
import pandas as pd
from utils.predict_utils import predict_user_input_knn, predict_user_input_ann
from utils.model_utils import scaler, features_to_use, y_map_inv, preprocess_user_input
import os
import joblib
from keras.models import load_model


st.title("User Sleep Disorder Prediction")
# Cek apakah model sudah ada di session
if 'knn_model' not in st.session_state or 'ann_model' not in st.session_state:
    if os.path.exists('utils/knn_model.pkl'):
        st.session_state['knn_model'] = joblib.load('utils/knn_model.pkl')
    else:
        st.error("Model Machine Learning belum tersedia. Silakan latih model terlebih dahulu di halaman 'Model Training'.")
        st.stop()

# Load ANN dari file jika tidak ada di session_state
if 'ann_model' not in st.session_state:
    if os.path.exists('utils/ann_model.h5'):
        st.session_state['ann_model'] = load_model('utils/ann_model.h5')
    else:
        st.error("Model ANN belum tersedia. Silakan latih model terlebih dahulu di halaman 'Model Training'.")
        st.stop()



# Form input user
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 150, 25)

# Daftar pekerjaan yang tersedia
occupations = ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher',
               'Nurse', 'Engineer', 'Accountant', 'Scientist', 'Lawyer',
               'Salesperson', 'Manager', 'Other']

# Pilihan pekerjaan
selected_occupation = st.selectbox("Select your occupation:", occupations)

# Jika "Other", minta input manual
if selected_occupation == 'Other':
    occupation = st.text_input("Please enter your occupation:")
else:
    occupation = selected_occupation


sleep_duration = st.number_input("Sleep Duration (Average for the last 7 days)", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
quality = st.slider("Quality of Sleep (Think about it)", 1, 10, 5)
activity = st.number_input("Physical Activity Level (in minutes/day)", min_value=0, max_value=9999999, value=30, step=1)
stress = st.slider("Stress Level", 1, 10, 5)
st.markdown("#### BMI Category")
st.markdown(
    """
    *Underweight*: BMI less than **18.5**  
    *Healthy Weight*: BMI between **18.5 and 24.9**  
    *Overweight*: BMI between **25 and 29.9**
    """
)
bmi = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])

hr = st.number_input("Your heart rate", min_value=50, max_value=250, value=90, step=1)
steps = st.number_input("Daily Steps (Average daily step for the last 7 days)", min_value=0, max_value=9999999, value=1000, step=1)
sbp = st.number_input("Systolic Blood Pressure", min_value=70, max_value=300, value=120, step=1)
dbp = st.number_input("Diastolic Blood Pressure", min_value=0, max_value=150, value=85, step=1)


user_input = {
    'Gender': gender,
    'Age': age,
    'Occupation': occupation,
    'Sleep Duration': sleep_duration,
    'Quality of Sleep': quality,
    'Physical Activity Level': activity,
    'Stress Level': stress,
    'BMI Category': bmi,
    'Heart Rate': hr,
    'Daily Steps': steps,
    'Systolic_BP': sbp,
    'Diastolic_BP': dbp
}

# Cek apakah model sudah ada
if 'knn_model' not in st.session_state or 'ann_model' not in st.session_state:
    st.error("Model belum dilatih. Silakan buka halaman 'Model Training' terlebih dahulu.")
else:
    # Hanya tampilkan tombol jika belum prediksi
    if 'predicted' not in st.session_state:
        if st.button("Predict Sleep Disorder", key="predict_button"):
            # Jalankan prediksi
            knn_result = predict_user_input_knn(user_input, scaler, features_to_use, st.session_state['knn_model'])
            ann_result = predict_user_input_ann(user_input, scaler, features_to_use, st.session_state['ann_model'])

            # Tampilkan hasil
            st.success(f"KNN Prediction: {knn_result[0]}")
            st.success(f"ANN Prediction: {ann_result[0]}")

            # Simpan state
            st.session_state.predicted = True
            st.session_state['user_input'] = user_input  # Agar bisa digunakan di halaman Optimized Report

    else:
        st.success("Prediksi sudah dilakukan.")
        st.write("Silakan lanjut ke halaman 'Optimized Report'.")

