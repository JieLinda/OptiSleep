import streamlit as st
from utils.ga_utils import run_genetic_algorithm
import os
import joblib  # untuk KNN
from keras.models import load_model  # untuk ANN


st.title("Lifestyle Optimization Result")
if 'knn_model' not in st.session_state or 'ann_model' not in st.session_state and 'user_input' not in st.session_state:
    st.warning("Model atau input belum tersedia. Silakan lakukan model training terlebih dahulu di halaman 'Model Training'")
    
    # 🛠️ Optional: Load model dari file jika tersedia
    if os.path.exists("models/knn_model.pkl") and os.path.exists("models/ann_model.h5"):
        st.info("Model ditemukan di file. Mengambil model dari file...")
        st.session_state['knn_model'] = joblib.load("models/knn_model.pkl")
        st.session_state['ann_model'] = load_model("models/ann_model.h5")
    else:
        st.stop()
elif 'user_input' not in st.session_state:
    st.warning("Input belum tersedia. Silakan lakukan prediksi terlebih dahulu di halaman 'User Input Prediction'.")
    # 🛠️ Optional: Load model dari file jika tersedia
    if os.path.exists("models/knn_model.pkl") and os.path.exists("models/ann_model.h5"):
        st.info("Model ditemukan di file. Mengambil model dari file...")
        st.session_state['knn_model'] = joblib.load("models/knn_model.pkl")
        st.session_state['ann_model'] = load_model("models/ann_model.h5")
    else:
        st.stop()
elif 'knn_model' not in st.session_state or 'ann_model' not in st.session_state:
    st.warning("Model belum tersedia. Silakan lakukan model training terlebih dahulu di halaman 'Model Training'")
    
    # 🛠️ Optional: Load model dari file jika tersedia
    if os.path.exists("models/knn_model.pkl") and os.path.exists("models/ann_model.h5"):
        st.info("Model ditemukan di file. Mengambil model dari file...")
        st.session_state['knn_model'] = joblib.load("models/knn_model.pkl")
        st.session_state['ann_model'] = load_model("models/ann_model.h5")
    else:
        st.stop()


if st.button("Run Genetic Optimization"):
    best_chromosome, prediction, result_log = run_genetic_algorithm()

    st.subheader("Kromosom Terbaik (Hasil Evaluasi GA Awal)")
    st.json(best_chromosome['chromosome'])

    st.markdown(f"""
    #### Hasil Prediksi Model:
    - Prediksi KNN: **{best_chromosome['knn_label']}** (Fitness: {best_chromosome['knn_fitness']:.3f})
    - Prediksi ANN: **{best_chromosome['ann_label']}** (Fitness: {best_chromosome['ann_fitness']:.3f})
    - Rata-rata Skor Fitness: **{best_chromosome['avg_fitness']:.3f}**
    """)

    st.subheader("Kromosom Terbaik di Akhir GA")
    st.json(result_log['final']['chromosome'])

    st.markdown(f"""
    #### Prediksi Sleep Disorder setelah Optimasi:
    - Prediksi KNN: **{result_log['final']['knn_label']}** (Fitness: {result_log['final']['knn_fitness']:.3f})
    - Prediksi ANN: **{result_log['final']['ann_label']}** (Fitness: {result_log['final']['ann_fitness']:.3f})
    - Fitness (avg): **{result_log['final']['avg_fitness']:.3f}**
    """)