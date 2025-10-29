import streamlit as st
from utils.ga_utils import run_genetic_algorithm
import os
import joblib  # untuk KNN
from keras.models import load_model  # untuk ANN

st.title("Lifestyle Optimization Result")

if 'knn_model' not in st.session_state or 'ann_model' not in st.session_state and 'user_input' not in st.session_state:
    st.warning("Model atau input belum tersedia. Silakan lakukan model training terlebih dahulu di halaman 'Model Training'")
    if os.path.exists("models/knn_model.pkl") and os.path.exists("models/ann_model.h5"):
        st.info("Model ditemukan di file. Mengambil model dari file...")
        st.session_state['knn_model'] = joblib.load("models/knn_model.pkl")
        st.session_state['ann_model'] = load_model("models/ann_model.h5")
    else:
        st.stop()
elif 'user_input' not in st.session_state:
    st.warning("Input belum tersedia. Silakan lakukan prediksi terlebih dahulu di halaman 'User Input Prediction'.")
    if os.path.exists("models/knn_model.pkl") and os.path.exists("models/ann_model.h5"):
        st.info("Model ditemukan di file. Mengambil model dari file...")
        st.session_state['knn_model'] = joblib.load("models/knn_model.pkl")
        st.session_state['ann_model'] = load_model("models/ann_model.h5")
    else:
        st.stop()
elif 'knn_model' not in st.session_state or 'ann_model' not in st.session_state:
    st.warning("Model belum tersedia. Silakan lakukan model training terlebih dahulu di halaman 'Model Training'")
    if os.path.exists("models/knn_model.pkl") and os.path.exists("models/ann_model.h5"):
        st.info("Model ditemukan di file. Mengambil model dari file...")
        st.session_state['knn_model'] = joblib.load("models/knn_model.pkl")
        st.session_state['ann_model'] = load_model("models/ann_model.h5")
    else:
        st.stop()

# Definisikan list fitur optimasi
optimized_features = [
    "Sleep Duration",
    "Physical Activity Level",
    "Daily Steps",
    "Systolic_BP",
    "Diastolic_BP",
    "Stress Level"
]

if st.button("Run Genetic Optimization"):
    best_chromosome, prediction, result_log, user_input = run_genetic_algorithm()

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

    st.subheader("Rekomendasi Spesifik untuk Setiap Fitur yang Dioptimasi:")

    for key in optimized_features:
        original = user_input.get(key)
        recommended = result_log['final']['chromosome'].get(key)

        if original is not None and recommended is not None and original != recommended:
            if isinstance(original, (int, float)) and isinstance(recommended, (int, float)):
                if recommended > original:
                    direction = "naik"
                else:
                    direction = "turun"
            else:
                direction = "ubah"

            if key == "BMI Category":
                if original != "Normal":
                    if original == "Underweight":
                        st.markdown(
                            f"- **BMI Category**: Input={original}, Rekomendasi={recommended}  \n"
                            f":green[Anda disarankan menaikkan berat badan agar mencapai kategori normal.]"
                        )
                    elif original in ["Overweight", "Obese"]:
                        st.markdown(
                            f"- **BMI Category**: Input={original}, Rekomendasi={recommended}  \n"
                            f":orange[Anda disarankan menurunkan berat badan agar mencapai kategori normal.]"
                        )
                continue  # Lewati ke fitur berikutnya
            # Tampilkan rekomendasi spesifik
            if key == "Sleep Duration":
                if direction == "naik":
                    st.markdown(f"- **Durasi Tidur**: Input={original}, Rekomendasi={recommended}  \n"
                                f":blue[Tingkatkan durasi tidur Anda menjadi {recommended} jam untuk kualitas istirahat lebih baik.]")
                else:
                    st.markdown(f"- **Durasi Tidur**: Input={original}, Rekomendasi={recommended}  \n"
                                f":blue[Kurangi durasi tidur Anda menjadi {recommended} jam untuk keseimbangan yang lebih baik.]")
            elif key == "Physical Activity Level":
                if direction == "naik":
                    st.markdown(f"- **Tingkat Aktivitas Fisik**: Input={original}, Rekomendasi={recommended}  \n"
                                f":green[Tingkatkan aktivitas fisik Anda menjadi {recommended} menit/hari demi kesehatan tubuh.]")
                else:
                    st.markdown(f"- **Tingkat Aktivitas Fisik**: Input={original}, Rekomendasi={recommended}  \n"
                                f":green[Kurangi aktivitas fisik menjadi {recommended} menit/hari untuk menjaga keseimbangan.]")
            elif key == "Daily Steps":
                if direction == "naik":
                    st.markdown(f"- **Langkah Harian**: Input={original}, Rekomendasi={recommended}  \n"
                                f":green[Perbanyak langkah harian Anda menjadi {recommended} langkah.]")
                else:
                    st.markdown(f"- **Langkah Harian**: Input={original}, Rekomendasi={recommended}  \n"
                                f":green[Kurangi langkah harian menjadi {recommended} langkah untuk menghindari kelelahan.]")
            elif key == "Systolic_BP":
                if direction == "naik":
                    st.markdown(f"- **Tekanan Darah Sistolik**: Input={original}, Rekomendasi={recommended}  \n"
                                f":gray[Tingkatkan tekanan darah sistolik menjadi {recommended} mmHg untuk kestabilan tekanan darah.]")
                else:
                    st.markdown(f"- **Tekanan Darah Sistolik**: Input={original}, Rekomendasi={recommended}  \n"
                                f":gray[Turunkan tekanan darah sistolik menjadi {recommended} mmHg demi kesehatan jantung.]")
            elif key == "Diastolic_BP":
                if direction == "naik":
                    st.markdown(f"- **Tekanan Darah Diastolik**: Input={original}, Rekomendasi={recommended}  \n"
                                f":gray[Tingkatkan tekanan darah diastolik menjadi {recommended} mmHg untuk kestabilan.]")
                else:
                    st.markdown(f"- **Tekanan Darah Diastolik**: Input={original}, Rekomendasi={recommended}  \n"
                                f":gray[Turunkan tekanan darah diastolik menjadi {recommended} mmHg demi kesehatan jantung.]")
            else:
                st.markdown(f"- **{key}**: Input={original}, Rekomendasi={recommended}  \n"
                            f":gray[Pertimbangkan untuk menyesuaikan {key.lower()} Anda demi kesehatan yang lebih baik.]")