import streamlit as st
from utils.model_utils import run_knn, run_ann
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np


st.title("Model Training Result")

knn_model, knn_result, y_test_knn = run_knn()
ann_model, ann_result, ann_history, y_test_ann = run_ann()
ann_pred_labels = np.argmax(ann_result, axis=1)

st.subheader("KNN Classification Report")
st.write(f"Accuracy: {accuracy_score(y_test_knn, knn_result):.2f}")

cm_knn = confusion_matrix(y_test_knn, knn_result)
st.text("Confusion Matrix:")
st.dataframe(pd.DataFrame(cm_knn, columns=["Pred 0", "Pred 1", "Pred 2"], index=["True 0", "True 1", "True 2"]))

report_knn = classification_report(y_test_knn, knn_result, output_dict=True)
st.text("Classification Report:")
st.dataframe(pd.DataFrame(report_knn).transpose())

st.subheader("ANN Classification Report")
st.write(f"Accuracy: {accuracy_score(y_test_ann, ann_pred_labels):.2f}")

cm_ann = confusion_matrix(y_test_ann, ann_pred_labels)
st.text("Confusion Matrix:")
st.dataframe(pd.DataFrame(cm_ann, columns=["Pred 0", "Pred 1", "Pred 2"], index=["True 0", "True 1", "True 2"]))

report_ann = classification_report(y_test_ann, ann_pred_labels, output_dict=True)
st.text("Classification Report:")
st.dataframe(pd.DataFrame(report_ann).transpose())


# Visualisasi hasil ANN
st.subheader("ANN Training History")
fig, ax = plt.subplots()
ax.plot(ann_history.history['accuracy'], label='Train Accuracy')
ax.plot(ann_history.history['val_accuracy'], label='Val Accuracy')
ax.legend()
st.pyplot(fig)

# Simpan model
st.session_state['knn_model'] = knn_model
st.session_state['ann_model'] = ann_model

import os
import joblib

# Buat folder models jika belum ada
os.makedirs("models", exist_ok=True)

# Simpan model ke file
knn_model_filename = "models/knn_model.pkl"
ann_model_filename = "models/ann_model.h5"

joblib.dump(knn_model, knn_model_filename)
ann_model.save(ann_model_filename)

st.success("Model berhasil disimpan ke dalam folder 'models'")
