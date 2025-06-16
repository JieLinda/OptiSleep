import numpy as np
from utils.model_utils import preprocess_user_input, y_map_inv

def predict_user_input_knn(user_input, scaler, features_to_use, knn_model):
    X_input = preprocess_user_input(user_input, scaler, features_to_use)
    y_pred = knn_model.predict(X_input)[0]
    return y_map_inv[int(y_pred)], int(y_pred)

def predict_user_input_ann(user_input, scaler, features_to_use, ann_model):
    X_input = preprocess_user_input(user_input, scaler, features_to_use)
    y_pred = np.argmax(ann_model.predict(X_input), axis=1)[0]
    return y_map_inv[int(y_pred)], int(y_pred)
