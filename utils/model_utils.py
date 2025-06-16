from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


# Load and preprocess data

# === DATA PREPROCESSING ===
df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')
df['Sleep Disorder'].fillna('Normal', inplace=True)

# Mapping
gender_mapping = {'Male': 0, 'Female': 1}
occupation_mapping = {
    'Manager': 0, 'Engineer': 1, 'Doctor': 2, 'Lawyer': 3, 'Accountant': 4,
    'Software Engineer': 5, 'Scientist': 6, 'Teacher': 7,
    'Nurse': 8, 'Salesperson': 9, 'Sales Representative': 10
}
df['BMI Category'] = df['BMI Category'].replace({'Normal Weight': 'Underweight'})
bmi_mapping = {'Normal': 0, 'Underweight': 1, 'Overweight': 2, 'Obese': 3}

df['Gender'] = df['Gender'].map(gender_mapping)
df['Occupation'] = df['Occupation'].map(occupation_mapping)
df['BMI Category'] = df['BMI Category'].map(bmi_mapping)
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df.drop(columns=['Blood Pressure'], inplace=True)

# Target dan fitur
features = ['Gender', 'Age', 'Occupation','Sleep Duration','Quality of Sleep','Physical Activity Level','Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']
y_map = {'Normal': 0, 'Insomnia': 1, 'Sleep Apnea': 2}
y_map_inv = {0: 'Normal', 1: 'Insomnia', 2: 'Sleep Apnea'}
X = df[features]
y = df['Sleep Disorder'].map(y_map)

# Scaling & Split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42, stratify=y)
features_to_use = features

def preprocess_user_input(user_input, scaler, features_to_use): #encoder, occ_columns,
    df_input = pd.DataFrame([user_input])

    # Mapping berdasarkan analisis persentase
    gender_mapping = {'Male': 0, 'Female': 1.5}
    bmi_mapping = {
        'Normal': 0,
        'Underweight': 1.5,
        'Overweight': 4,
        'Obese': 4.5
    }
    occupation_mapping = {
        'Manager': 0,
        'Engineer': 1,
        'Doctor': 2,
        'Lawyer': 3,
        'Accountant': 4,
        'Software Engineer': 5,
        'Scientist': 6,
        'Teacher': 7,
        'Nurse': 8,
        'Salesperson': 9,
        'Sales Representative': 10
    }

    df_input['Gender'] = df_input['Gender'].map(gender_mapping)
    df_input['BMI Category'] = df_input['BMI Category'].map(bmi_mapping)
    median_occupation_value = np.median(list(occupation_mapping.values()))
    df_input['Occupation'] = df_input['Occupation'].map(lambda x: occupation_mapping.get(x, median_occupation_value))

    for feature in features_to_use:
        if feature not in df_input.columns:
            df_input[feature] = 0  # atau nilai median default lain jika ada

    # Urutkan kolom agar sesuai training
    df_input = df_input[features_to_use]

    print("Preprocessed Input Shape:", df_input.shape)
    print("Columns:", df_input.columns.tolist())


    # Scaling
    X_scaled = scaler.transform(df_input)

    return X_scaled



def run_knn(n_neighbors=7, metric='minkowski'):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    # Simpan model ke file
    joblib.dump(knn, 'utils/knn_model.pkl')
    print("=== KNN Results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return knn, y_pred, y_test 


def run_ann(hidden_layers=[32, 16], activation='relu', epochs=50, batch_size=16):
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_shape=(X_train.shape[1],), activation=activation))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation=activation))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    y_pred = model.predict(X_test)

    model.save('utils/ann_model.h5')  # Simpan ANN
    print("=== ANN Results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred.argmax(axis=1)))
    return model, y_pred, history, y_test 
