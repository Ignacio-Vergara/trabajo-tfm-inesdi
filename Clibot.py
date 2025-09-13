import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

from google.colab import files
uploaded = files.upload()  # para subir archivo desde  PC

modelo = joblib.load('modelo_lealtad.pkl')

ignore_cols = ['customerID', 'Churn']
feature_cols = [c for c in df_nuevo.columns if c not in ignore_cols]

df_nuevo = pd.read_csv('dataset_prueba.csv') ### Dataset de Telco SAS

df_nuevo['TotalCharges'] = df_nuevo['TotalCharges'].replace(' ', np.nan)
df_nuevo['TotalCharges'] = pd.to_numeric(df_nuevo['TotalCharges'])
df_nuevo['TotalCharges'] = df_nuevo['TotalCharges'].fillna(df_nuevo['TotalCharges'].median())

# Predecir usando las mismas columnas de entrenamiento
X_nuevo = df_nuevo[feature_cols]
predicciones = modelo.predict(X_nuevo)
print(predicciones)