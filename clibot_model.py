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
import os

# 1. Cargar datos (ruta relativa)
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')  # El archivo ya está en tu directorio

# Limpieza y preprocesamiento (igual que antes)
df = df[df['TotalCharges'] != ' '].copy()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# 2. Crear loyalty_index y loyalty_class (igual que antes)
def calc_loyalty(row):
    score = int(row['tenure'])
    score += 10 if row['Churn'] == 'No' else 0
    return score

df['loyalty_index'] = df.apply(calc_loyalty, axis=1)

loyalty_bins = np.percentile(df['loyalty_index'], [33, 66])
def loyalty_category(x):
    if x <= loyalty_bins[0]:
        return 0  # Baja
    elif x <= loyalty_bins[1]:
        return 1  # Media
    else:
        return 2  # Alta
df['loyalty_class'] = df['loyalty_index'].apply(loyalty_category)

# 3. Features y pipeline (igual que antes)
ignore_cols = ['customerID', 'Churn', 'loyalty_index', 'loyalty_class']
feature_cols = [c for c in df.columns if c not in ignore_cols]

numeric_features = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
categorical_features = list(set(feature_cols) - set(numeric_features))

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

clf = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, multi_class='multinomial'))
])

# 4. Entrenamiento y evaluación (igual que antes)
X = df[feature_cols]
y = df['loyalty_class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 5. Métricas y gráficos (igual que antes)
class_names = ['Baja', 'Media', 'Alta']
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap="Blues")
plt.title('Matriz de confusión - Lealtad')
plt.show()

# 6. Guardar resultados localmente
df.to_csv('Dataset_with_Loyalty_Fields.csv', index=False)  # Sobrescribe el archivo existente
joblib.dump(clf, 'modelo_lealtad.pkl')

print("✅ Proceso completado. Archivos guardados:")
print(f"- Dataset con lealtad: {os.path.abspath('Dataset_with_Loyalty_Fields.csv')}")
print(f"- Modelo entrenado: {os.path.abspath('modelo_lealtad.pkl')}")