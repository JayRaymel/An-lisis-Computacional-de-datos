import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Cargar los datos
file_path = 'E:\DATA\churn_data.csv'  # Cambia esto con la ruta correcta
data = pd.read_csv(file_path)

# Preprocesamiento de los datos
data_cleaned = data.drop(columns=["ClienteID"])  # Eliminar columna innecesaria
data_cleaned['Género'] = data_cleaned['Género'].map({'F': 0, 'M': 1})  # Codificar Género

# Separar características (X) y variable objetivo (y)
X = data_cleaned.drop(columns=["Churn"])
y = data_cleaned["Churn"]

# Dividir datos en conjuntos de entrenamiento y prueba (70% - 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar un modelo básico de Árbol de Decisión
model_basic = DecisionTreeClassifier(random_state=42)
model_basic.fit(X_train, y_train)

# Evaluar el rendimiento inicial
y_pred = model_basic.predict(X_test)
y_proba = model_basic.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva (1)

# Calcular métricas detalladas
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Mostrar resultados
print("Classification Report:")
print(report)
print(f"ROC-AUC: {roc_auc:.2f}")

# Generar la curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.grid()
plt.show()
