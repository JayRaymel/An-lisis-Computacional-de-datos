import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Cargar los datos
file_path = 'E:/DATA/churn_data.csv'  # Cambia esto con la ruta correcta
data = pd.read_csv(file_path)

# Preprocesamiento de los datos
data_cleaned = data.drop(columns=["ClienteID"])  # Eliminar columna innecesaria
data_cleaned['Género'] = data_cleaned['Género'].map({'F': 0, 'M': 1})  # Codificar Género

# Separar características (X) y variable objetivo (y)
X = data_cleaned.drop(columns=["Churn"])
y = data_cleaned["Churn"]

# Dividir datos en conjuntos de entrenamiento y prueba (70% - 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Entrenar un modelo básico de Árbol de Decisión
model_basic = DecisionTreeClassifier(random_state=42)
model_basic.fit(X_train, y_train)

# 2. Evaluar el rendimiento inicial
y_pred = model_basic.predict(X_test)
y_proba = model_basic.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva (1)

# Calcular métricas detalladas
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Mostrar resultados
print("Classification Report (Modelo Básico):")
print(report)
print(f"ROC-AUC (Modelo Básico): {roc_auc:.2f}")

# Generar la curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC (Modelo Básico)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# 3. Optimización de Hiperparámetros con GridSearch
param_grid = {
    'max_depth': [3, 5, 7, 10],                # Profundidad máxima del árbol
    'min_samples_split': [2, 5, 10],           # Número mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 5],             # Número mínimo de muestras en una hoja
    'criterion': ['gini', 'entropy'],         # Función de medida de calidad de división
    'max_features': [None, 'sqrt', 'log2']    # Número máximo de características a considerar
}

# Crear el objeto GridSearchCV
grid_search = GridSearchCV(estimator=model_basic, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Entrenar el modelo con GridSearchCV
grid_search.fit(X_train, y_train)

# Imprimir los mejores parámetros encontrados
print("Mejores parámetros del modelo optimizado:", grid_search.best_params_)
model_optimized = grid_search.best_estimator_

# 4. Evaluar el modelo optimizado
y_pred_optimized = model_optimized.predict(X_test)
y_proba_optimized = model_optimized.predict_proba(X_test)[:, 1]
roc_auc_optimized = roc_auc_score(y_test, y_proba_optimized)

print(f"ROC-AUC (Modelo Optimizado): {roc_auc_optimized:.2f}")

# Generar la curva ROC del modelo optimizado
fpr_optimized, tpr_optimized, _ = roc_curve(y_test, y_proba_optimized)
plt.figure(figsize=(8, 6))
plt.plot(fpr_optimized, tpr_optimized, label=f"ROC Curve (AUC = {roc_auc_optimized:.2f})", color="green")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC (Modelo Optimizado)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# 5. Comparación de Resultados
print(f"Comparación de ROC-AUC:")
print(f"Modelo Básico: {roc_auc:.2f}")
print(f"Modelo Optimizado: {roc_auc_optimized:.2f}")

# 6. Análisis de Sesgo y Varianza (Curvas de Aprendizaje)
train_sizes, train_scores, test_scores = learning_curve(model_basic, X, y, cv=5, scoring='accuracy')

plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Cross-validation score")
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Curvas de Aprendizaje (Modelo Básico)')
plt.legend()
plt.grid()
plt.show()
