# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt


# Cargar datos desde un archivo CSV
df = pd.read_csv("data.csv")  

# Vista inicial de los datos
print("Vista inicial de los datos:")
print(df.head())

# a. Eliminar duplicados
df = df.drop_duplicates()

# Mostrar valores faltantes por columna
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# Imputación básica: usar la media para valores numéricos y la moda para categóricos
df.fillna({
    'Precio': df['Precio'].mean(),
    'Cantidad_Vendida': df['Cantidad_Vendida'].mean(),
    'Costo': df['Costo'].mean(),
    'Ventas': df['Ventas'].mean(),
}, inplace=True)

# Normalización
scaler = MinMaxScaler()
df[['Precio', 'Cantidad_Vendida', 'Costo', 'Ventas']] = scaler.fit_transform(df[['Precio', 'Cantidad_Vendida', 'Costo', 'Ventas']])

# Estandarización
scaler = StandardScaler()
df[['Precio', 'Cantidad_Vendida', 'Costo', 'Ventas']] = scaler.fit_transform(df[['Precio', 'Cantidad_Vendida', 'Costo', 'Ventas']])

print("\nDatos después de la normalización y estandarización:")
print(df.head())

# a. Ganancia: Ventas - Costo
df['Ganancia'] = df['Ventas'] - df['Costo']

# b. Margen de ganancia: (Ganancia / Ventas)
df['Margen_Ganancia'] = df['Ganancia'] / df['Ventas']

print("\nDatos con nuevas variables:")
print(df[['Ganancia', 'Margen_Ganancia']].head())


# a. Promedio de ventas por región
promedio_ventas_region = df.groupby('Region')['Ventas'].mean().reset_index()
print("\nPromedio de ventas por región:")
print(promedio_ventas_region)

# b. Total de ventas por producto
total_ventas_producto = df.groupby('Producto')['Ventas'].sum().reset_index()
print("\nTotal de ventas por producto:")
print(total_ventas_producto)

# Detección y manejo de outliers usando el rango intercuartílico (IQR)
Q1 = df['Ganancia'].quantile(0.25)
Q3 = df['Ganancia'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

df['Ganancia_ajustada'] = df['Ganancia'].clip(lower=limite_inferior, upper=limite_superior)

print("\nDatos con manejo de outliers (Ganancia ajustada):")
print(df[['Ganancia', 'Ganancia_ajustada']].head())

# a. Graficar distribución de ganancia antes y después de ajustar outliers
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['Ganancia'], bins=20, color='blue', alpha=0.7, label='Original')
plt.title('Distribución de Ganancia (Original)')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(df['Ganancia_ajustada'], bins=20, color='green', alpha=0.7, label='Ajustada')
plt.title('Distribución de Ganancia (Ajustada)')
plt.legend()

plt.tight_layout()
plt.show()

# Exportar los datos transformados a un archivo CSV
df.to_csv("datos_transformados.csv", index=False)
print("\nArchivo 'datos_transformados.csv' generado con éxito.")
