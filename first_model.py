import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Cargar el dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Nombres de las clases
class_names = ["Avión", "Auto", "Pájaro", "Gato", "Ciervo", 
               "Perro", "Rana", "Caballo", "Barco", "Camión"]

# 2️⃣ Visualizar algunas imágenes
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis("off")
plt.show()

# 3️⃣ Normalizar los datos (escalar valores de píxeles entre 0 y 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 4️⃣ Construir la red neuronal convolucional (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 clases de salida
])

# 5️ Mostrar la arquitectura del modelo
model.summary()

# 6️ Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 7️ Entrenar el modelo
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 8️ Evaluar el modelo con datos de prueba
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nPrecisión en el conjunto de prueba: {test_acc:.2f}')

# 9️ Hacer una predicción sobre una imagen de prueba
predictions = model.predict(x_test)

# Mostrar una imagen con su predicción
plt.imshow(x_test[0])
plt.title(f"Predicción: {class_names[np.argmax(predictions[0])]}")
plt.axis("off")
plt.show()
