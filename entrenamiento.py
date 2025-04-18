import numpy as np
import cv2
from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Flatten
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Función para cargar y procesar las imágenes y etiquetas de clase
def cargar_imagenes_y_etiquetas(ruta,rango1,rango2):
    # Cargamos las imágenes
    imagenes = []
    etiquetas = []
    nombreClase=["letra-A","letra-B","letra-C","letra-D","letra-E","letra-F","letra-G","letra-H","letra-I","letra-J","letra-K","letra-L","letra-M","letra-N","letra-O","letra-P","letra-Q","letra-R","letra-S","letra-T","letra-U","letra-V","letra-W","letra-X","letra-Y","letra-Z"]
   
    for clase in range(0, 26):
        for i in range(rango1, rango2):
           # print(ruta + nombreClase[clase] + '/' + nombreClase[clase]+'-'+str(i) + '.jpg')
            imagen = cv2.imread(ruta + nombreClase[clase] + '/' + nombreClase[clase]+'-'+str(i) + '.jpg')
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Convertimos a escala de grises
            borde = cv2.Canny(imagen,100,200)    # Extraemos el borde
            imagen = borde.astype('float32') / 255.0 # Normalizamos los píxeles
            imagenes.append(imagen)
            etiquetas.append(clase)
   
    # Convertimos las listas de imágenes y etiquetas a arrays numpy
    imagenes = np.array(imagenes)
    etiquetas = np.array(etiquetas)  
   
       
    # Devolvemos las imágenes y etiquetas
    return imagenes, etiquetas


# Cargamos las imágenes y etiquetas de clase
ruta1 = './Recursos/Train/'
imagenesTrain, etiquetasTrain = cargar_imagenes_y_etiquetas(ruta1, 41,100)
ruta2 = './Recursos/Test/'
imagenesTest, etiquetasTest = cargar_imagenes_y_etiquetas(ruta2, 1,40)

# Definimos la arquitectura de la red neuronal
modelo = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])

# Compilamos el modelo
modelo.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

# Entrenamos el modelo
historial=modelo.fit(imagenesTrain, etiquetasTrain,
           validation_data=(imagenesTest, etiquetasTest),
           epochs=50)

modelo.save("modelo_perceptronFigG.h5")



# Graficar los errores de entrenamiento y validación
plt.plot(historial.history['loss'], label='Error de entrenamiento')
plt.plot(historial.history['val_loss'], label='Error de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (Loss)')
plt.title('Evolución del error en el entrenamiento')
plt.legend()
plt.show()