import tensorflow as tf
from tensorflow import keras
from keras import layers

from config import CNN_INPUT_SHAPE

if tf.test.is_gpu_available():

    physical_devices = tf.config.list_physical_devices('GPU')
    print("Dostępne karty graficzne:")
    for device in physical_devices:
        print(f"- {device.name}")
else:
    print("Nie znaleziono dostępnych kart graficznych. Używane będą obliczenia na CPU.")

print("---------------------------------------------------------------------------------------")


model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=CNN_INPUT_SHAPE, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(1, activation='sigmoid')
    ])