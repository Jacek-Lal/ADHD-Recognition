from tensorflow import keras
from keras import layers
from config import *

model = keras.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=CNN_INPUT_SHAPE, padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    layers.Dense(64, activation='relu'),
    
    layers.Dense(32, activation='relu'),
    
    layers.Dense(1, activation='sigmoid')
])
